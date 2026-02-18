from __future__ import annotations

import asyncio
import glob
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import fitz

from config import INPUT_DIR, KEYWORDS, OUTPUT_DIR, RELEVANCE_THRESHOLD
from modules.glm_agent import GLMAgent
from modules.report_generator import generate_consolidated_report
from tools.paper_source_tools import (
    batch_fetch_s2_tool,
    download_pdf_tool,
    filter_by_keywords_tool,
    search_arxiv_tool,
    search_dblp_tool,
)
from tools.pdf_tools import (
    crop_specific_figure_tool,
    extract_all_captions_tool,
    extract_text_and_metadata_tool,
)
from workflow.state import GraphState


@lru_cache(maxsize=1)
def _get_agent() -> GLMAgent:
    return GLMAgent()


def _ensure_runtime_dirs() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)


def _deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for paper in papers:
        title = str(paper.get("title", "")).strip().lower()
        url = str(paper.get("url", "")).strip().lower()
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


def _slugify_filename(text: str, max_len: int = 50) -> str:
    text = os.path.splitext(text)[0]
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)
    return text[:max_len] if text else "paper"


def _extract_fallback_intro_and_abstract(pdf_path: str) -> Tuple[str, str]:
    intro = ""
    fallback_abstract = ""
    try:
        doc = fitz.open(pdf_path)
        intro = doc[0].get_text("text")
        doc.close()
        fallback_abstract = intro[:2000]
    except Exception:
        pass
    return intro, fallback_abstract


def _is_valid_pdf_file(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as file:
            return file.read(4) == b"%PDF"
    except Exception:
        return False


def _build_report_filename(source: str, user_query: Dict[str, Any]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    mandatory = user_query.get("mandatory_keywords") or []
    first_kw = mandatory[0].replace(" ", "-") if mandatory else "Papers"

    if source == "dblp" and user_query.get("venue") and user_query.get("year"):
        source_tag = f"{user_query['venue']}{user_query['year']}"
    else:
        source_tag = source.upper()
    return f"{source_tag}_{first_kw}_{date_str}.md"


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


async def keyword_expansion_node(state: GraphState) -> Dict[str, Any]:
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))

    mandatory = user_query.get("mandatory_keywords") or list(KEYWORDS.get("mandatory", []))
    bonus = user_query.get("bonus_keywords")
    if bonus is None:
        bonus = list(KEYWORDS.get("bonus", []))

    try:
        agent = _get_agent()
        expanded_mandatory = await agent.expand_keywords_batch(mandatory, mode="strict")
    except Exception as exc:
        expanded_mandatory = {k: [k] for k in mandatory}
        errors.append(f"[Keyword Expansion] failed: {exc}")

    expanded_bonus = list(bonus)
    user_query["mandatory_keywords"] = mandatory
    user_query["bonus_keywords"] = expanded_bonus

    return {
        "user_query": user_query,
        "expanded_mandatory": expanded_mandatory,
        "expanded_bonus": expanded_bonus,
        "errors": errors,
    }


async def information_retrieval_node(state: GraphState) -> Dict[str, Any]:
    _ensure_runtime_dirs()
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))

    source = str(user_query.get("source", "local")).lower()
    days = int(user_query.get("days", 1))
    venue = user_query.get("venue")
    year = user_query.get("year")
    max_local_papers = int(user_query.get("max_local_papers", 15))

    if source == "local":
        pdf_files = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
        pdf_files.sort(key=os.path.getmtime, reverse=True)
        candidate_papers = [
            {
                "local_path": pdf_path,
                "title": os.path.basename(pdf_path),
                "abstract": "",
                "venue": "Local",
            }
            for pdf_path in pdf_files[:max_local_papers]
        ]
        return {"candidate_papers": candidate_papers, "errors": errors}

    async def fetch_arxiv_chain() -> List[Dict[str, Any]]:
        mandatory = user_query.get("mandatory_keywords", [])
        return await search_arxiv_tool.ainvoke(
            {"query_keywords": mandatory, "days": days}
        )

    async def fetch_dblp_s2_chain() -> List[Dict[str, Any]]:
        if not venue or not year:
            raise ValueError("DBLP retrieval requires both venue and year.")
        dblp_hits = await search_dblp_tool.ainvoke(
            {"venue": str(venue), "year": int(year)}
        )
        return await batch_fetch_s2_tool.ainvoke({"papers_data": dblp_hits})

    chain_tasks: List[Tuple[str, Any]] = []
    if source in {"arxiv", "all"}:
        chain_tasks.append(("arxiv", fetch_arxiv_chain()))
    if source in {"dblp", "all"}:
        chain_tasks.append(("dblp_s2", fetch_dblp_s2_chain()))

    candidate_papers: List[Dict[str, Any]] = []
    if not chain_tasks:
        errors.append(f"[Information Retrieval] unsupported source: {source}")
    else:
        results = await asyncio.gather(
            *(task for _, task in chain_tasks),
            return_exceptions=True,
        )
        for (chain_name, _), result in zip(chain_tasks, results):
            if isinstance(result, Exception):
                errors.append(f"[Information Retrieval:{chain_name}] failed: {result}")
                continue
            candidate_papers.extend(result or [])

    candidate_papers = _deduplicate_papers(candidate_papers)
    return {"candidate_papers": candidate_papers, "errors": errors}


async def filter_scoring_node(state: GraphState) -> Dict[str, Any]:
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))
    source = str(user_query.get("source", "local")).lower()

    candidate_papers = list(state.get("candidate_papers", []))
    expanded_mandatory = dict(state.get("expanded_mandatory", {}))
    expanded_bonus = list(state.get("expanded_bonus", []))

    if source == "local":
        return {
            "filtered_papers": candidate_papers,
            "scored_papers": candidate_papers,
            "errors": errors,
        }

    if not candidate_papers:
        return {"filtered_papers": [], "scored_papers": [], "errors": errors}

    try:
        filtered_papers = await filter_by_keywords_tool.ainvoke(
            {
                "papers": candidate_papers,
                "expanded_mandatory": expanded_mandatory,
                "expanded_bonus": expanded_bonus,
            }
        )
    except Exception as exc:
        errors.append(f"[Filter] failed: {exc}")
        filtered_papers = []

    if not filtered_papers:
        return {"filtered_papers": [], "scored_papers": [], "errors": errors}

    try:
        agent = _get_agent()
    except Exception as exc:
        errors.append(f"[Scoring] GLM agent init failed: {exc}")
        return {
            "filtered_papers": filtered_papers,
            "scored_papers": [],
            "errors": errors,
        }

    threshold = int(user_query.get("relevance_threshold", RELEVANCE_THRESHOLD))
    max_workers = max(1, int(user_query.get("max_workers", 5)))
    top_k = int(user_query.get("top_k", 10))
    mandatory = user_query.get("mandatory_keywords", [])
    bonus = user_query.get("bonus_keywords", [])
    semaphore = asyncio.Semaphore(max_workers)

    async def score_single(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with semaphore:
            abstract = str(paper.get("abstract", ""))
            score = await agent.score_relevance(abstract, mandatory, bonus)
            if score < threshold:
                return None
            enriched = dict(paper)
            enriched["score"] = score
            return enriched

    scored_papers: List[Dict[str, Any]] = []
    outcomes = await asyncio.gather(
        *(score_single(paper) for paper in filtered_papers),
        return_exceptions=True,
    )
    for paper, outcome in zip(filtered_papers, outcomes):
        if isinstance(outcome, Exception):
            title = str(paper.get("title", ""))[:80]
            errors.append(f"[Scoring] failed on `{title}`: {outcome}")
            continue
        if outcome:
            scored_papers.append(outcome)

    scored_papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    scored_papers = scored_papers[:top_k]
    return {
        "filtered_papers": filtered_papers,
        "scored_papers": scored_papers,
        "errors": errors,
    }


async def _process_single_paper(
    paper_info: Dict[str, Any],
    user_query: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    title_for_log = str(paper_info.get("title", "Unknown Title"))

    try:
        agent = _get_agent()
        threshold = int(user_query.get("relevance_threshold", RELEVANCE_THRESHOLD))
        mandatory = user_query.get("mandatory_keywords", [])
        bonus = user_query.get("bonus_keywords", [])

        pdf_path = paper_info.get("local_path")
        if not pdf_path:
            url = paper_info.get("url")
            if not url:
                return None, f"[Deep Analysis] `{title_for_log}` missing PDF url."
            pdf_path = await download_pdf_tool.ainvoke(
                {"url": str(url), "save_dir": INPUT_DIR}
            )

        if not pdf_path or not os.path.exists(pdf_path):
            return None, f"[Deep Analysis] `{title_for_log}` download/open pdf failed."

        is_valid_pdf = await asyncio.to_thread(_is_valid_pdf_file, pdf_path)
        if not is_valid_pdf:
            paper_url = paper_info.get("url")
            if paper_url:
                re_downloaded = await download_pdf_tool.ainvoke(
                    {"url": str(paper_url), "save_dir": INPUT_DIR}
                )
                if (
                    re_downloaded
                    and os.path.exists(re_downloaded)
                    and await asyncio.to_thread(_is_valid_pdf_file, re_downloaded)
                ):
                    pdf_path = re_downloaded
                else:
                    return (
                        None,
                        f"[Deep Analysis] `{title_for_log}` invalid PDF content (non-PDF body).",
                    )
            else:
                return None, f"[Deep Analysis] `{title_for_log}` invalid local PDF content."

        filename = os.path.basename(pdf_path)

        raw_metadata = await extract_text_and_metadata_tool.ainvoke({"pdf_path": pdf_path})
        title = raw_metadata.get("title") or filename
        abstract = raw_metadata.get("abstract") or str(paper_info.get("abstract", ""))

        first_page_text = ""
        if not abstract or len(abstract) < 100:
            first_page_text, fallback_abstract = await asyncio.to_thread(
                _extract_fallback_intro_and_abstract,
                pdf_path,
            )
            if fallback_abstract:
                abstract = fallback_abstract
        else:
            first_page_text, _ = await asyncio.to_thread(
                _extract_fallback_intro_and_abstract,
                pdf_path,
            )

        score = paper_info.get("score")
        if score is None:
            score = await agent.score_relevance(abstract, mandatory, bonus)
        score = int(score)
        if score < threshold:
            return None, None

        captions = await extract_all_captions_tool.ainvoke({"pdf_path": pdf_path})
        if not captions:
            return None, None

        captions_text_list = [
            f"{c.get('figure_id', '')}: {str(c.get('caption_text', ''))[:200]}..."
            for c in captions
        ]
        best_figure_id = await agent.select_best_figure(captions_text_list)
        if not best_figure_id:
            return None, f"[Deep Analysis] `{title_for_log}` no suitable figure selected."

        paper_slug = _slugify_filename(filename)
        paper_image_dir = os.path.join(OUTPUT_DIR, "images", paper_slug)
        os.makedirs(paper_image_dir, exist_ok=True)

        image_path = await crop_specific_figure_tool.ainvoke(
            {
                "pdf_path": pdf_path,
                "target_figure_id": best_figure_id,
                "captions": captions,
                "output_dir": paper_image_dir,
            }
        )
        if not image_path or not os.path.exists(image_path):
            return (
                None,
                f"[Deep Analysis] `{title_for_log}` figure crop failed for {best_figure_id}.",
            )

        text_analysis = await agent.analyze_text_brain(
            abstract,
            introduction=first_page_text,
        )
        vision_analysis = await agent.analyze_vision_brain(image_path)
        synthesis = await agent.synthesize_report(
            text_analysis=text_analysis,
            vision_analysis=vision_analysis,
            relevance_score=score,
            keywords=mandatory,
        )

        result = {
            "title": title,
            "filename": filename,
            "score": score,
            "image_path": image_path,
            "synthesis": synthesis,
            "text_analysis": text_analysis,
            "vision_analysis": vision_analysis,
        }
        return result, None
    except Exception as exc:
        return None, f"[Deep Analysis] `{title_for_log}` failed: {exc}"


async def deep_analysis_node(state: GraphState) -> Dict[str, Any]:
    _ensure_runtime_dirs()
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))
    source = str(user_query.get("source", "local")).lower()

    papers_to_process = list(state.get("scored_papers", []))
    if source == "local" and not papers_to_process:
        papers_to_process = list(state.get("candidate_papers", []))

    if not papers_to_process:
        return {"processed_papers": [], "errors": errors}

    max_workers = max(1, int(user_query.get("max_workers", 5)))
    semaphore = asyncio.Semaphore(max_workers)
    processed_papers: List[Dict[str, Any]] = []

    async def process_with_limit(paper: Dict[str, Any]):
        async with semaphore:
            return await _process_single_paper(paper, user_query)

    outcomes = await asyncio.gather(
        *(process_with_limit(paper) for paper in papers_to_process),
        return_exceptions=True,
    )

    for paper, outcome in zip(papers_to_process, outcomes):
        if isinstance(outcome, Exception):
            title = str(paper.get("title", "Unknown Title"))
            errors.append(f"[Deep Analysis] `{title}` task failed: {outcome}")
            continue

        result, err = outcome
        if result:
            processed_papers.append(result)
        if err:
            errors.append(err)

    processed_papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"processed_papers": processed_papers, "errors": errors}


async def report_generation_node(state: GraphState) -> Dict[str, Any]:
    _ensure_runtime_dirs()
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))
    source = str(user_query.get("source", "local")).lower()

    processed_papers = list(state.get("processed_papers", []))
    mandatory = user_query.get("mandatory_keywords", [])
    keywords_text = ", ".join(mandatory)

    report_filename = _build_report_filename(source, user_query)
    report_path = os.path.join(OUTPUT_DIR, report_filename)

    if processed_papers:
        await asyncio.to_thread(
            generate_consolidated_report,
            processed_papers,
            report_path,
            keywords_text,
        )
        try:
            final_report = await asyncio.to_thread(_read_text_file, report_path)
        except Exception:
            final_report = ""
    else:
        final_report = (
            "# Paper Analysis Report\n\n"
            f"**Keywords:** {keywords_text or 'N/A'}\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
            "## Result\n\n"
            "No papers passed filtering, or deep analysis produced no usable output.\n"
        )
        await asyncio.to_thread(_write_text_file, report_path, final_report)

    return {
        "final_report": final_report,
        "report_path": report_path,
        "errors": errors,
    }

