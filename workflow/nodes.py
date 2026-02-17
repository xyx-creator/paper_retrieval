"""LangGraph 节点实现。

设计原则：
1. 编排层使用 LangGraph，核心处理逻辑仍复用原生模块；
2. 所有 IO 密集流程保持并发（检索/下载/打分/深度分析）；
3. 节点尽量返回“最小增量状态”，降低状态污染风险。
"""

from __future__ import annotations

import concurrent.futures
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
    """延迟初始化 GLM Agent，避免图编译阶段触发模型依赖。"""

    return GLMAgent()


def _ensure_runtime_dirs() -> None:
    """确保运行期目录存在。"""

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)


def _deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 title + url 去重，减少后续重复打分与重复下载。"""

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
    """将论文标题/文件名转成安全目录名。"""

    text = os.path.splitext(text)[0]
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-zA-Z0-9_]", "", text)
    return text[:max_len] if text else "paper"


def _extract_fallback_intro_and_abstract(pdf_path: str) -> Tuple[str, str]:
    """当 metadata 抽取不足时，从首页提取 fallback 文本。"""

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
    """用文件头快速判断是否为有效 PDF。"""

    try:
        with open(file_path, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def _build_report_filename(source: str, user_query: Dict[str, Any]) -> str:
    """生成报告文件名，保持与原工程命名习惯接近。"""

    date_str = datetime.now().strftime("%Y-%m-%d")
    mandatory = user_query.get("mandatory_keywords") or []
    first_kw = mandatory[0].replace(" ", "-") if mandatory else "Papers"

    if source == "dblp" and user_query.get("venue") and user_query.get("year"):
        source_tag = f"{user_query['venue']}{user_query['year']}"
    else:
        source_tag = source.upper()
    return f"{source_tag}_{first_kw}_{date_str}.md"


def keyword_expansion_node(state: GraphState) -> Dict[str, Any]:
    """Node 1: Keyword Expansion。"""

    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))

    mandatory = user_query.get("mandatory_keywords") or list(KEYWORDS.get("mandatory", []))
    bonus = user_query.get("bonus_keywords")
    if bonus is None:
        bonus = list(KEYWORDS.get("bonus", []))

    try:
        agent = _get_agent()
        expanded_mandatory = agent.expand_keywords_batch(mandatory, mode="strict")
    except Exception as exc:
        # 关键词扩展失败时，回退为“原词本身”，保证后续流程可继续。
        expanded_mandatory = {k: [k] for k in mandatory}
        errors.append(f"[Keyword Expansion] failed: {exc}")

    # 与阶段一约定保持一致：bonus 默认不做扩展，直接透传。
    expanded_bonus = list(bonus)

    user_query["mandatory_keywords"] = mandatory
    user_query["bonus_keywords"] = expanded_bonus

    return {
        "user_query": user_query,
        "expanded_mandatory": expanded_mandatory,
        "expanded_bonus": expanded_bonus,
        "errors": errors,
    }


def information_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """Node 2: Information Retrieval。

    并行策略说明：
    - 对于 `source=all`，并行触发 arXiv 与 DBLP->S2 两条检索链；
    - 对于单一来源，仍复用同一套工具调用逻辑；
    - DBLP->S2 必须串行（后者依赖前者结果），但该链可以与其他来源并行。
    """

    _ensure_runtime_dirs()
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))

    source = str(user_query.get("source", "local")).lower()
    days = int(user_query.get("days", 1))
    venue = user_query.get("venue")
    year = user_query.get("year")
    max_local_papers = int(user_query.get("max_local_papers", 15))

    # Local 模式直接读取本地 PDF 列表，不走网络检索。
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

    def fetch_arxiv_chain() -> List[Dict[str, Any]]:
        mandatory = user_query.get("mandatory_keywords", [])
        return search_arxiv_tool.invoke(
            {"query_keywords": mandatory, "days": days}
        )

    def fetch_dblp_s2_chain() -> List[Dict[str, Any]]:
        if not venue or not year:
            raise ValueError("DBLP 检索需要 venue 与 year。")
        dblp_hits = search_dblp_tool.invoke({"venue": str(venue), "year": int(year)})
        return batch_fetch_s2_tool.invoke({"papers_data": dblp_hits})

    candidate_papers: List[Dict[str, Any]] = []
    futures: Dict[concurrent.futures.Future, str] = {}

    # 统一并发入口：即使只有一条链路也用同样的执行结构，便于扩展。
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        if source in {"arxiv", "all"}:
            futures[executor.submit(fetch_arxiv_chain)] = "arxiv"
        if source in {"dblp", "all"}:
            futures[executor.submit(fetch_dblp_s2_chain)] = "dblp_s2"

        if not futures:
            errors.append(f"[Information Retrieval] unsupported source: {source}")

        for future in concurrent.futures.as_completed(futures):
            chain_name = futures[future]
            try:
                papers = future.result()
                candidate_papers.extend(papers or [])
            except Exception as exc:
                errors.append(f"[Information Retrieval:{chain_name}] failed: {exc}")

    candidate_papers = _deduplicate_papers(candidate_papers)
    return {"candidate_papers": candidate_papers, "errors": errors}


def filter_scoring_node(state: GraphState) -> Dict[str, Any]:
    """Node 3: Filter & Scoring。

    并行策略说明：
    - 先进行轻量关键词过滤（同步）；
    - 再对过滤后论文执行并行摘要打分（线程池）。
    """

    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))
    source = str(user_query.get("source", "local")).lower()

    candidate_papers = list(state.get("candidate_papers", []))
    expanded_mandatory = dict(state.get("expanded_mandatory", {}))
    expanded_bonus = list(state.get("expanded_bonus", []))

    # Local 模式不做 metadata 过滤/打分，留到深度分析时基于 PDF 内容再评估。
    if source == "local":
        return {
            "filtered_papers": candidate_papers,
            "scored_papers": candidate_papers,
            "errors": errors,
        }

    if not candidate_papers:
        return {"filtered_papers": [], "scored_papers": [], "errors": errors}

    try:
        filtered_papers = filter_by_keywords_tool.invoke(
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
    max_workers = int(user_query.get("max_workers", 5))
    top_k = int(user_query.get("top_k", 10))
    mandatory = user_query.get("mandatory_keywords", [])
    bonus = user_query.get("bonus_keywords", [])

    def score_single(paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        abstract = str(paper.get("abstract", ""))
        score = agent.score_relevance(abstract, mandatory, bonus)
        if score < threshold:
            return None
        enriched = dict(paper)
        enriched["score"] = score
        return enriched

    scored_papers: List[Dict[str, Any]] = []
    # 评分是典型 IO-bound（模型请求），线程池可以显著降低总耗时。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(score_single, paper): paper for paper in filtered_papers
        }
        for future in concurrent.futures.as_completed(future_to_paper):
            paper = future_to_paper[future]
            try:
                result = future.result()
                if result:
                    scored_papers.append(result)
            except Exception as exc:
                title = str(paper.get("title", ""))[:80]
                errors.append(f"[Scoring] failed on `{title}`: {exc}")

    scored_papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    scored_papers = scored_papers[:top_k]
    return {
        "filtered_papers": filtered_papers,
        "scored_papers": scored_papers,
        "errors": errors,
    }


def _process_single_paper(
    paper_info: Dict[str, Any],
    user_query: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Node4 的单论文处理函数（供线程池并行调用）。

    线程安全说明：
    - 本函数不写共享全局变量；
    - 每个线程只处理一个 paper_info，并返回结果给主线程汇总；
    - 避免在子线程直接 append 共享列表，从根源减少锁竞争和竞态。
    """

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
            pdf_path = download_pdf_tool.invoke({"url": str(url), "save_dir": INPUT_DIR})

        if not pdf_path or not os.path.exists(pdf_path):
            return None, f"[Deep Analysis] `{title_for_log}` download/open pdf failed."

        # 防止历史缓存中存在“扩展名是 .pdf 但实际是 HTML”的脏文件。
        # 若当前论文有 URL，则尝试重新下载一次；否则直接跳过并记录原因。
        if not _is_valid_pdf_file(pdf_path):
            paper_url = paper_info.get("url")
            if paper_url:
                re_downloaded = download_pdf_tool.invoke({"url": str(paper_url), "save_dir": INPUT_DIR})
                if re_downloaded and os.path.exists(re_downloaded) and _is_valid_pdf_file(re_downloaded):
                    pdf_path = re_downloaded
                else:
                    return None, f"[Deep Analysis] `{title_for_log}` invalid PDF content (non-PDF body)."
            else:
                return None, f"[Deep Analysis] `{title_for_log}` invalid local PDF content."

        filename = os.path.basename(pdf_path)

        # 1) 文本元数据抽取（调用工具封装，底层仍是原生实现）。
        raw_metadata = extract_text_and_metadata_tool.invoke({"pdf_path": pdf_path})
        title = raw_metadata.get("title") or filename
        abstract = raw_metadata.get("abstract") or str(paper_info.get("abstract", ""))

        # 当摘要不足时，回退首页文本，提升后续打分与文本脑稳定性。
        first_page_text = ""
        if not abstract or len(abstract) < 100:
            first_page_text, fallback_abstract = _extract_fallback_intro_and_abstract(pdf_path)
            if fallback_abstract:
                abstract = fallback_abstract
        else:
            first_page_text, _ = _extract_fallback_intro_and_abstract(pdf_path)

        # 2) 相关性分数：在线检索场景复用 Node3 结果；本地模式在这里补算。
        score = paper_info.get("score")
        if score is None:
            score = agent.score_relevance(abstract, mandatory, bonus)
        score = int(score)
        if score < threshold:
            return None, None

        # 3) 图注抽取 + 结构图选择。
        captions = extract_all_captions_tool.invoke({"pdf_path": pdf_path})
        if not captions:
            return None, None

        captions_text_list = [
            f"{c.get('figure_id', '')}: {str(c.get('caption_text', ''))[:200]}..."
            for c in captions
        ]
        best_figure_id = agent.select_best_figure(captions_text_list)
        if not best_figure_id:
            return None, f"[Deep Analysis] `{title_for_log}` no suitable figure selected."

        # 4) 构造论文专属图像目录，避免不同论文输出冲突。
        paper_slug = _slugify_filename(filename)
        paper_image_dir = os.path.join(OUTPUT_DIR, "images", paper_slug)
        os.makedirs(paper_image_dir, exist_ok=True)

        image_path = crop_specific_figure_tool.invoke(
            {
                "pdf_path": pdf_path,
                "target_figure_id": best_figure_id,
                "captions": captions,
                "output_dir": paper_image_dir,
            }
        )
        if not image_path or not os.path.exists(image_path):
            return None, f"[Deep Analysis] `{title_for_log}` figure crop failed for {best_figure_id}."

        # 5) 双脑分析 + 融合报告。
        text_analysis = agent.analyze_text_brain(abstract, introduction=first_page_text)
        vision_analysis = agent.analyze_vision_brain(image_path)
        synthesis = agent.synthesize_report(
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


def deep_analysis_node(state: GraphState) -> Dict[str, Any]:
    """Node 4: Deep Analysis (Dual-Brain)。"""

    _ensure_runtime_dirs()
    user_query = dict(state.get("user_query", {}))
    errors = list(state.get("errors", []))
    source = str(user_query.get("source", "local")).lower()

    papers_to_process = list(state.get("scored_papers", []))
    if source == "local" and not papers_to_process:
        papers_to_process = list(state.get("candidate_papers", []))

    if not papers_to_process:
        return {"processed_papers": [], "errors": errors}

    max_workers = int(user_query.get("max_workers", 5))
    processed_papers: List[Dict[str, Any]] = []

    # 多论文并行处理：每个 future 对应一篇论文的完整深度分析流水线。
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_paper = {
            executor.submit(_process_single_paper, paper, user_query): paper
            for paper in papers_to_process
        }

        for future in concurrent.futures.as_completed(future_to_paper):
            try:
                result, err = future.result()
                if result:
                    processed_papers.append(result)
                if err:
                    errors.append(err)
            except Exception as exc:
                title = str(future_to_paper[future].get("title", "Unknown Title"))
                errors.append(f"[Deep Analysis] `{title}` thread failed: {exc}")

    processed_papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"processed_papers": processed_papers, "errors": errors}


def report_generation_node(state: GraphState) -> Dict[str, Any]:
    """Node 5: Report Generation。"""

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
        generate_consolidated_report(
            results_list=processed_papers,
            output_path=report_path,
            keywords=keywords_text,
        )
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                final_report = f.read()
        except Exception:
            final_report = ""
    else:
        # 空结果也输出报告文件，便于自动化任务统一消费。
        final_report = (
            "# Paper Analysis Report\n\n"
            f"**Keywords:** {keywords_text or 'N/A'}\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
            "## Result\n\n"
            "未筛选到满足条件的论文，或深度分析阶段未产生有效结果。\n"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

    return {
        "final_report": final_report,
        "report_path": report_path,
        "errors": errors,
    }
