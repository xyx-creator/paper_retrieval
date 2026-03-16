from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import fitz
from langchain_core.tools import tool

from config import OUTPUT_DIR, RELEVANCE_THRESHOLD
from modules.glm_agent import GLMAgent
from modules.report_generator import generate_consolidated_report
from modules.pdf_processor import crop_specific_figure, extract_all_captions, extract_text_and_metadata
from tools.tool_schemas import AnalyzePdfInput, GenerateMarkdownReportInput
from tools.tool_schemas import ListLocalPdfsInput

_ANALYSIS_SESSIONS: Dict[str, List[Dict[str, Any]]] = {}


@tool("list_local_pdfs", args_schema=ListLocalPdfsInput)
async def list_local_pdfs_tool(
    folder_path: str = "paper",
    recursive: int | bool = False,
    max_files: int = 8,
) -> Dict[str, Any]:
    """List local PDF files under a folder for downstream analyze_pdf calls."""
    folder_abs = os.path.abspath(folder_path)
    if not os.path.isdir(folder_abs):
        return {
            "ok": False,
            "error": f"Folder not found: {folder_path}",
            "folder_path": folder_abs,
            "pdf_paths": [],
        }

    pdf_paths: List[str] = []
    recursive_flag = bool(recursive)

    if recursive_flag:
        for root, _, files in os.walk(folder_abs):
            for file_name in files:
                if file_name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, file_name))
    else:
        for file_name in os.listdir(folder_abs):
            candidate = os.path.join(folder_abs, file_name)
            if os.path.isfile(candidate) and file_name.lower().endswith(".pdf"):
                pdf_paths.append(candidate)

    pdf_paths.sort()
    pdf_paths = pdf_paths[: max(1, int(max_files))]

    return {
        "ok": True,
        "folder_path": folder_abs,
        "count": len(pdf_paths),
        "pdf_paths": pdf_paths,
    }


def _slugify_filename(text: str, max_len: int = 60) -> str:
    base = os.path.splitext(os.path.basename(text))[0]
    base = base.replace(" ", "_")
    base = re.sub(r"[^a-zA-Z0-9_]", "", base)
    return base[:max_len] if base else "paper"


def _ensure_runtime_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)


def _extract_first_page_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = doc[0].get_text("text")
        doc.close()
        return str(text)
    except Exception:
        return ""


def _is_valid_pdf_file(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as file:
            return file.read(4) == b"%PDF"
    except Exception:
        return False


def _render_first_page_preview(pdf_path: str, output_dir: str) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{_slugify_filename(pdf_path)}_preview.png")
        pix.save(output_path)
        doc.close()
        return output_path
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_agent() -> GLMAgent:
    return GLMAgent()


@tool("analyze_pdf", args_schema=AnalyzePdfInput)
async def analyze_pdf_tool(
    pdf_path: str,
    mandatory_keywords: List[str],
    bonus_keywords: List[str] | None = None,
    relevance_threshold: int = 1,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Analyze a local PDF end-to-end and cache the result for markdown report generation."""
    _ensure_runtime_dirs()

    if not os.path.exists(pdf_path):
        return {"ok": False, "error": f"PDF not found: {pdf_path}"}

    if not _is_valid_pdf_file(pdf_path):
        return {"ok": False, "error": f"Invalid PDF content: {pdf_path}"}

    agent = _get_agent()
    bonus = bonus_keywords or []

    metadata = await asyncio.to_thread(extract_text_and_metadata, pdf_path)
    title = str(metadata.get("title") or os.path.basename(pdf_path))
    abstract = str(metadata.get("abstract") or "")
    first_page_text = await asyncio.to_thread(_extract_first_page_text, pdf_path)

    if not abstract.strip():
        abstract = first_page_text[:2000]

    score = await agent.score_relevance(abstract, mandatory_keywords, bonus)
    if score < max(1, int(relevance_threshold or RELEVANCE_THRESHOLD)):
        return {
            "ok": False,
            "skipped": True,
            "title": title,
            "score": score,
            "reason": "Below relevance threshold",
        }

    captions = await asyncio.to_thread(extract_all_captions, pdf_path)
    figure_id: Optional[str] = None
    image_path: Optional[str] = None

    if captions:
        caption_candidates = [
            f"{c.get('figure_id', '')}: {str(c.get('caption_text', ''))[:200]}..."
            for c in captions
        ]
        figure_id = await agent.select_best_figure(caption_candidates)
        if figure_id:
            image_output_dir = os.path.join(OUTPUT_DIR, "images", _slugify_filename(pdf_path))
            image_path = await asyncio.to_thread(
                crop_specific_figure,
                pdf_path,
                figure_id,
                captions,
                image_output_dir,
            )

    if not image_path or not os.path.exists(image_path):
        image_output_dir = os.path.join(OUTPUT_DIR, "images", _slugify_filename(pdf_path))
        image_path = await asyncio.to_thread(_render_first_page_preview, pdf_path, image_output_dir)

    text_analysis = await agent.analyze_text_brain(abstract, introduction=first_page_text)
    vision_analysis = (
        await agent.analyze_vision_brain(image_path)
        if image_path and os.path.exists(image_path)
        else {"visible_modules": [], "visible_connections": [], "notes": "No image available."}
    )

    synthesis = await agent.synthesize_report(
        text_analysis=text_analysis,
        vision_analysis=vision_analysis,
        relevance_score=int(score),
        keywords=mandatory_keywords,
    )

    result = {
        "title": title,
        "filename": os.path.basename(pdf_path),
        "pdf_path": pdf_path,
        "score": int(score),
        "image_path": image_path or "",
        "selected_figure": figure_id or "",
        "synthesis": synthesis,
        "text_analysis": text_analysis,
        "vision_analysis": vision_analysis,
    }

    if session_id not in _ANALYSIS_SESSIONS:
        _ANALYSIS_SESSIONS[session_id] = []
    _ANALYSIS_SESSIONS[session_id].append(result)

    return {
        "ok": True,
        "session_id": session_id,
        "cached_results": len(_ANALYSIS_SESSIONS[session_id]),
        "title": result["title"],
        "score": result["score"],
        "image_path": result["image_path"],
    }


@tool("generate_markdown_report", args_schema=GenerateMarkdownReportInput)
async def generate_markdown_report_tool(
    session_id: str = "default",
    keywords: List[str] | None = None,
    output_filename: Optional[str] = None,
    clear_session_after_report: bool = False,
) -> Dict[str, Any]:
    """Generate a consolidated markdown report from cached analysis results in a session."""
    _ensure_runtime_dirs()

    session_results = list(_ANALYSIS_SESSIONS.get(session_id, []))
    if not session_results:
        return {
            "ok": False,
            "error": f"No cached analysis results in session '{session_id}'. Call analyze_pdf first.",
        }

    if output_filename:
        report_filename = output_filename if output_filename.lower().endswith(".md") else f"{output_filename}.md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"react_report_{session_id}_{timestamp}.md"

    report_path = os.path.join(OUTPUT_DIR, report_filename)
    keyword_text = ", ".join(keywords or [])

    await asyncio.to_thread(
        generate_consolidated_report,
        session_results,
        report_path,
        keyword_text,
    )

    if clear_session_after_report:
        _ANALYSIS_SESSIONS.pop(session_id, None)

    return {
        "ok": True,
        "session_id": session_id,
        "report_path": report_path,
        "paper_count": len(session_results),
    }


ANALYSIS_TOOLS = [
    list_local_pdfs_tool,
    analyze_pdf_tool,
    generate_markdown_report_tool,
]
