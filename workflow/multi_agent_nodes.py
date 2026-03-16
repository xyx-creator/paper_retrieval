"""Role-specific nodes for the Phase 3 multi-agent workflow."""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from config import ZHIPUAI_API_KEY
from tools.analysis_tools import list_local_pdfs_tool
from tools.paper_source_tools import PAPER_SOURCE_TOOLS, download_pdf_tool
from tools.pdf_tools import PDF_TOOLS, crop_specific_figure_tool, extract_all_captions_tool
from workflow.multi_agent_state import MultiAgentState


def _build_chat_model(model_name: str = "glm-4-plus", temperature: float = 0.1):
    """Build a ChatZhipuAI instance with compatible constructor args."""
    if not ZHIPUAI_API_KEY:
        raise ValueError("ZHIPUAI_API_KEY is not configured.")

    try:
        from langchain_community.chat_models import ChatZhipuAI  # type: ignore
    except ImportError as exc:
        raise ImportError("langchain-community is required to use ChatZhipuAI.") from exc

    try:
        return ChatZhipuAI(
            model=model_name,
            api_key=ZHIPUAI_API_KEY,
            temperature=temperature,
        )
    except TypeError:
        legacy_kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "zhipuai_api_key": ZHIPUAI_API_KEY,
        }
        return ChatZhipuAI(
            **legacy_kwargs,
        )


def _parse_any_payload(payload: Any) -> Any:
    if isinstance(payload, (dict, list, int, float, bool)) or payload is None:
        return payload

    text = str(payload).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _iter_tool_messages(messages: Iterable[BaseMessage], tool_name: str) -> Iterable[ToolMessage]:
    for message in messages:
        if isinstance(message, ToolMessage) and getattr(message, "name", "") == tool_name:
            yield message


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


def _extract_candidate_papers(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    # Prefer filtered output first, then enriched, then raw retrieval outputs.
    priority_tools = ["filter_by_keywords", "batch_fetch_s2", "search_arxiv", "search_dblp"]

    for tool_name in priority_tools:
        tool_msgs = list(_iter_tool_messages(messages, tool_name))
        if not tool_msgs:
            continue

        for tool_msg in reversed(tool_msgs):
            payload = _parse_any_payload(tool_msg.content)
            if isinstance(payload, list):
                dict_items = [x for x in payload if isinstance(x, dict)]
                if dict_items:
                    return _deduplicate_papers(dict_items)
            if isinstance(payload, dict):
                papers = payload.get("papers") or payload.get("results") or payload.get("data")
                if isinstance(papers, list):
                    dict_items = [x for x in papers if isinstance(x, dict)]
                    if dict_items:
                        return _deduplicate_papers(dict_items)

    return []


def _extract_downloaded_pdfs(messages: List[BaseMessage]) -> List[str]:
    paths: List[str] = []
    for tool_msg in _iter_tool_messages(messages, "download_pdf"):
        payload = _parse_any_payload(tool_msg.content)
        if isinstance(payload, str) and payload.strip():
            paths.append(payload.strip())
        elif isinstance(payload, dict):
            path = payload.get("pdf_path") or payload.get("path")
            if isinstance(path, str) and path.strip():
                paths.append(path.strip())

    # Keep existing order while removing duplicates.
    seen = set()
    deduped: List[str] = []
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def _extract_visual_assets(messages: List[BaseMessage]) -> List[str]:
    assets: List[str] = []
    for tool_msg in _iter_tool_messages(messages, "crop_specific_figure"):
        payload = _parse_any_payload(tool_msg.content)
        if isinstance(payload, str) and payload.strip():
            assets.append(payload.strip())
        elif isinstance(payload, dict):
            path = payload.get("image_path") or payload.get("output_path") or payload.get("path")
            if isinstance(path, str) and path.strip():
                assets.append(path.strip())

    seen = set()
    deduped: List[str] = []
    for asset in assets:
        norm = os.path.abspath(asset)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def _is_local_only_task(task_instruction: str) -> bool:
    text = task_instruction.lower()
    local_hints = ["local", "paper folder", "under paper", "pdf at", "local pdf", "on disk"]
    remote_hints = ["arxiv", "dblp", "semantic scholar", "s2", "online", "web"]
    has_local = any(token in text for token in local_hints)
    has_remote = any(token in text for token in remote_hints)
    return has_local and not has_remote


async def _build_local_retrieved_papers(max_files: int = 12) -> List[Dict[str, Any]]:
    payload = await list_local_pdfs_tool.ainvoke(
        {"folder_path": "paper", "recursive": False, "max_files": max_files}
    )
    if not isinstance(payload, dict):
        return []

    pdf_paths = payload.get("pdf_paths") or []
    local_papers: List[Dict[str, Any]] = []
    for pdf_path in pdf_paths:
        if not isinstance(pdf_path, str) or not pdf_path.strip():
            continue
        local_papers.append(
            {
                "title": os.path.basename(pdf_path),
                "url": "",
                "local_path": os.path.abspath(pdf_path),
                "venue": "Local",
            }
        )

    return local_papers


async def researcher_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    """Researcher role: retrieve high-quality papers using search tools only."""
    print("[Researcher Agent working...]")

    task_instruction = str(state.get("task_instruction", "")).strip()
    errors = list(state.get("errors", []))
    incoming_messages = list(state.get("messages", []))

    if _is_local_only_task(task_instruction):
        print("[Researcher] Local-only mode: using local PDFs without remote retrieval.")
        retrieved_papers = await _build_local_retrieved_papers(max_files=12)
        if not retrieved_papers:
            errors.append("[Researcher] No local PDFs found under paper/.")
        local_summary = AIMessage(
            content=(
                "Researcher selected local-only mode and prepared "
                f"{len(retrieved_papers)} local PDF candidates."
            )
        )
        return {
            "retrieved_papers": retrieved_papers,
            "messages": [*incoming_messages, local_summary],
            "errors": errors,
        }

    model = _build_chat_model(model_name="glm-4-plus", temperature=0.1)
    researcher_prompt = (
        "You are a senior academic researcher. "
        "Use only the bound retrieval tools to find high-quality papers relevant to the user instruction. "
        "Prefer fresh and reliable sources. "
        "If enough data exists, call filter_by_keywords before final answer."
    )

    researcher_agent = create_react_agent(model, PAPER_SOURCE_TOOLS, prompt=researcher_prompt)

    user_message = HumanMessage(
        content=(
            "User task:\n"
            f"{task_instruction}\n\n"
            "Goal:\n"
            "1) retrieve candidate papers,\n"
            "2) enrich when needed,\n"
            "3) provide a concise handoff summary for downstream vision processing."
        )
    )

    result = await researcher_agent.ainvoke({"messages": [user_message]})
    messages = list(result.get("messages", []))
    retrieved_papers = _extract_candidate_papers(messages)

    if not retrieved_papers:
        errors.append("[Researcher] No papers retrieved from tool outputs.")

    return {
        "retrieved_papers": retrieved_papers,
        "messages": messages,
        "errors": errors,
    }


async def vision_expert_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    """Vision expert role: download PDFs and extract architecture figure assets."""
    print("[Vision Expert Agent working...]")

    task_instruction = str(state.get("task_instruction", "")).strip()
    errors = list(state.get("errors", []))
    retrieved_papers = list(state.get("retrieved_papers", []))
    incoming_messages = list(state.get("messages", []))
    local_only = _is_local_only_task(task_instruction)

    if not retrieved_papers:
        return {
            "downloaded_pdfs": [],
            "visual_assets": [],
            "messages": incoming_messages,
            "errors": errors,
        }

    local_pdf_paths = [
        os.path.abspath(str(paper.get("local_path")))
        for paper in retrieved_papers
        if str(paper.get("local_path", "")).strip() and os.path.exists(str(paper.get("local_path")))
    ]

    vision_messages: List[BaseMessage] = []
    downloaded_pdfs: List[str] = []
    visual_assets: List[str] = []

    if local_only and local_pdf_paths:
        print("[Vision Expert] Local-only mode: processing local PDF files directly.")
        downloaded_pdfs = list(dict.fromkeys(local_pdf_paths[:5]))
    else:
        model = _build_chat_model(model_name="glm-4-plus", temperature=0.1)
        vision_prompt = (
            "You are a vision parsing expert. "
            "Use only the bound tools to download PDFs and crop architecture/overview diagrams. "
            "Prioritize Figure 1 or architecture/overview figures when possible."
        )

        vision_agent_tools = [*PDF_TOOLS, download_pdf_tool]
        vision_agent = create_react_agent(model, vision_agent_tools, prompt=vision_prompt)

        papers_brief = [
            {
                "title": paper.get("title", ""),
                "url": paper.get("url", ""),
                "local_path": paper.get("local_path", ""),
            }
            for paper in retrieved_papers[:5]
        ]

        vision_message = HumanMessage(
            content=(
                "User task:\n"
                f"{task_instruction}\n\n"
                "Input papers (top 5):\n"
                f"{json.dumps(papers_brief, ensure_ascii=False)}\n\n"
                "For each paper, if local_path exists use it directly; otherwise attempt download_pdf.\n"
                "Then run extract_all_captions and crop_specific_figure (prefer Figure 1 if present).\n"
                "Finally summarize completed assets."
            )
        )

        result = await vision_agent.ainvoke({"messages": [vision_message]})
        vision_messages = list(result.get("messages", []))

        downloaded_pdfs = _extract_downloaded_pdfs(vision_messages)
        visual_assets = _extract_visual_assets(vision_messages)
        downloaded_pdfs = list(dict.fromkeys([*local_pdf_paths[:5], *downloaded_pdfs]))

    # Deterministic fallback: if no figure was cropped, try one pass per downloaded pdf.
    if downloaded_pdfs and not visual_assets:
        for pdf_path in downloaded_pdfs[:3]:
            try:
                captions = await extract_all_captions_tool.ainvoke({"pdf_path": pdf_path})
                if not captions:
                    continue

                figure_id = "Figure 1"
                caption_ids = {
                    str(item.get("figure_id", "")).strip(): item for item in captions if isinstance(item, dict)
                }
                if figure_id not in caption_ids:
                    figure_id = str(captions[0].get("figure_id", "Figure 1"))

                image_path = await crop_specific_figure_tool.ainvoke(
                    {
                        "pdf_path": pdf_path,
                        "target_figure_id": figure_id,
                        "captions": captions,
                    }
                )
                if isinstance(image_path, str) and image_path.strip() and os.path.exists(image_path):
                    visual_assets.append(os.path.abspath(image_path))
            except Exception as exc:
                errors.append(f"[Vision Expert] fallback crop failed for {pdf_path}: {exc}")

    if not downloaded_pdfs:
        errors.append("[Vision Expert] No PDFs downloaded from tool outputs.")

    merged_messages = [*incoming_messages, *vision_messages]

    return {
        "downloaded_pdfs": downloaded_pdfs,
        "visual_assets": list(dict.fromkeys(visual_assets)),
        "messages": merged_messages,
        "errors": errors,
    }


async def writer_agent_node(state: MultiAgentState) -> Dict[str, Any]:
    """Writer role: produce final markdown report without external tools."""
    print("[Writer Agent working...]")

    task_instruction = str(state.get("task_instruction", "")).strip()
    retrieved_papers = list(state.get("retrieved_papers", []))
    downloaded_pdfs = list(state.get("downloaded_pdfs", []))
    visual_assets = list(state.get("visual_assets", []))
    errors = list(state.get("errors", []))
    incoming_messages = list(state.get("messages", []))

    model = _build_chat_model(model_name="glm-4-plus", temperature=0.2)

    system = SystemMessage(
        content=(
            "You are a senior scientific writer. "
            "Write a concise, high-signal markdown report using only provided data. "
            "Do not fabricate unavailable details."
        )
    )
    prompt = HumanMessage(
        content=(
            "Create a markdown report for this task:\n"
            f"{task_instruction}\n\n"
            "Available inputs:\n"
            f"- Retrieved papers (count={len(retrieved_papers)}): {json.dumps(retrieved_papers[:10], ensure_ascii=False)}\n"
            f"- Downloaded PDFs: {json.dumps(downloaded_pdfs, ensure_ascii=False)}\n"
            f"- Visual assets: {json.dumps(visual_assets, ensure_ascii=False)}\n"
            f"- Errors: {json.dumps(errors, ensure_ascii=False)}\n\n"
            "Output requirements:\n"
            "1) Title and short executive summary\n"
            "2) Paper shortlist table (title, source/url if available)\n"
            "3) Visual evidence section listing image paths\n"
            "4) Key findings and limitations\n"
            "5) Next-step recommendations"
        )
    )

    response = await model.ainvoke([system, prompt])
    content = response.content if hasattr(response, "content") else str(response)
    final_report = str(content).strip()

    writer_message = AIMessage(content=final_report)
    merged_messages = [*incoming_messages, writer_message]

    return {
        "final_report": final_report,
        "messages": merged_messages,
        "errors": errors,
    }
