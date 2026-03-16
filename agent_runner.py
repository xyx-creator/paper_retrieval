"""Dynamic ReAct agent runner with tool calling."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from config import ZHIPUAI_API_KEY
from tools.analysis_tools import generate_markdown_report_tool
from tools import ALL_TOOLS


REACT_SYSTEM_PROMPT = (
    "You are a paper retrieval and analysis assistant with tool-calling. "
    "When the user asks for analysis, summary, or markdown report, you must continue after retrieval/download: "
    "(1) call analyze_pdf for each downloaded/local PDF, "
    "(2) then call generate_markdown_report, "
    "(3) finally return the report path and a concise result summary. "
    "If a downloaded file path is needed, extract it from download_pdf observations. "
    "If the user asks for local folder analysis, call list_local_pdfs first and then analyze each returned pdf_path. "
    "For local folder requests, analyze a small batch first (default up to 5 files) unless the user explicitly asks to process all files. "
    "Do not call remote search tools (search_arxiv/search_dblp) for local-only requests unless the user explicitly asks remote retrieval."
)


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
        return ChatZhipuAI(
            model=model_name,
            zhipuai_api_key=ZHIPUAI_API_KEY,
            temperature=temperature,
        )


def _safe_preview(payload: Any, limit: int = 500) -> str:
    text = str(payload)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated]"


def _iter_new_messages(
    all_messages: Iterable[BaseMessage],
    emitted_ids: set[str],
) -> Iterable[BaseMessage]:
    for message in all_messages:
        message_id = getattr(message, "id", None)
        stable_id = message_id or f"{type(message).__name__}:{hash(str(message))}"
        if stable_id in emitted_ids:
            continue
        emitted_ids.add(stable_id)
        yield message


async def process_user_query(query: str) -> Dict[str, Any]:
    """Run a single natural-language query through a ReAct agent and stream progress."""
    model_name = "glm-4-plus"
    model = _build_chat_model(model_name=model_name, temperature=0.1)
    agent_executor = create_react_agent(model, ALL_TOOLS, prompt=REACT_SYSTEM_PROMPT)

    print("Initializing ReAct tool-calling agent...")
    print(f"[Model] {model_name}")
    print(f"[Tools] {len(ALL_TOOLS)} registered")
    print(f"\n[User Query] {query}\n")

    emitted_ids: set[str] = set()
    final_answer: Optional[str] = None
    called_tool_names: set[str] = set()
    analyzed_session_id: Optional[str] = None

    report_filename_match = re.search(r"named\s+([A-Za-z0-9_.-]+\.md)", query, re.IGNORECASE)
    requested_report_filename = report_filename_match.group(1) if report_filename_match else None

    async for state in agent_executor.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values",
    ):
        messages = state.get("messages", [])
        for message in _iter_new_messages(messages, emitted_ids):
            if isinstance(message, AIMessage):
                if getattr(message, "tool_calls", None):
                    thought = (message.content or "").strip()
                    if thought:
                        print(f"[Thought] {thought}")
                    for call in message.tool_calls:
                        tool_name = call.get("name", "unknown_tool")
                        args = call.get("args", {})
                        if tool_name == "analyze_pdf" and isinstance(args, dict):
                            analyzed_session_id = str(args.get("session_id", analyzed_session_id or "default"))
                        print(f"[Action] {tool_name}({ _safe_preview(call.get('args', {}), 240) })")
                else:
                    content = str(message.content or "").strip()
                    if content:
                        final_answer = content
                        print(f"[Final Answer] {content}")

            elif isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", None) or "tool"
                called_tool_names.add(tool_name)
                print(f"[Observation:{tool_name}] {_safe_preview(message.content, 500)}")

    if not final_answer:
        final_state = await agent_executor.ainvoke({"messages": [HumanMessage(content=query)]})
        final_messages = final_state.get("messages", [])
        for message in reversed(final_messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                candidate = str(message.content or "").strip()
                if candidate:
                    final_answer = candidate
                    break

    if not final_answer:
        final_answer = "No final textual answer produced by the agent."

    wants_report = any(token in query.lower() for token in ["report", "markdown", "md"])
    analyzed_done = "analyze_pdf" in called_tool_names
    report_done = "generate_markdown_report" in called_tool_names

    # Deterministic fallback: if model stopped early after analysis, still create report.
    if wants_report and analyzed_done and not report_done:
        session_id = analyzed_session_id or "default"
        fallback_payload: Dict[str, Any] = {"session_id": session_id}
        if requested_report_filename:
            fallback_payload["output_filename"] = requested_report_filename

        print("[Fallback] Model ended before report generation. Calling generate_markdown_report directly...")
        fallback_result = await generate_markdown_report_tool.ainvoke(fallback_payload)
        report_path = fallback_result.get("report_path") if isinstance(fallback_result, dict) else None
        print(f"[Fallback Result] {_safe_preview(fallback_result, 500)}")
        if report_path:
            final_answer = f"Report generated at: {report_path}"

    return {
        "query": query,
        "final_answer": final_answer,
    }
