"""State definitions for the Phase 3 multi-agent workflow."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langchain_core.messages import BaseMessage


class MultiAgentState(TypedDict, total=False):
    """Shared state passed across researcher, vision expert, and writer agents."""

    task_instruction: str
    retrieved_papers: List[Dict[str, Any]]
    downloaded_pdfs: List[str]
    visual_assets: List[str]
    final_report: str
    messages: List[BaseMessage]
    errors: List[str]
