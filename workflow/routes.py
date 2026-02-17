"""LangGraph 条件路由函数。"""

from __future__ import annotations

from workflow.state import GraphState


def route_after_information_retrieval(state: GraphState) -> str:
    """Node2 后路由：
    - 有候选论文 -> 进入过滤打分；
    - 无候选论文 -> 直接生成报告。
    """

    candidates = state.get("candidate_papers") or []
    return "to_filter_scoring" if candidates else "to_report"


def route_after_filter_scoring(state: GraphState) -> str:
    """Node3 后路由：
    - 有待深度分析论文 -> 进入双脑分析；
    - 无论文 -> 直接生成报告。
    """

    scored = state.get("scored_papers") or []
    return "to_deep_analysis" if scored else "to_report"

