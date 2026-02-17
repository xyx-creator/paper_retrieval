"""LangGraph 状态定义。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class UserQuery(TypedDict, total=False):
    """用户输入与运行参数。"""

    source: str
    days: int
    venue: Optional[str]
    year: Optional[int]
    mandatory_keywords: List[str]
    bonus_keywords: List[str]
    relevance_threshold: int
    max_workers: int
    top_k: int
    max_local_papers: int


class GraphState(TypedDict, total=False):
    """工作流全局状态（在各节点间流转）。"""

    # 用户查询配置（mandatory/bonus/source 等）。
    user_query: UserQuery

    # 检索阶段输出：初步候选论文。
    candidate_papers: List[Dict[str, Any]]

    # Node3 中间态：关键词过滤后的论文。
    filtered_papers: List[Dict[str, Any]]

    # Node3 输出：经过摘要打分后的论文。
    scored_papers: List[Dict[str, Any]]

    # Node4 输出：完成下载、选图、双脑分析后的论文。
    processed_papers: List[Dict[str, Any]]

    # Node1 输出：mandatory 扩展词典与 bonus 词列表。
    expanded_mandatory: Dict[str, List[str]]
    expanded_bonus: List[str]

    # Node5 输出：最终报告内容及落盘路径。
    final_report: str
    report_path: str

    # 统一错误收集，避免单点异常中断整图。
    errors: List[str]

