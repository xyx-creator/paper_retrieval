"""LangGraph 图编排定义。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from workflow.nodes import (
    deep_analysis_node,
    filter_scoring_node,
    information_retrieval_node,
    keyword_expansion_node,
    report_generation_node,
)
from workflow.routes import route_after_filter_scoring, route_after_information_retrieval
from workflow.state import GraphState


def build_graph():
    """构建并编译 Dual-Brain Paper Retrieval 工作流。"""

    graph = StateGraph(GraphState)

    # 注册节点
    graph.add_node("keyword_expansion", keyword_expansion_node)
    graph.add_node("information_retrieval", information_retrieval_node)
    graph.add_node("filter_scoring", filter_scoring_node)
    graph.add_node("deep_analysis", deep_analysis_node)
    graph.add_node("report_generation", report_generation_node)

    # 主链路
    graph.add_edge(START, "keyword_expansion")
    graph.add_edge("keyword_expansion", "information_retrieval")

    # 条件跳转：检索为空时直接报告。
    graph.add_conditional_edges(
        "information_retrieval",
        route_after_information_retrieval,
        {
            "to_filter_scoring": "filter_scoring",
            "to_report": "report_generation",
        },
    )

    # 条件跳转：过滤/打分后无论文时直接报告。
    graph.add_conditional_edges(
        "filter_scoring",
        route_after_filter_scoring,
        {
            "to_deep_analysis": "deep_analysis",
            "to_report": "report_generation",
        },
    )

    graph.add_edge("deep_analysis", "report_generation")
    graph.add_edge("report_generation", END)

    return graph.compile()


# 便于主程序直接导入调用。
APP = build_graph()

