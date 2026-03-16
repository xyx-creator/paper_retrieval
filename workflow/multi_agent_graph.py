"""LangGraph orchestration for the Phase 3 multi-agent workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from workflow.multi_agent_nodes import (
    researcher_agent_node,
    vision_expert_agent_node,
    writer_agent_node,
)
from workflow.multi_agent_state import MultiAgentState


def _route_after_researcher(state: MultiAgentState) -> str:
    papers = state.get("retrieved_papers") or []
    return "to_vision" if papers else "to_end"


def build_multi_agent_graph():
    """Build and compile the multi-agent graph."""
    graph = StateGraph(MultiAgentState)

    graph.add_node("researcher", researcher_agent_node)
    graph.add_node("vision_expert", vision_expert_agent_node)
    graph.add_node("writer", writer_agent_node)

    graph.add_edge(START, "researcher")
    graph.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {
            "to_vision": "vision_expert",
            "to_end": END,
        },
    )
    graph.add_edge("vision_expert", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


MULTI_AGENT_APP = build_multi_agent_graph()
