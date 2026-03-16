"""Dynamic ReAct Paper Retrieval Agent entrypoint."""

from __future__ import annotations

import argparse
import asyncio

from agent_runner import process_user_query
from langchain_core.messages import HumanMessage
from workflow.multi_agent_graph import MULTI_AGENT_APP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper Retrieval ReAct Agent (LangGraph prebuilt)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["react", "multi-agent"],
        default="react",
        help="Execution mode: react (default) or multi-agent",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language query to drive autonomous tool planning",
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "react":
        result = await process_user_query(args.query)
        print("\nExecution Complete!")
        print(f"Final answer: {result['final_answer']}")
        return

    initial_state = {
        "task_instruction": args.query,
        "retrieved_papers": [],
        "downloaded_pdfs": [],
        "visual_assets": [],
        "final_report": "",
        "messages": [HumanMessage(content=args.query)],
        "errors": [],
    }
    result = await MULTI_AGENT_APP.ainvoke(initial_state)

    print("\nExecution Complete!")
    print("Final report:")
    print(result.get("final_report", ""))
    if result.get("errors"):
        print("\nErrors:")
        for err in result["errors"]:
            print(f"- {err}")


if __name__ == "__main__":
    asyncio.run(main())

