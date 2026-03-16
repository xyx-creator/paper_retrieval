"""Dynamic ReAct Paper Retrieval Agent entrypoint."""

from __future__ import annotations

import argparse
import asyncio

from agent_runner import process_user_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper Retrieval ReAct Agent (LangGraph prebuilt)"
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
    result = await process_user_query(args.query)
    print("\nExecution Complete!")
    print(f"Final answer: {result['final_answer']}")


if __name__ == "__main__":
    asyncio.run(main())

