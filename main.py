"""Dual-Brain Paper Retrieval Agent entrypoint."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List

from config import KEYWORDS, RELEVANCE_THRESHOLD
from workflow.graph import APP


def _parse_csv_keywords(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_user_query(args: argparse.Namespace) -> Dict[str, Any]:
    if args.mandatory:
        mandatory = _parse_csv_keywords(args.mandatory)
    else:
        mandatory = list(KEYWORDS.get("mandatory", []))

    if args.bonus:
        bonus = _parse_csv_keywords(args.bonus)
    else:
        bonus = list(KEYWORDS.get("bonus", [])) if not args.mandatory else []

    return {
        "source": args.source,
        "days": args.days,
        "venue": args.venue,
        "year": args.year,
        "mandatory_keywords": mandatory,
        "bonus_keywords": bonus,
        "relevance_threshold": args.threshold,
        "max_workers": args.max_workers,
        "top_k": args.top_k,
        "max_local_papers": args.max_local_papers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GLM Dual-Brain Paper Retrieval Agent (LangGraph)"
    )
    parser.add_argument(
        "--source",
        choices=["local", "arxiv", "dblp", "all"],
        default="local",
        help="Paper source: local/arxiv/dblp/all",
    )
    parser.add_argument("--days", type=int, default=1, help="arXiv lookback days")
    parser.add_argument("--venue", type=str, help="DBLP venue, e.g. CVPR")
    parser.add_argument("--year", type=int, help="DBLP year, e.g. 2025")
    parser.add_argument(
        "--mandatory",
        type=str,
        help="Comma-separated mandatory keywords",
    )
    parser.add_argument(
        "--bonus",
        type=str,
        help="Comma-separated bonus keywords",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Concurrent worker limit",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k papers to keep")
    parser.add_argument(
        "--max-local-papers",
        type=int,
        default=15,
        help="Max local PDFs to process",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=RELEVANCE_THRESHOLD,
        help="Relevance threshold (1-10)",
    )
    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.source in {"dblp", "all"} and (not args.venue or not args.year):
        parser.error("--source=dblp or --source=all requires --venue and --year.")

    user_query = _build_user_query(args)
    print("Initializing LangGraph Dual-Brain Agent...")
    print(f"[Source] {user_query['source']}")
    print(f"[Mandatory] {user_query['mandatory_keywords']}")
    print(f"[Bonus] {user_query['bonus_keywords']}")

    initial_state = {
        "user_query": user_query,
        "candidate_papers": [],
        "filtered_papers": [],
        "scored_papers": [],
        "processed_papers": [],
        "errors": [],
    }

    final_state = await APP.ainvoke(initial_state)

    report_path = final_state.get("report_path")
    processed_count = len(final_state.get("processed_papers", []))
    errors = final_state.get("errors", [])

    print("\nProcessing Complete!")
    print(f"Processed papers: {processed_count}")
    if report_path:
        print(f"Report generated: {report_path}")
    if errors:
        print(f"Warnings/Errors: {len(errors)}")
        for err in errors[:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    asyncio.run(main())

