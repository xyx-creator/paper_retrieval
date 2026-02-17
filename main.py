"""Dual-Brain Paper Retrieval Agent（LangGraph 入口）。"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from config import KEYWORDS, RELEVANCE_THRESHOLD
from workflow.graph import APP


def _parse_csv_keywords(value: str | None) -> List[str]:
    """将逗号分隔关键词字符串解析为列表。"""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_user_query(args: argparse.Namespace) -> Dict[str, Any]:
    """根据命令行参数构造图输入的 `user_query`。"""

    # mandatory：优先级 mandatory > config 默认值
    if args.mandatory:
        mandatory = _parse_csv_keywords(args.mandatory)
    else:
        mandatory = list(KEYWORDS.get("mandatory", []))

    # bonus：若用户显式传入则使用传入；否则在“完全默认检索词”场景下应用默认 bonus。
    if args.bonus:
        bonus = _parse_csv_keywords(args.bonus)
    else:
        if not args.mandatory:
            bonus = list(KEYWORDS.get("bonus", []))
        else:
            bonus = []

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
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="GLM Dual-Brain Paper Retrieval Agent (LangGraph)"
    )
    parser.add_argument(
        "--source",
        choices=["local", "arxiv", "dblp", "all"],
        default="local",
        help="论文来源：local/arxiv/dblp/all",
    )
    parser.add_argument("--days", type=int, default=1, help="arXiv 检索最近 N 天")
    parser.add_argument("--venue", type=str, help="DBLP 会议信息，例如 CVPR")
    parser.add_argument("--year", type=int, help="DBLP 年份，例如 2025")

    # 新参数：分层关键词
    parser.add_argument("--mandatory", type=str, help="逗号分隔的 mandatory 关键词")
    parser.add_argument("--bonus", type=str, help="逗号分隔的 bonus 关键词")

    # 并发与筛选参数
    parser.add_argument("--max-workers", type=int, default=5, help="并行线程数")
    parser.add_argument("--top-k", type=int, default=10, help="保留高分论文数量")
    parser.add_argument(
        "--max-local-papers", type=int, default=15, help="local 模式最多处理论文数"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=RELEVANCE_THRESHOLD,
        help="相关性分数阈值（1-10）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 参数约束：dblp/all 需要 venue + year。
    if args.source in {"dblp", "all"} and (not args.venue or not args.year):
        parser.error("--source=dblp 或 --source=all 时，必须提供 --venue 和 --year。")

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

    # 入口执行：LangGraph 编译图一次构建，多次可复用。
    final_state = APP.invoke(initial_state)

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
    main()
