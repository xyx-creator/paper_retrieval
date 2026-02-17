"""论文来源相关工具封装。

核心原则：
1. 仅做 LangChain Tool 适配层，不修改底层业务实现；
2. 所有真正检索/下载逻辑仍由 `modules/paper_source.py` 承担；
3. 通过 Pydantic schema 提升 Tool Calling 的参数稳定性。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from modules.paper_source import (
    batch_fetch_s2 as _batch_fetch_s2,
    download_pdf as _download_pdf,
    filter_by_keywords as _filter_by_keywords,
    search_arxiv as _search_arxiv,
    search_dblp as _search_dblp,
)
from tools.tool_schemas import (
    BatchFetchS2Input,
    DownloadPdfInput,
    FilterByKeywordsInput,
    SearchArxivInput,
    SearchDblpInput,
)


@tool("search_arxiv", args_schema=SearchArxivInput)
def search_arxiv_tool(query_keywords: List[str], days: int = 1) -> List[Dict[str, Any]]:
    """在 arXiv 检索最近 N 天候选论文。

    参数：
    - query_keywords: 检索关键词列表。
    - days: 最近天数窗口。

    返回：
    - 候选论文列表，每个元素通常包含 `title`、`abstract`、`url`、`date`、`venue`。
    """

    return _search_arxiv(query_keywords=query_keywords, days=days)


@tool("search_dblp", args_schema=SearchDblpInput)
def search_dblp_tool(venue: str, year: int) -> List[Dict[str, Any]]:
    """在 DBLP 按会议/年份检索论文元数据。

    参数：
    - venue: 会议或期刊名称/简称。
    - year: 年份。

    返回：
    - DBLP 结果列表（通常含 `title`、`doi`、`year`、`venue`）。
    """

    return _search_dblp(venue=venue, year=year)


@tool("batch_fetch_s2", args_schema=BatchFetchS2Input)
def batch_fetch_s2_tool(papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """调用 Semantic Scholar 批量接口补全论文详情。

    参数：
    - papers_data: 初始论文列表（优先含 DOI）。

    返回：
    - 补全后的论文列表（通常含 `title`、`abstract`、`url`、`date`、`venue`）。
    """

    return _batch_fetch_s2(papers_data=papers_data)


@tool("download_pdf", args_schema=DownloadPdfInput)
def download_pdf_tool(url: str, save_dir: Optional[str] = None) -> Optional[str]:
    """下载论文 PDF 到本地目录。

    参数：
    - url: PDF 链接。
    - save_dir: 目标目录；为空时使用默认输入目录。

    返回：
    - 下载后的本地路径；下载失败则返回 `None`。
    """

    if save_dir:
        return _download_pdf(url=url, save_dir=save_dir)
    return _download_pdf(url=url)


@tool("filter_by_keywords", args_schema=FilterByKeywordsInput)
def filter_by_keywords_tool(
    papers: List[Dict[str, Any]],
    expanded_mandatory: Dict[str, List[str]],
    expanded_bonus: List[str],
) -> List[Dict[str, Any]]:
    """执行元数据层关键词硬过滤（Tiered 逻辑）。

    参数：
    - papers: 候选论文列表。
    - expanded_mandatory: mandatory 扩展词组（组间 AND，组内 OR）。
    - expanded_bonus: bonus 关键词（兼容签名，当前主要用于打分阶段）。

    返回：
    - 通过 mandatory 约束的论文列表。
    """

    return _filter_by_keywords(
        papers=papers,
        expanded_mandatory=expanded_mandatory,
        expanded_bonus=expanded_bonus,
    )


# 便于在 workflow 中一键注册工具集合。
PAPER_SOURCE_TOOLS = [
    search_arxiv_tool,
    search_dblp_tool,
    batch_fetch_s2_tool,
    download_pdf_tool,
    filter_by_keywords_tool,
]

