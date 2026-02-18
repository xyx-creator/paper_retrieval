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
async def search_arxiv_tool(
    query_keywords: List[str],
    days: int = 1,
) -> List[Dict[str, Any]]:
    return await _search_arxiv(query_keywords=query_keywords, days=days)


@tool("search_dblp", args_schema=SearchDblpInput)
async def search_dblp_tool(venue: str, year: int) -> List[Dict[str, Any]]:
    return await _search_dblp(venue=venue, year=year)


@tool("batch_fetch_s2", args_schema=BatchFetchS2Input)
async def batch_fetch_s2_tool(papers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return await _batch_fetch_s2(papers_data=papers_data)


@tool("download_pdf", args_schema=DownloadPdfInput)
async def download_pdf_tool(url: str, save_dir: Optional[str] = None) -> Optional[str]:
    if save_dir:
        return await _download_pdf(url=url, save_dir=save_dir)
    return await _download_pdf(url=url)


@tool("filter_by_keywords", args_schema=FilterByKeywordsInput)
async def filter_by_keywords_tool(
    papers: List[Dict[str, Any]],
    expanded_mandatory: Dict[str, List[str]],
    expanded_bonus: List[str],
) -> List[Dict[str, Any]]:
    return await _filter_by_keywords(
        papers=papers,
        expanded_mandatory=expanded_mandatory,
        expanded_bonus=expanded_bonus,
    )


PAPER_SOURCE_TOOLS = [
    search_arxiv_tool,
    search_dblp_tool,
    batch_fetch_s2_tool,
    download_pdf_tool,
    filter_by_keywords_tool,
]

