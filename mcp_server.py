import json
import os

from mcp.server.fastmcp import FastMCP

from config import INPUT_DIR
from modules.paper_source import download_pdf, search_arxiv, search_dblp
from modules.pdf_processor import crop_specific_figure, extract_all_captions

mcp = FastMCP("PaperBrain")


def _serialize_captions(captions: list[dict]) -> list[dict]:
    serialized: list[dict] = []
    for caption in captions:
        item = dict(caption)
        rect = item.get("rect")
        if rect is not None:
            item["rect"] = [rect.x0, rect.y0, rect.x1, rect.y1]
        serialized.append(item)
    return serialized


@mcp.tool()
async def search_arxiv_tool(query_keywords: list[str], days: int = 3) -> str:
    papers = await search_arxiv(query_keywords, days)
    return json.dumps(papers, ensure_ascii=False)


@mcp.tool()
async def search_dblp_tool(venue: str, year: int) -> str:
    papers = await search_dblp(venue, year)
    return json.dumps(papers, ensure_ascii=False)


@mcp.tool()
async def download_and_extract_captions_tool(url: str) -> str:
    pdf_path = await download_pdf(url, INPUT_DIR)
    if not pdf_path:
        return json.dumps({"error": "Failed to download PDF", "pdf_path": None, "captions": []})

    captions = extract_all_captions(pdf_path)
    return json.dumps(
        {"pdf_path": os.path.abspath(pdf_path), "captions": _serialize_captions(captions)},
        ensure_ascii=False,
    )


@mcp.tool()
async def crop_figure_tool(pdf_path: str, target_figure_id: str) -> str:
    captions = extract_all_captions(pdf_path)
    result = crop_specific_figure(pdf_path, target_figure_id, captions)
    if not result:
        return json.dumps(
            {
                "error": "Failed to crop figure",
                "cropped_image_path": None,
                "target_figure_id": target_figure_id,
            },
            ensure_ascii=False,
        )
    return json.dumps({"cropped_image_path": os.path.abspath(result)}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
