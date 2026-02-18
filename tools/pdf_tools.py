from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import fitz
from langchain_core.tools import tool
from pydantic import BaseModel

from modules.pdf_processor import (
    crop_specific_figure as _crop_specific_figure,
    extract_all_captions as _extract_all_captions,
    extract_text_and_metadata as _extract_text_and_metadata,
)
from tools.tool_schemas import (
    CropSpecificFigureInput,
    ExtractAllCaptionsInput,
    ExtractTextMetadataInput,
)


def _rect_to_list(rect_value: Any) -> List[float]:
    if isinstance(rect_value, fitz.Rect):
        return [
            float(rect_value.x0),
            float(rect_value.y0),
            float(rect_value.x1),
            float(rect_value.y1),
        ]
    if isinstance(rect_value, dict):
        return [
            float(rect_value.get("x0", 0.0)),
            float(rect_value.get("y0", 0.0)),
            float(rect_value.get("x1", 0.0)),
            float(rect_value.get("y1", 0.0)),
        ]
    if isinstance(rect_value, BaseModel):
        data = rect_value.model_dump()
        return [
            float(data.get("x0", 0.0)),
            float(data.get("y0", 0.0)),
            float(data.get("x1", 0.0)),
            float(data.get("y1", 0.0)),
        ]
    if isinstance(rect_value, (list, tuple)) and len(rect_value) == 4:
        return [float(x) for x in rect_value]
    return [0.0, 0.0, 0.0, 0.0]


def _rect_to_fitz(rect_value: Any) -> fitz.Rect:
    rect_list = _rect_to_list(rect_value)
    return fitz.Rect(*rect_list)


def _caption_field(item: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field_name, default)
    if isinstance(item, BaseModel):
        return getattr(item, field_name, default)
    return default


@tool("extract_text_and_metadata", args_schema=ExtractTextMetadataInput)
async def extract_text_and_metadata_tool(pdf_path: str) -> Dict[str, str]:
    return await asyncio.to_thread(_extract_text_and_metadata, pdf_path)


@tool("extract_all_captions", args_schema=ExtractAllCaptionsInput)
async def extract_all_captions_tool(pdf_path: str) -> List[Dict[str, Any]]:
    captions = await asyncio.to_thread(_extract_all_captions, pdf_path)
    serialized: List[Dict[str, Any]] = []

    for item in captions:
        serialized.append(
            {
                "page_num": int(item.get("page_num", 0)),
                "figure_id": str(item.get("figure_id", "")),
                "caption_text": str(item.get("caption_text", "")),
                "rect": _rect_to_list(item.get("rect")),
            }
        )

    return serialized


@tool("crop_specific_figure", args_schema=CropSpecificFigureInput)
async def crop_specific_figure_tool(
    pdf_path: str,
    target_figure_id: str,
    captions: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    restored_captions: List[Dict[str, Any]] = []
    for item in captions:
        restored_captions.append(
            {
                "page_num": int(_caption_field(item, "page_num", 0)),
                "figure_id": str(_caption_field(item, "figure_id", "")),
                "caption_text": str(_caption_field(item, "caption_text", "")),
                "rect": _rect_to_fitz(_caption_field(item, "rect")),
            }
        )

    if output_dir:
        return await asyncio.to_thread(
            _crop_specific_figure,
            pdf_path,
            target_figure_id,
            restored_captions,
            output_dir,
        )

    return await asyncio.to_thread(
        _crop_specific_figure,
        pdf_path,
        target_figure_id,
        restored_captions,
    )


PDF_TOOLS = [
    extract_text_and_metadata_tool,
    extract_all_captions_tool,
    crop_specific_figure_tool,
]

