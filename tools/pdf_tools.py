"""PDF 处理相关工具封装。

关键约束（严格遵守）：
1. 不替换 `modules/pdf_processor.py` 内任何核心实现；
2. 尤其保留 `crop_specific_figure` 的坐标/图元/文本间距逻辑；
3. 本文件仅负责 Tool Calling 友好的输入输出适配。
"""

from __future__ import annotations

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
    """将不同形式的坐标统一为 `[x0, y0, x1, y1]`。

    该函数用于工具输出序列化，确保结果可以在 LangGraph 状态中稳定传递。
    """

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
    # 坐标异常时返回零框，避免工具层崩溃。
    return [0.0, 0.0, 0.0, 0.0]


def _rect_to_fitz(rect_value: Any) -> fitz.Rect:
    """将工具层坐标恢复成 `fitz.Rect`，以兼容原生裁剪函数。"""

    rect_list = _rect_to_list(rect_value)
    return fitz.Rect(*rect_list)


def _caption_field(item: Any, field_name: str, default: Any = None) -> Any:
    """兼容 dict / PydanticModel 两类 caption 输入。"""

    if isinstance(item, dict):
        return item.get(field_name, default)
    if isinstance(item, BaseModel):
        return getattr(item, field_name, default)
    return default


@tool("extract_text_and_metadata", args_schema=ExtractTextMetadataInput)
def extract_text_and_metadata_tool(pdf_path: str) -> Dict[str, str]:
    """从 PDF 首页面提取标题与摘要信息。

    注意：
    - 该工具直接调用原生函数；
    - 不引入 LangChain 文档加载器，保证原有行为一致。
    """

    return _extract_text_and_metadata(pdf_path=pdf_path)


@tool("extract_all_captions", args_schema=ExtractAllCaptionsInput)
def extract_all_captions_tool(pdf_path: str) -> List[Dict[str, Any]]:
    """提取 PDF 图注并返回可序列化结构。

    返回字段：
    - page_num: 页码（从 0 开始）
    - figure_id: 图号
    - caption_text: 图注文本
    - rect: [x0, y0, x1, y1]
    """

    captions = _extract_all_captions(pdf_path=pdf_path)
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
def crop_specific_figure_tool(
    pdf_path: str,
    target_figure_id: str,
    captions: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> Optional[str]:
    """根据指定 Figure ID 裁剪论文图像。

    重要说明：
    - 本工具不会改写原始裁剪算法；
    - 仅把工具层 captions 的可序列化坐标恢复为 `fitz.Rect`；
    - 其余逻辑完全交由 `modules/pdf_processor.py::crop_specific_figure` 执行。
    """

    restored_captions: List[Dict[str, Any]] = []
    for item in captions:
        # 工具层状态可能来自模型或上游节点，这里做稳健化兜底。
        restored_captions.append(
            {
                "page_num": int(_caption_field(item, "page_num", 0)),
                "figure_id": str(_caption_field(item, "figure_id", "")),
                "caption_text": str(_caption_field(item, "caption_text", "")),
                "rect": _rect_to_fitz(_caption_field(item, "rect")),
            }
        )

    if output_dir:
        return _crop_specific_figure(
            pdf_path=pdf_path,
            target_figure_id=target_figure_id,
            captions=restored_captions,
            output_dir=output_dir,
        )

    return _crop_specific_figure(
        pdf_path=pdf_path,
        target_figure_id=target_figure_id,
        captions=restored_captions,
    )


# 便于在 workflow 中一键注册工具集合。
PDF_TOOLS = [
    extract_text_and_metadata_tool,
    extract_all_captions_tool,
    crop_specific_figure_tool,
]
