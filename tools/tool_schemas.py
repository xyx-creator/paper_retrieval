"""LangChain Tools 的输入参数模型定义。

说明：
1. 本文件只描述“工具输入契约”，不实现业务逻辑；
2. 所有字段都带中文语义，便于 Tool Calling 时提升参数命中率；
3. 对关键字段添加范围约束，减少无效调用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchArxivInput(BaseModel):
    """arXiv 检索参数。"""

    query_keywords: List[str] = Field(
        ..., min_length=1, description="检索关键词列表，例如 ['Vision-Language Models', 'Hallucination']。"
    )
    days: int = Field(default=1, ge=1, le=30, description="仅检索最近 N 天内的论文。")


class SearchDblpInput(BaseModel):
    """DBLP 检索参数。"""

    venue: str = Field(..., min_length=1, description="会议/期刊简称或名称，例如 CVPR。")
    year: int = Field(..., ge=1900, le=2100, description="目标年份，例如 2025。")


class BatchFetchS2Input(BaseModel):
    """Semantic Scholar 批量补全参数。"""

    papers_data: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "来自 DBLP 或其他来源的初始论文列表。"
            "每个元素建议包含 title、doi、year、venue 等字段。"
        ),
    )


class DownloadPdfInput(BaseModel):
    """PDF 下载参数。"""

    url: str = Field(..., min_length=1, description="论文 PDF 链接。")
    save_dir: Optional[str] = Field(
        default=None, description="保存目录；为空时使用系统默认 INPUT_DIR。"
    )


class FilterByKeywordsInput(BaseModel):
    """元数据关键词过滤参数。"""

    papers: List[Dict[str, Any]] = Field(..., description="候选论文列表。")
    expanded_mandatory: Dict[str, List[str]] = Field(
        ...,
        description="mandatory 扩展词典，格式为 {原词: [变体列表]}。",
    )
    expanded_bonus: List[str] = Field(
        default_factory=list,
        description="bonus 关键词列表。此字段主要用于兼容原函数签名。",
    )


class ExtractTextMetadataInput(BaseModel):
    """PDF 文本元数据提取参数。"""

    pdf_path: str = Field(..., min_length=1, description="本地 PDF 文件绝对路径或相对路径。")


class ExtractAllCaptionsInput(BaseModel):
    """PDF 图注提取参数。"""

    pdf_path: str = Field(..., min_length=1, description="本地 PDF 文件路径。")


class CaptionItemSchema(BaseModel):
    """可序列化图注结构。

    注意：
    - 原生函数中 `rect` 为 `fitz.Rect`，不便直接用于 Tool 序列化；
    - 这里统一为 `[x0, y0, x1, y1]` 四元数组，便于跨节点传递。
    """

    page_num: int = Field(..., ge=0, description="页码（从 0 开始）。")
    figure_id: str = Field(..., min_length=1, description='图号，例如 "Figure 1"。')
    caption_text: str = Field(default="", description="图注全文。")
    rect: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="图注文本块坐标 [x0, y0, x1, y1]。",
    )


class CropSpecificFigureInput(BaseModel):
    """按图号裁剪图像参数。"""

    pdf_path: str = Field(..., min_length=1, description="本地 PDF 文件路径。")
    target_figure_id: str = Field(
        ..., min_length=1, description='目标图号，例如 "Figure 1"。'
    )
    captions: List[CaptionItemSchema] = Field(
        ...,
        min_length=1,
        description="由 `extract_all_captions` 工具返回的图注列表。",
    )
    output_dir: Optional[str] = Field(
        default=None, description="裁剪图保存目录；为空时使用系统默认 OUTPUT_DIR。"
    )

