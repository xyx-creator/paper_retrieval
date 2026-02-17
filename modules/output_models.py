"""定义大模型结构化输出的数据模型。

设计目标：
1. 让 Prompt 输出与代码解析之间形成稳定契约；
2. 用 Pydantic 做类型约束，减少正则解析的脆弱性；
3. 保持字段语义清晰，便于后续 LangGraph 节点直接消费。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class KeywordExpansionEntry(BaseModel):
    """单个关键词的扩展结果。"""

    keyword: str = Field(..., description="输入的原始关键词。")
    variations: List[str] = Field(
        default_factory=list,
        description="该关键词对应的扩展词列表，建议包含原关键词本身。",
    )


class KeywordExpansionOutput(BaseModel):
    """关键词扩展任务的结构化输出。"""

    expansions: List[KeywordExpansionEntry] = Field(
        default_factory=list, description="所有关键词的扩展条目。"
    )

    def to_keyword_dict(self) -> Dict[str, List[str]]:
        """转换为主流程常用字典格式：{keyword: [variations...]}。"""
        keyword_dict: Dict[str, List[str]] = {}
        for item in self.expansions:
            cleaned = [x.strip() for x in item.variations if x and x.strip()]
            # 保证原词在第一个位置，提升后续 mandatory 逻辑稳定性。
            if item.keyword not in cleaned:
                cleaned.insert(0, item.keyword)
            keyword_dict[item.keyword] = cleaned
        return keyword_dict


class RelevanceScoreOutput(BaseModel):
    """摘要相关性评分结果。"""

    mandatory_match: bool = Field(..., description="是否满足 mandatory 关键词覆盖。")
    bonus_hits: List[str] = Field(
        default_factory=list, description="命中的 bonus 关键词列表。"
    )
    score: int = Field(..., ge=1, le=10, description="1-10 的整数评分。")
    reason: str = Field(..., description="一句话评分理由。")


class FigureSelectionOutput(BaseModel):
    """图表选择结果。"""

    figure_id: Optional[str] = Field(
        default=None, description='选中的图号，例如 "Figure 1"；若无匹配则为 null。'
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="选择置信度（0-1）。"
    )
    reason: str = Field(default="", description="选择理由。")


class TextBrainOutput(BaseModel):
    """文本脑抽取结构。"""

    year_venue: str = Field(default="Unknown", description="年份与会议/期刊信息。")
    paper_link: str = Field(default="Unknown", description="论文链接。")
    motivation: str = Field(default="", description="核心研究动机。")
    validation_tasks: List[str] = Field(
        default_factory=list, description="验证任务或数据集。"
    )
    core_conclusion: str = Field(default="", description="核心结论或性能收益。")
    core_modules: List[str] = Field(default_factory=list, description="核心模块列表。")
    data_flow: str = Field(default="", description="方法的数据流描述。")


class VisionBrainOutput(BaseModel):
    """视觉脑抽取结构。"""

    visible_modules: List[str] = Field(
        default_factory=list, description="图中可见模块名称。"
    )
    visible_connections: List[str] = Field(
        default_factory=list, description="图中可见连接关系。"
    )
    notes: str = Field(default="", description="可见性限制或不确定项说明。")


class SynthesisOutput(BaseModel):
    """融合报告输出结构。"""

    markdown: str = Field(..., description="最终 Markdown 报告正文（不含代码块包裹）。")

