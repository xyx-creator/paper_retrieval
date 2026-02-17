"""GLM Agent（LangChain 重构版）。

重构目标：
1. 使用 LangChain 聊天模型接口替代原生 SDK 直调；
2. 使用结构化输出（Pydantic）替代正则解析；
3. 保留原有“双脑分析”方法接口，确保主流程调用方式稳定。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from config import (
    MODEL_KEYWORD_EXPANSION,
    MODEL_TEXT,
    MODEL_VISION,
    ZHIPUAI_API_KEY,
)
from modules.output_models import (
    FigureSelectionOutput,
    KeywordExpansionOutput,
    RelevanceScoreOutput,
    SynthesisOutput,
    TextBrainOutput,
    VisionBrainOutput,
)
from modules.prompts import (
    FIGURE_SELECTION_PROMPT,
    KEYWORD_EXPANSION_BROAD_PROMPT,
    KEYWORD_EXPANSION_STRICT_PROMPT,
    RELEVANCE_SCORING_PROMPT,
    SYNTHESIS_PROMPT,
    TEXT_BRAIN_PROMPT,
    VISION_BRAIN_PROMPT,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class GLMAgent:
    """基于 LangChain 的双脑 Agent。

    说明：
    - 文本任务（关键词扩展/打分/融合）统一走文本模型；
    - 视觉任务（架构图解析）走视觉模型；
    - 所有关键输出都走 Pydantic 结构化校验，避免 fragile 正则。
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ZHIPUAI_API_KEY
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY 未配置，无法初始化 GLMAgent。")

        # 分别构建关键词扩展、文本分析、视觉分析模型句柄。
        # 这样做可以对不同任务设置不同 temperature。
        self.keyword_llm = self._build_chat_model(
            model_name=MODEL_KEYWORD_EXPANSION,
            temperature=0.3,
        )
        self.text_llm = self._build_chat_model(
            model_name=MODEL_TEXT,
            temperature=0.1,
        )
        self.vision_llm = self._build_chat_model(
            model_name=MODEL_VISION,
            temperature=0.1,
        )

    def _build_chat_model(self, model_name: str, temperature: float):
        """构建聊天模型实例（仅使用 ChatZhipuAI）。"""

        try:
            from langchain_community.chat_models import ChatZhipuAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "未检测到 `langchain_community`，无法使用 ChatZhipuAI。"
                "请先安装 `langchain-community`。"
            ) from exc

        try:
            # 新版本通常使用 api_key 参数。
            return ChatZhipuAI(
                model=model_name,
                api_key=self.api_key,
                temperature=temperature,
            )
        except TypeError:
            # 兼容部分版本参数名为 zhipuai_api_key。
            return ChatZhipuAI(
                model=model_name,
                zhipuai_api_key=self.api_key,
                temperature=temperature,
            )

    @staticmethod
    def _normalize_keywords(keywords: Union[str, Sequence[str], None]) -> List[str]:
        """统一关键词输入格式为字符串列表。"""

        if keywords is None:
            return []
        if isinstance(keywords, str):
            return [k.strip() for k in keywords.split(",") if k.strip()]
        return [str(k).strip() for k in keywords if str(k).strip()]

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """清理模型输出中的 Markdown 包裹，便于 JSON 反序列化。"""

        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return cleaned

    def _invoke_structured(
        self,
        llm: Any,
        prompt_template: Any,
        payload: Dict[str, Any],
        output_model: Type[OutputModelT],
    ) -> OutputModelT:
        """统一结构化调用入口。

        实现策略：
        1. 先走 `with_structured_output`（最稳妥，能直接返回 Pydantic 对象）；
        2. 若模型端不支持或输出异常，再回退“普通调用 + JSON 解析”。
        """

        try:
            chain = prompt_template | llm.with_structured_output(output_model)
            result = chain.invoke(payload)
            if isinstance(result, output_model):
                return result
            if isinstance(result, dict):
                return output_model.model_validate(result)
        except Exception:
            pass

        # 回退：普通消息调用 + 手工 JSON 校验。
        messages = prompt_template.format_messages(**payload)
        raw = llm.invoke(messages)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        cleaned = self._clean_json_text(raw_text)
        return output_model.model_validate_json(cleaned)

    def _encode_image_to_data_url(self, image_path: str) -> str:
        """将本地图像编码为 data URL，供视觉模型输入。"""

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _heuristic_select_best_figure(captions_list: Sequence[str]) -> Optional[str]:
        """当结构化选图失败时的启发式兜底。

        策略：
        1. 优先选择包含 Overview/Architecture/Framework/Pipeline/Model 的图；
        2. 若有 Figure 1 或 Figure 2，优先早期图；
        3. 最后回退到首个可解析图号。
        """

        if not captions_list:
            return None

        # 预解析所有 caption 的 figure id，便于后续优先级筛选。
        parsed: List[tuple[str, str]] = []
        for caption in captions_list:
            match = re.search(r"(Figure|Fig\.?)\s*(\d+)", caption, re.IGNORECASE)
            if not match:
                continue
            fig = f"Figure {match.group(2)}"
            parsed.append((fig, caption.lower()))

        if not parsed:
            return None

        priority_keywords = ["overview", "architecture", "framework", "pipeline", "model"]

        # 先找“关键词命中 + 早期图”
        for target in ("Figure 1", "Figure 2"):
            for fig, lower_caption in parsed:
                if fig == target and any(k in lower_caption for k in priority_keywords):
                    return fig

        # 再找“关键词命中任意图”
        for fig, lower_caption in parsed:
            if any(k in lower_caption for k in priority_keywords):
                return fig

        # 最后回退首图
        return parsed[0][0]

    def expand_keywords_batch(
        self, keywords: Sequence[str], mode: str = "strict"
    ) -> Dict[str, List[str]]:
        """批量扩展关键词并返回 `{原词: [变体...]}`。

        参数：
        - keywords: 原始关键词列表。
        - mode:
          - `strict`: 只做词形变化；
          - `broad`: 可做语义相关扩展。
        """

        normalized = self._normalize_keywords(list(keywords))
        if not normalized:
            return {}

        prompt = (
            KEYWORD_EXPANSION_STRICT_PROMPT
            if mode == "strict"
            else KEYWORD_EXPANSION_BROAD_PROMPT
        )
        max_variations = 3 if mode == "strict" else 5
        keywords_text = "\n".join(f"- {k}" for k in normalized)

        try:
            output = self._invoke_structured(
                llm=self.keyword_llm,
                prompt_template=prompt,
                payload={
                    "keywords_text": keywords_text,
                    "max_variations": max_variations,
                },
                output_model=KeywordExpansionOutput,
            )
            result = output.to_keyword_dict()
        except Exception:
            # 失败兜底：返回原词，确保流程可继续。
            result = {k: [k] for k in normalized}

        # 补齐所有输入词，防止模型漏项导致后续 mandatory 过滤失真。
        for keyword in normalized:
            if keyword not in result:
                result[keyword] = [keyword]
        return result

    def score_relevance(
        self,
        abstract: str,
        mandatory_keywords: Union[str, Sequence[str], None],
        bonus_keywords: Union[str, Sequence[str], None],
    ) -> int:
        """对摘要进行 1-10 相关性打分（结构化输出）。"""

        mandatory_list = self._normalize_keywords(mandatory_keywords)
        bonus_list = self._normalize_keywords(bonus_keywords)

        try:
            output = self._invoke_structured(
                llm=self.text_llm,
                prompt_template=RELEVANCE_SCORING_PROMPT,
                payload={
                    "mandatory_keywords": ", ".join(mandatory_list),
                    "bonus_keywords": ", ".join(bonus_list),
                    "abstract": abstract or "",
                },
                output_model=RelevanceScoreOutput,
            )
            return max(1, min(10, int(output.score)))
        except Exception:
            return 1

    def select_best_figure(self, captions_list: Sequence[str]) -> Optional[str]:
        """从图注中选择最适合展示方法架构的图号。"""

        if not captions_list:
            return None

        captions_text = "\n".join(f"- {c}" for c in captions_list)
        try:
            output = self._invoke_structured(
                llm=self.text_llm,
                prompt_template=FIGURE_SELECTION_PROMPT,
                payload={"captions_text": captions_text},
                output_model=FigureSelectionOutput,
            )
        except Exception:
            return self._heuristic_select_best_figure(captions_list)

        if not output.figure_id:
            return self._heuristic_select_best_figure(captions_list)

        # 规范化 "Fig. 1" / "Figure 1" 为统一格式 "Figure 1"。
        match = re.search(r"(Figure|Fig\.?)\s*(\d+)", output.figure_id, re.IGNORECASE)
        if not match:
            return self._heuristic_select_best_figure(captions_list)
        return f"Figure {match.group(2)}"

    def analyze_text_brain(
        self, abstract: str, introduction: str = ""
    ) -> Dict[str, Any]:
        """文本脑：从摘要与引言抽取结构化关键信息。"""

        payload = {
            "abstract": abstract or "",
            "introduction": (introduction or "")[:3000],
        }
        try:
            output = self._invoke_structured(
                llm=self.text_llm,
                prompt_template=TEXT_BRAIN_PROMPT,
                payload=payload,
                output_model=TextBrainOutput,
            )
            return output.model_dump()
        except Exception as exc:
            return {
                "year_venue": "Unknown",
                "paper_link": "Unknown",
                "motivation": "",
                "validation_tasks": [],
                "core_conclusion": f"Text analysis failed: {exc}",
                "core_modules": [],
                "data_flow": "",
            }

    def analyze_vision_brain(self, image_path: str) -> Dict[str, Any]:
        """视觉脑：读取图像并输出可见模块与连接关系。"""

        image_data_url = self._encode_image_to_data_url(image_path)

        # 这里复用 prompts.py 中的文本模板，保证提示词集中管理。
        vision_messages = VISION_BRAIN_PROMPT.format_messages()
        system_message = vision_messages[0]
        instruction_text = vision_messages[-1].content

        # LangChain 的多模态输入：HumanMessage.content 使用 list[dict]。
        messages = [
            system_message,
            HumanMessage(
                content=[
                    {"type": "text", "text": str(instruction_text)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]
            ),
        ]

        # 先尝试结构化输出，若失败则兜底普通文本。
        try:
            structured = self.vision_llm.with_structured_output(VisionBrainOutput)
            output = structured.invoke(messages)
            if isinstance(output, VisionBrainOutput):
                return output.model_dump()
            if isinstance(output, dict):
                return VisionBrainOutput.model_validate(output).model_dump()
        except Exception:
            pass

        try:
            raw = self.vision_llm.invoke(messages)
            content = raw.content if hasattr(raw, "content") else str(raw)
            return {
                "visible_modules": [],
                "visible_connections": [],
                "notes": str(content),
            }
        except Exception as exc:
            return {
                "visible_modules": [],
                "visible_connections": [],
                "notes": f"Vision analysis failed: {exc}",
            }

    def synthesize_report(
        self,
        text_analysis: Union[str, Dict[str, Any]],
        vision_analysis: Union[str, Dict[str, Any]],
        relevance_score: int,
        keywords: Union[str, Sequence[str], None],
    ) -> str:
        """融合文本脑与视觉脑结果，生成最终 Markdown 段落。"""

        if isinstance(text_analysis, str):
            text_payload = text_analysis
        else:
            text_payload = json.dumps(text_analysis, ensure_ascii=False, indent=2)

        if isinstance(vision_analysis, str):
            vision_payload = vision_analysis
        else:
            vision_payload = json.dumps(vision_analysis, ensure_ascii=False, indent=2)

        keyword_text = ", ".join(self._normalize_keywords(keywords))

        try:
            output = self._invoke_structured(
                llm=self.text_llm,
                prompt_template=SYNTHESIS_PROMPT,
                payload={
                    "text_analysis": text_payload,
                    "vision_analysis": vision_payload,
                    "relevance_score": relevance_score,
                    "keywords": keyword_text,
                },
                output_model=SynthesisOutput,
            )
            return output.markdown.strip()
        except Exception as exc:
            # 合成失败时返回最小可读兜底，避免整篇报告中断。
            return (
                "## 1. Basic Information\n"
                f"* **Relevance Score**: {relevance_score} / 10\n\n"
                "## 2. Background\n"
                "* **Core Motivation**: 解析失败。\n\n"
                "## 3. Core Architecture and Method\n"
                f"* **Method Description**: Synthesis failed: {exc}\n\n"
                "## 4. Experimental Performance\n"
                "* **Validation Tasks**: Unknown\n"
                "* **Core Conclusion**: Unknown\n"
            )
