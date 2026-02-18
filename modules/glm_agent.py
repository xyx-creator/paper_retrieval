from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, Union

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from config import MODEL_KEYWORD_EXPANSION, MODEL_TEXT, MODEL_VISION, ZHIPUAI_API_KEY
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
    """Dual-brain agent wrapper built on LangChain Chat models."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ZHIPUAI_API_KEY
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY is not configured.")

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
        """Build ChatZhipuAI model instance with version-compatible args."""
        try:
            from langchain_community.chat_models import ChatZhipuAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "langchain-community is required to use ChatZhipuAI."
            ) from exc

        try:
            return ChatZhipuAI(
                model=model_name,
                api_key=self.api_key,
                temperature=temperature,
            )
        except TypeError:
            return ChatZhipuAI(
                model=model_name,
                zhipuai_api_key=self.api_key,
                temperature=temperature,
            )

    @staticmethod
    def _normalize_keywords(keywords: Union[str, Sequence[str], None]) -> List[str]:
        if keywords is None:
            return []
        if isinstance(keywords, str):
            return [k.strip() for k in keywords.split(",") if k.strip()]
        return [str(k).strip() for k in keywords if str(k).strip()]

    @staticmethod
    def _clean_json_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return cleaned

    async def _invoke_structured(
        self,
        llm: Any,
        prompt_template: Any,
        payload: Dict[str, Any],
        output_model: Type[OutputModelT],
    ) -> OutputModelT:
        """
        Unified structured invoke with two-stage fallback:
        1) with_structured_output + ainvoke
        2) plain ainvoke + JSON parse
        """
        try:
            chain = prompt_template | llm.with_structured_output(output_model)
            result = await chain.ainvoke(payload)
            if isinstance(result, output_model):
                return result
            if isinstance(result, dict):
                return output_model.model_validate(result)
        except Exception:
            pass

        messages = prompt_template.format_messages(**payload)
        raw = await llm.ainvoke(messages)
        raw_text = raw.content if hasattr(raw, "content") else str(raw)
        cleaned = self._clean_json_text(str(raw_text))
        return output_model.model_validate_json(cleaned)

    def _encode_image_to_data_url(self, image_path: str) -> str:
        with open(image_path, "rb") as file:
            image_bytes = file.read()

        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/png"
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _heuristic_select_best_figure(captions_list: Sequence[str]) -> Optional[str]:
        if not captions_list:
            return None

        parsed: List[tuple[str, str]] = []
        for caption in captions_list:
            match = re.search(r"(Figure|Fig\.?)\s*(\d+)", caption, re.IGNORECASE)
            if not match:
                continue
            parsed.append((f"Figure {match.group(2)}", caption.lower()))

        if not parsed:
            return None

        priority_keywords = ["overview", "architecture", "framework", "pipeline", "model"]
        for target in ("Figure 1", "Figure 2"):
            for fig, lower_caption in parsed:
                if fig == target and any(k in lower_caption for k in priority_keywords):
                    return fig

        for fig, lower_caption in parsed:
            if any(k in lower_caption for k in priority_keywords):
                return fig

        return parsed[0][0]

    async def expand_keywords_batch(
        self,
        keywords: Sequence[str],
        mode: str = "strict",
    ) -> Dict[str, List[str]]:
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
            output = await self._invoke_structured(
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
            result = {k: [k] for k in normalized}

        for keyword in normalized:
            if keyword not in result:
                result[keyword] = [keyword]
        return result

    async def score_relevance(
        self,
        abstract: str,
        mandatory_keywords: Union[str, Sequence[str], None],
        bonus_keywords: Union[str, Sequence[str], None],
    ) -> int:
        mandatory_list = self._normalize_keywords(mandatory_keywords)
        bonus_list = self._normalize_keywords(bonus_keywords)

        try:
            output = await self._invoke_structured(
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

    async def select_best_figure(self, captions_list: Sequence[str]) -> Optional[str]:
        if not captions_list:
            return None

        captions_text = "\n".join(f"- {caption}" for caption in captions_list)
        try:
            output = await self._invoke_structured(
                llm=self.text_llm,
                prompt_template=FIGURE_SELECTION_PROMPT,
                payload={"captions_text": captions_text},
                output_model=FigureSelectionOutput,
            )
        except Exception:
            return self._heuristic_select_best_figure(captions_list)

        if not output.figure_id:
            return self._heuristic_select_best_figure(captions_list)

        match = re.search(r"(Figure|Fig\.?)\s*(\d+)", output.figure_id, re.IGNORECASE)
        if not match:
            return self._heuristic_select_best_figure(captions_list)
        return f"Figure {match.group(2)}"

    async def analyze_text_brain(
        self,
        abstract: str,
        introduction: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "abstract": abstract or "",
            "introduction": (introduction or "")[:3000],
        }
        try:
            output = await self._invoke_structured(
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

    async def analyze_vision_brain(self, image_path: str) -> Dict[str, Any]:
        image_data_url = await asyncio.to_thread(self._encode_image_to_data_url, image_path)

        vision_messages = VISION_BRAIN_PROMPT.format_messages()
        system_message = vision_messages[0]
        instruction_text = vision_messages[-1].content
        messages = [
            system_message,
            HumanMessage(
                content=[
                    {"type": "text", "text": str(instruction_text)},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ]
            ),
        ]

        try:
            structured = self.vision_llm.with_structured_output(VisionBrainOutput)
            output = await structured.ainvoke(messages)
            if isinstance(output, VisionBrainOutput):
                return output.model_dump()
            if isinstance(output, dict):
                return VisionBrainOutput.model_validate(output).model_dump()
        except Exception:
            pass

        try:
            raw = await self.vision_llm.ainvoke(messages)
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

    async def synthesize_report(
        self,
        text_analysis: Union[str, Dict[str, Any]],
        vision_analysis: Union[str, Dict[str, Any]],
        relevance_score: int,
        keywords: Union[str, Sequence[str], None],
    ) -> str:
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
            output = await self._invoke_structured(
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
            return (
                "## 1. Basic Information\n"
                f"* **Relevance Score**: {relevance_score} / 10\n\n"
                "## 2. Background\n"
                "* **Core Motivation**: parse failed.\n\n"
                "## 3. Core Architecture and Method\n"
                f"* **Method Description**: Synthesis failed: {exc}\n\n"
                "## 4. Experimental Performance\n"
                "* **Validation Tasks**: Unknown\n"
                "* **Core Conclusion**: Unknown\n"
            )

