"""统一管理 LangChain Prompt 模板。

注意：
1. 本文件中的 JSON 示例必须使用双花括号 `{{` `}}` 转义，
   否则 `ChatPromptTemplate` 会把它们误识别为变量占位符；
2. 真正需要传入的模板变量仍使用单花括号，例如 `{keywords_text}`。
"""

from langchain_core.prompts import ChatPromptTemplate


# 关键词扩展（严格模式）Prompt
KEYWORD_EXPANSION_STRICT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是学术检索关键词归一化助手。"
                "你只能做语法形态层面的变体扩展（单复数、连字符、缩写、大小写变体）。"
                "禁止引入新概念、同义词、上位词或下位词。"
            ),
        ),
        (
            "human",
            (
                "输入关键词列表：\n{keywords_text}\n\n"
                "请输出 JSON，对应结构如下：\n"
                "{{\n"
                '  "expansions": [\n'
                '    {{"keyword": "原关键词", "variations": ["原词", "变体1", "变体2"]}}\n'
                "  ]\n"
                "}}\n\n"
                "约束：\n"
                "1. 每个关键词最多 {max_variations} 个变体；\n"
                "2. 必须包含原关键词；\n"
                "3. 不要输出 JSON 以外的任何文本。"
            ),
        ),
    ]
)


# 关键词扩展（宽松模式）Prompt
KEYWORD_EXPANSION_BROAD_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是学术检索助手。"
                "你需要为每个关键词扩展高相关术语（同义词、常见缩写、学术表达）。"
                "扩展必须保持学术语义一致，避免无关泛化。"
            ),
        ),
        (
            "human",
            (
                "输入关键词列表：\n{keywords_text}\n\n"
                "请输出 JSON，对应结构如下：\n"
                "{{\n"
                '  "expansions": [\n'
                '    {{"keyword": "原关键词", "variations": ["原词", "扩展1", "扩展2"]}}\n'
                "  ]\n"
                "}}\n\n"
                "约束：\n"
                "1. 每个关键词最多 {max_variations} 个术语；\n"
                "2. 必须包含原关键词；\n"
                "3. 不要输出 JSON 以外的任何文本。"
            ),
        ),
    ]
)


# 相关性打分 Prompt
RELEVANCE_SCORING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是严格的学术审稿人。"
                "请根据 mandatory 与 bonus 关键词对摘要进行相关性评分。"
                "评分范围为 1 到 10，必须给出可审计的简短理由。"
            ),
        ),
        (
            "human",
            (
                "MANDATORY 关键词：{mandatory_keywords}\n"
                "BONUS 关键词：{bonus_keywords}\n\n"
                "论文摘要：\n{abstract}\n\n"
                "请输出 JSON，字段如下：\n"
                "{{\n"
                '  "mandatory_match": true/false,\n'
                '  "bonus_hits": ["命中的 bonus 词"],\n'
                '  "score": 1-10 的整数,\n'
                '  "reason": "一句话理由"\n'
                "}}\n\n"
                "约束：\n"
                "1. 若未覆盖 mandatory，score 必须小于 5；\n"
                "2. 只输出 JSON。"
            ),
        ),
    ]
)


# 图表选择 Prompt
FIGURE_SELECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是论文图表筛选助手。"
                "目标是从 caption 列表中选出最能展示整体方法架构/流程的图。"
            ),
        ),
        (
            "human",
            (
                "候选图注列表：\n{captions_text}\n\n"
                "请输出 JSON：\n"
                "{{\n"
                '  "figure_id": "Figure X" 或 null,\n'
                '  "confidence": 0 到 1 的小数,\n'
                '  "reason": "一句话说明"\n'
                "}}\n\n"
                "优先级：Overview > Architecture > Framework > Pipeline > Model。\n"
                "若均不符合，请返回 figure_id=null。\n"
                "只输出 JSON。"
            ),
        ),
    ]
)


# 文本脑 Prompt
TEXT_BRAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是论文文本信息抽取助手。"
                "你只能根据提供文本抽取信息，不可虚构。"
            ),
        ),
        (
            "human",
            (
                "Abstract:\n{abstract}\n\n"
                "Introduction:\n{introduction}\n\n"
                "请输出 JSON，字段如下：\n"
                "{{\n"
                '  "year_venue": "年份与会议/期刊，未知则 Unknown",\n'
                '  "paper_link": "论文链接，未知则 Unknown",\n'
                '  "motivation": "核心动机",\n'
                '  "validation_tasks": ["任务或数据集"],\n'
                '  "core_conclusion": "核心结论",\n'
                '  "core_modules": ["核心模块"],\n'
                '  "data_flow": "关键数据流描述"\n'
                "}}\n\n"
                "只输出 JSON。"
            ),
        ),
    ]
)


# 视觉脑文本指令 Prompt
VISION_BRAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是架构图解析助手。"
                "请严格基于图中可见内容描述模块与连接关系，不可臆测。"
            ),
        ),
        (
            "human",
            (
                "请输出 JSON：\n"
                "{{\n"
                '  "visible_modules": ["图中可见模块名"],\n'
                '  "visible_connections": ["可见箭头或连接关系"],\n'
                '  "notes": "不确定项或可见性限制"\n'
                "}}\n\n"
                "只输出 JSON。"
            ),
        ),
    ]
)


# 融合报告 Prompt
SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "你是 Dual-Brain 报告生成助手。"
                "你需要融合文本脑与视觉脑结果，输出结构化 Markdown。"
            ),
        ),
        (
            "human",
            (
                "Text Brain:\n{text_analysis}\n\n"
                "Vision Brain:\n{vision_analysis}\n\n"
                "Relevance Score: {relevance_score}\n"
                "Keywords: {keywords}\n\n"
                "请输出 JSON：\n"
                "{{\n"
                '  "markdown": "完整 markdown 文本（不要包裹 ```）"\n'
                "}}\n\n"
                "要求：\n"
                "1. 使用如下章节：Basic Information / Background / Core Architecture and Method / Experimental Performance；\n"
                "2. Method Description 需交叉验证文本与图像；\n"
                "3. 只输出 JSON。"
            ),
        ),
    ]
)


__all__ = [
    "KEYWORD_EXPANSION_STRICT_PROMPT",
    "KEYWORD_EXPANSION_BROAD_PROMPT",
    "RELEVANCE_SCORING_PROMPT",
    "FIGURE_SELECTION_PROMPT",
    "TEXT_BRAIN_PROMPT",
    "VISION_BRAIN_PROMPT",
    "SYNTHESIS_PROMPT",
]

