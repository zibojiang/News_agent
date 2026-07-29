"""
agent.py — Industry News Agent 核心逻辑

支持 OpenAI 与 Google Gemini 两种模型提供方，
通过 Structured Outputs 强制返回规整 JSON，
从新闻正文中提炼带量化数据的商业案例。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from database import append_cases_batch_with_summary, load_topics, record_task_run
from quality_scorer import (
    NEWS_QUALITY_RULE_VERSION,
    QualitySummary,
    apply_ai_source_score,
    enrich_with_ai_result,
    score_article_pre_ai,
)
from scraper import fetch_and_extract_batch

# 加载 .env 环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# AI 提供方与模型名称。函数在每次调用时读取环境，支持 Streamlit
# 重新运行后立即使用本地 .env 的最新配置。
def get_ai_provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").strip().lower()


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()


def get_ai_analysis_workers() -> int:
    """返回单次任务的 AI 并发数，限制在安全范围内。"""
    try:
        configured = int(os.getenv("AI_ANALYSIS_WORKERS", "3"))
    except ValueError:
        configured = 3
    return max(1, min(6, configured))


# 保留兼容旧调用的导入时快照；核心分析和页面展示使用上面的动态函数。
DEFAULT_AI_PROVIDER = get_ai_provider()
DEFAULT_OPENAI_MODEL = get_openai_model()
DEFAULT_GEMINI_MODEL = get_gemini_model()
DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()

# API 调用重试配置
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # 秒，指数退避基数

# 定时任务默认相关性分数门槛
DEFAULT_MIN_SCORE = 70
SEARCH_RELEVANCE_DISPLAY_THRESHOLD = 40
BODY_ANALYSIS_RELEVANCE_THRESHOLD = 60
SEARCH_RELEVANCE_RULE_VERSION = "r2"
RECOMMENDATION_RELEVANCE_WEIGHT = 0.7
RECOMMENDATION_QUALITY_WEIGHT = 0.3

SEARCH_RELEVANCE_DIMENSION_MAXIMA = {
    "core_topic_match": 40,
    "information_need_match": 30,
    "semantic_coverage": 20,
    "directness": 10,
}


class SearchInterpretationSchema(BaseModel):
    """宽泛或歧义查询的一个可确认解释。"""

    label: str = Field(description="供用户选择的简短标题")
    description: str = Field(description="说明该解释的范围和关注点")
    intent_summary: str = Field(description="选中后可直接使用的最终搜索目标")
    target_topics: list[str] = Field(description="该解释的核心主题")
    chinese_queries: list[str] = Field(description="该解释的中文新闻检索词")
    english_queries: list[str] = Field(description="该解释的英文新闻检索词")
    relevance_criteria: list[str] = Field(description="该解释的相关性判断标准")


class SearchIntentSchema(BaseModel):
    """AI 对用户搜索意图和检索词的结构化理解。"""

    intent_summary: str = Field(description="一句话概括用户真正想了解的内容")
    target_topics: list[str] = Field(description="2-5 个必须关注的主题或方向")
    chinese_queries: list[str] = Field(description="1-3 个中文新闻检索词")
    english_queries: list[str] = Field(description="1-2 个英文新闻检索词")
    relevance_criteria: list[str] = Field(description="2-5 条判断新闻是否相关的标准")
    scope_level: Literal["broad", "focused", "specific"] = Field(
        default="specific",
        description="搜索范围：broad 宽泛、focused 适中、specific 具体",
    )
    needs_clarification: bool = Field(
        default=False,
        description="是否因范围过大或存在歧义而需要用户确认",
    )
    clarification_question: str = Field(
        default="",
        description="向用户提出的单个简洁澄清问题",
    )
    interpretations: list[SearchInterpretationSchema] = Field(
        default_factory=list,
        description="宽泛或歧义查询的 2-4 个可选解释",
    )
    recommended_interpretation_index: int = Field(
        default=0,
        ge=0,
        le=3,
        description="推荐解释在 interpretations 中的下标",
    )


class SourceCredibilitySchema(BaseModel):
    """AI 对单篇新闻的搜索相关性与来源权威度评分。"""

    score: int = Field(
        ge=0,
        le=25,
        description="新闻原始信息来源的权威度评分，0-25 分",
    )
    reason: str = Field(description="评分依据，说明发布者、域名和一手资料属性")
    core_topic_match_score: int = Field(
        ge=0,
        le=40,
        description="新闻核心主体与搜索主题的匹配度，0-40 分",
    )
    information_need_match_score: int = Field(
        ge=0,
        le=30,
        description="新闻是否回答用户具体想了解的问题，0-30 分",
    )
    semantic_coverage_score: int = Field(
        ge=0,
        le=20,
        description="正文是否实质讨论相关内容而非仅命中词语，0-20 分",
    )
    directness_score: int = Field(
        ge=0,
        le=10,
        description="搜索主题是新闻核心还是背景中顺带提及，0-10 分",
    )
    relevance_reason: str = Field(description="简短说明新闻命中或偏离了哪些搜索意图")

    @property
    def relevance_score(self) -> int:
        """由代码汇总四个维度，避免 AI 总分与细则不一致。"""
        return sum(self.relevance_dimension_scores().values())

    def relevance_dimension_scores(self) -> dict[str, int]:
        """返回稳定的相关性维度键名，供持久化与前端展示。"""
        return {
            "core_topic_match": self.core_topic_match_score,
            "information_need_match": self.information_need_match_score,
            "semantic_coverage": self.semantic_coverage_score,
            "directness": self.directness_score,
        }


def calculate_recommendation_score(
    relevance_score: int | float,
    quality_score: int | float,
) -> int:
    """按“搜索体验优先”原则计算 0-100 推荐分。"""
    relevance = max(0.0, min(100.0, float(relevance_score)))
    quality = max(0.0, min(100.0, float(quality_score)))
    return round(
        relevance * RECOMMENDATION_RELEVANCE_WEIGHT
        + quality * RECOMMENDATION_QUALITY_WEIGHT
    )


class BodyQualitySchema(BaseModel):
    """AI 对新闻正文质量的结构化评分，总分 50 分。"""

    evidence_score: int = Field(
        ge=0,
        le=15,
        description="关键事实和结论是否有正文内证据支撑，0-15 分",
    )
    evidence_reason: str = Field(description="证据充分性评分依据")
    completeness_score: int = Field(
        ge=0,
        le=10,
        description="正文是否交代事件、背景、主体、影响和必要上下文，0-10 分",
    )
    completeness_reason: str = Field(description="报道完整性评分依据")
    transparency_score: int = Field(
        ge=0,
        le=10,
        description="信源、引用、数据口径和研究方法是否透明，0-10 分",
    )
    transparency_reason: str = Field(description="透明度评分依据")
    headline_body_consistency_score: int = Field(
        ge=0,
        le=5,
        description="标题是否准确反映正文且不夸大，0-5 分",
    )
    headline_body_consistency_reason: str = Field(description="标题正文一致性评分依据")
    balance_score: int = Field(
        ge=0,
        le=5,
        description="表述是否客观，并说明局限、风险或不同立场，0-5 分",
    )
    balance_reason: str = Field(description="客观与平衡性评分依据")
    clarity_score: int = Field(
        ge=0,
        le=5,
        description="结构是否清晰、逻辑是否连贯、表达是否准确，0-5 分",
    )
    clarity_reason: str = Field(description="清晰与连贯性评分依据")
    has_serious_unsupported_claims: bool = Field(
        description="是否存在会显著影响结论、但正文没有证据支撑的严重断言"
    )
    unsupported_claims_reason: str = Field(
        description="严重无支持断言的说明；若不存在则返回空字符串"
    )


class NewsCaseSchema(BaseModel):
    """
    新闻商业案例结构化输出 Schema。

    AI 模型将严格按照此 Pydantic 模型返回 JSON，
    确保字段类型与业务语义一致。
    """

    title: str = Field(description="新闻标题，简洁准确")
    url: str = Field(description="原文网页链接")
    summary: str = Field(description="80-150 字的客观新闻摘要")
    bullet_points: list[str] = Field(
        description=(
            "从文章中提取的带具体数字的清单化案例条目，"
            "每条应包含可量化的商业指标（金额、百分比、客流量、产能等）"
        )
    )
    evidence_quotes: list[str] = Field(
        description="支撑量化案例的原文短句，不得改写或编造"
    )
    involved_companies: list[str] = Field(description="文章涉及的主要企业/机构")
    regions: list[str] = Field(description="文章涉及的主要国家或地区")
    metric_tags: list[str] = Field(
        description="文章量化指标类型，如营收、RevPAR、客流量、投资额"
    )
    relevance_score: int = Field(
        ge=0,
        le=100,
        description="0-100 的行业相关性打分，100 表示高度相关",
    )
    source_credibility_score: int = Field(
        ge=0,
        le=25,
        description="AI 对新闻原始发布来源的 0-25 权威度评分",
    )
    source_credibility_reason: str = Field(
        description="AI 给出来源权威度分数的简短、可核对依据"
    )
    body_quality: BodyQualitySchema = Field(
        description="AI 对正文质量六个子维度的结构化评分，总分 50 分"
    )


class ArticleAnalysisError(RuntimeError):
    """单篇新闻经过重试后仍无法完成 AI 分析。"""


def _classify_openai_error(exc: Exception, model_name: str) -> str:
    """将 OpenAI SDK 异常转换为不包含密钥的用户可读信息。"""
    message = str(exc).lower()
    if any(key in message for key in ("429", "quota", "rate", "insufficient_quota")):
        return "OpenAI 配额不足或请求频率受限（429）"
    if any(key in message for key in ("401", "403", "api key", "unauthorized")):
        return "OpenAI API Key 无效、已失效或无模型访问权限"
    if "not found" in message or "404" in message:
        return f"OpenAI 模型不可用：{model_name}"
    if any(key in message for key in ("timeout", "timed out")):
        return "OpenAI 请求超时"
    return f"OpenAI API 调用失败（{type(exc).__name__}）"


def _classify_gemini_error(exc: Exception, model_name: str) -> str:
    """将 SDK 异常转换为不包含密钥的用户可读信息。"""
    message = str(exc).lower()
    if any(key in message for key in ("429", "quota", "rate", "resource_exhausted")):
        return "Gemini 配额不足或请求频率受限（429）"
    if any(key in message for key in ("401", "403", "api key", "permission_denied")):
        return "Gemini API Key 无效、已失效或无模型访问权限"
    if "not found" in message or "404" in message:
        return f"Gemini 模型不可用：{model_name}"
    if any(key in message for key in ("timeout", "deadline", "timed out")):
        return "Gemini 请求超时"
    return f"Gemini API 调用失败（{type(exc).__name__}）"


def _qualification_reasons(result: NewsCaseSchema, min_score: int) -> list[str]:
    """返回分析结果未进入案例库的具体原因。"""
    reasons: list[str] = []
    if result.relevance_score < min_score:
        reasons.append(f"相关性 {result.relevance_score} 低于门槛 {min_score}")
    if not result.bullet_points:
        reasons.append("未提取到量化案例")
    if not result.evidence_quotes:
        reasons.append("没有可在正文中验证的证据原文")
    return reasons


def _notify_progress(
    callback: Callable[[str, float], None] | None,
    message: str,
    value: float,
) -> None:
    """通知前端任务阶段；前端异常不应中断采集流水线。"""
    if callback is None:
        return
    try:
        callback(message, max(0.0, min(1.0, value)))
    except Exception as exc:
        logger.warning("更新任务进度失败: %s", type(exc).__name__)


# 商业分析师 Prompt 模板
ANALYST_SYSTEM_PROMPT = """你是一位资深商业分析师，擅长撰写行业深度研究报告。

你的任务是从给定的新闻正文中，提炼具有商业价值的量化案例，要求：

1. **量化优先**：bullet_points 中每条必须包含具体数字（金额、百分比、人数、规模、增长率等），
   避免空泛描述。示例：
   - "某头部企业 Q2 营收达 45.6 亿元，同比增长 23%"
   - "标杆工厂智能化改造后单位成本下降 18%，年节约 2.3 亿元"
   - "行业日均客流量突破 850 万人次，较去年同期增长 31%"

2. **主题聚焦**：仅提取与目标行业「{industry_keyword}」和研究主题
   「{topic_id} {topic_name}」直接相关的商业洞察。

3. **严谨语调**：模仿券商/咨询机构行业深度报告的表述风格，客观、精炼、数据驱动。

4. **relevance_score 评分标准**：
   - 90-100：核心行业动态，含丰富量化数据
   - 70-89：相关行业资讯，有一定量化信息
   - 50-69：间接相关或量化信息不足
   - 0-49：关联度低或缺乏商业价值

5. **source_credibility_score 来源权威度（0-25）**：
   评估原始发布者，不要把 Google News、Bing 等聚合页当作原始来源。
   结合官网域名、编辑/研究标准、一手资料属性、方法透明度和利益偏向评分：
   - 23-25：政府、国际公共机构、顶级通讯社，或方法透明的顶级独立研究机构
   - 19-22：大型主流媒体、专业研究机构、知名企业官方报告（一手但可能存在自述偏向）
   - 14-18：可验证的行业媒体、一般企业官网或有明确编辑责任的来源
   - 8-13：网站身份或编辑标准无法充分确认；信息不足时应保守评分
   - 0-7：内容农场、个人博客、无原始出处的聚合站或明显可疑来源
   source_credibility_reason 需说明判断依据；不得编造网站背景、奖项或影响力数据。

6. **body_quality 正文质量（0-50）**：只根据输入的正文评分，不要根据来源名补分，
   六个子维度必须分别给出分数和简短依据：
   - evidence_score（0-15）：关键事实、数据与结论是否有正文内证据支撑，证据能否核对
   - completeness_score（0-10）：是否交代事件、背景、主体、影响与必要上下文
   - transparency_score（0-10）：是否说明具名信源、直接引用、数据口径或研究方法
   - headline_body_consistency_score（0-5）：标题是否准确反映正文且不过度夸大
   - balance_score（0-5）：是否区分事实与观点，并呈现必要的局限、风险或不同立场
   - clarity_score（0-5）：结构是否清晰、逻辑是否连贯、表达是否准确
   has_serious_unsupported_claims 仅在影响核心结论的断言缺乏正文证据时设为 true，
   不要因为无法联网进行外部核验就判定为 true。

7. **证据可追溯**：evidence_quotes 只能摘录输入正文中真实存在的短句，
   不得补全、改写或编造数字。

8. 若正文无法提取有效量化案例，bullet_points 和 evidence_quotes 可为空数组，
   relevance_score 应相应降低。

请基于以下输入进行分析，并严格按 JSON Schema 返回结果。"""


def _get_gemini_client() -> genai.Client:
    """
    初始化 Gemini 客户端。

    Raises:
        ValueError: 未配置 GEMINI_API_KEY 时抛出
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "未配置有效的 GEMINI_API_KEY，请在 .env 文件中设置你的 Google AI API Key"
        )
    return genai.Client(api_key=api_key)


def _get_openai_client() -> OpenAI:
    """初始化 OpenAI 客户端，不在日志中暴露密钥。"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "未配置有效的 OPENAI_API_KEY，请在 .env 或 Streamlit Secrets 中设置"
        )
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_user_prompt(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
) -> str:
    """组装发送给 AI 模型的用户侧 Prompt。"""
    topic = topic or {}
    return f"""【目标行业】{industry_keyword}

【研究主题】{topic.get('topic_id', 'CUSTOM')} {topic.get('topic_name', '自定义行业研究')}
【主题分类】{topic.get('dimension', '自定义')} / {topic.get('category', '自定义')}

【新闻标题】{article_title}

【原文链接】{article_url}

【新闻正文】
{article_text}

请输出 title、url、summary、bullet_points、evidence_quotes、
involved_companies、regions、metric_tags、relevance_score、
source_credibility_score、source_credibility_reason、body_quality 字段。
其中 title 和 url 请直接使用上方提供的值。"""


def _keep_verifiable_evidence(quotes: list[str], article_text: str) -> list[str]:
    """只保留去除空白后能在输入正文中完整匹配的证据短句。"""
    normalized_article = re.sub(r"\s+", "", article_text)
    return [
        quote.strip()
        for quote in quotes
        if quote.strip() and re.sub(r"\s+", "", quote.strip()) in normalized_article
    ]


def _prepare_analysis_prompts(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None,
) -> tuple[str, str]:
    """生成两种模型提供方共用的 system/user prompts。"""
    topic = topic or {}
    system_prompt = ANALYST_SYSTEM_PROMPT.format(
        industry_keyword=industry_keyword,
        topic_id=topic.get("topic_id", "CUSTOM"),
        topic_name=topic.get("topic_name", "自定义行业研究"),
    )
    user_prompt = _build_user_prompt(
        article_title, article_url, article_text, industry_keyword, topic
    )
    return system_prompt, user_prompt


SEARCH_INTENT_SYSTEM_PROMPT = """你是一名新闻检索策略师。
先判断用户的输入是否范围过大或存在多种合理解释，再生成检索策略。

1. scope_level：
   - broad：仅有“AI”“芯片”等大类，可能产生大量无关新闻。
   - focused：有主题和方向，但仍可能存在语义歧义。
   - specific：对象、范围或想了解的问题已经清晰。
2. 若范围过大或存在歧义，needs_clarification 必须为 true，只提出一个简洁问题，
   并返回 2-4 个互斥、可操作的 interpretations。每个解释都必须自带最终搜索目标、
   中英文检索词和相关性判断标准，便于用户选中后直接搜索。
3. 若意图清晰，needs_clarification 为 false、interpretations 为空，但仍返回一句可供用户确认的 intent_summary。

保留用户的核心主题，可补充“新趋势、技术路线、市场、政策、产业化”等
与问题直接相关的角度，但不得凭空限定某家企业、某项技术或某个结论。
中文检索词返回 1-3 个，英文检索词返回 1-2 个；每个检索词都要尽量短。
英文检索词必须是对搜索意图的英文表达，不得把中文原词原样填入 english_queries。
相关性标准必须能用来判断一篇新闻是否回答了用户的问题。"""


def fallback_search_intent(
    query: str,
    english_query: str | None = None,
) -> SearchIntentSchema:
    """AI 意图理解失败时保留原始查询，不阻断新闻搜索。"""
    topic_parts = [
        part.strip()
        for part in re.split(r"\s*[+＋]\s*", query.strip())
        if part.strip()
    ]
    cleaned_query = " ".join(topic_parts) or query.strip()
    english = (english_query or "").strip()
    topic_description = "、".join(topic_parts) if topic_parts else cleaned_query
    normalized_query = re.sub(r"\s+", "", cleaned_query).casefold()
    interpretations: list[SearchInterpretationSchema] = []
    if normalized_query in {"ai", "人工智能"}:
        interpretations = [
            SearchInterpretationSchema(
                label="广义 AI 行业",
                description="覆盖模型、芯片、应用、投融资与政策等整体动态",
                intent_summary="了解全球人工智能行业的重要新闻、技术趋势和市场变化",
                target_topics=["人工智能", "技术趋势", "市场动态"],
                chinese_queries=["AI 行业 最新动态", "人工智能 技术 市场"],
                english_queries=["artificial intelligence industry news"],
                relevance_criteria=["新闻核心事件直接影响 AI 技术、市场或政策"],
            ),
            SearchInterpretationSchema(
                label="AI 应用与商业化",
                description="关注 AI 在具体行业的产品应用、客户案例和投入产出",
                intent_summary="查找 AI 在各行业的产品应用、商业化案例和可量化效果",
                target_topics=["AI 应用", "商业化", "行业案例"],
                chinese_queries=["AI 行业应用 商业化 案例"],
                english_queries=["AI applications commercialization case studies"],
                relevance_criteria=["新闻提供具体 AI 应用场景、客户或量化效果"],
            ),
            SearchInterpretationSchema(
                label="大模型与基础设施",
                description="关注基础模型、AI 芯片、算力、数据中心和开发工具",
                intent_summary="查找大模型、AI 芯片、算力基础设施和开发工具的最新动态",
                target_topics=["大模型", "AI 芯片", "算力基础设施"],
                chinese_queries=["大模型 AI 芯片 算力 最新动态"],
                english_queries=["foundation models AI chips compute infrastructure"],
                relevance_criteria=["新闻核心讨论模型、算力或 AI 基础设施"],
            ),
        ]
        scope_level: Literal["broad", "focused", "specific"] = "broad"
        clarification_question = "“AI”范围较广，您想检索广义 AI 新闻，还是某个具体方向？"
    elif len(topic_parts) >= 2:
        scope_level = "focused"
        clarification_question = (
            f"“{topic_description}”可能存在多种关系，请确认您具体想了解的对象和方向。"
        )
    else:
        scope_level = "broad" if len(cleaned_query) <= 6 else "specific"
        clarification_question = (
            f"请确认：您是否希望按“{cleaned_query}”这个目标搜索新闻？"
        )
    return SearchIntentSchema(
        intent_summary=f"查找与“{cleaned_query}”直接相关的最新新闻与产业信息",
        target_topics=topic_parts or [cleaned_query],
        chinese_queries=[cleaned_query],
        english_queries=[english] if english else [],
        relevance_criteria=[
            f"新闻核心内容实质涉及“{topic_description}”",
            f"正文说明“{topic_description}”之间的具体关系，并提供可核对的事实、趋势或数据",
        ],
        scope_level=scope_level,
        needs_clarification=True,
        clarification_question=clarification_question,
        interpretations=interpretations,
        recommended_interpretation_index=0,
    )


def _supports_openai_responses_api(provider_name: str) -> bool:
    """官方 OpenAI 使用 Responses；自定义兼容地址默认使用 Chat Completions。"""
    if provider_name != "openai":
        return False
    base_url = os.getenv("OPENAI_BASE_URL", "").strip().lower()
    return not base_url or "api.openai.com" in base_url


def _request_chat_completion_json(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    provider_name: str,
) -> dict[str, Any]:
    """使用 OpenAI 兼容的 Chat Completions 获取 JSON 对象。"""
    request_options: dict[str, Any] = {}
    if provider_name == "deepseek":
        thinking_mode = os.getenv(
            "DEEPSEEK_THINKING", "disabled"
        ).strip().lower()
        if thinking_mode not in {"enabled", "disabled"}:
            thinking_mode = "disabled"
        request_options["extra_body"] = {
            "thinking": {"type": thinking_mode}
        }
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt + "\nRespond with a valid JSON object.",
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        **request_options,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Chat Completions 返回空响应")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Chat Completions 未返回 JSON 对象")
    return parsed


def analyze_search_intent(
    query: str,
    provider: str | None = None,
    model: str | None = None,
) -> SearchIntentSchema:
    """在抓取前理解用户问题，生成中英文新闻检索词。"""
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("搜索内容不能为空")
    provider_name = (provider or get_ai_provider()).strip().lower()
    if provider_name not in {"openai", "gemini", "deepseek"}:
        raise ValueError(
            f"不支持的 AI_PROVIDER：{provider_name}，可选值为 openai、gemini 或 deepseek"
        )
    model_name = (
        model or get_gemini_model()
        if provider_name == "gemini"
        else model or get_openai_model()
    )
    user_prompt = f"【用户原始搜索】{cleaned_query}"
    last_error = "AI 未返回有效的搜索意图"

    for attempt in range(1, 3):
        try:
            if provider_name == "gemini":
                response = _get_gemini_client().models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=SEARCH_INTENT_SYSTEM_PROMPT),
                                types.Part(text=user_prompt),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SearchIntentSchema,
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini 返回空响应")
                return SearchIntentSchema.model_validate_json(response.text)

            client = _get_openai_client()
            if _supports_openai_responses_api(provider_name):
                response = client.responses.parse(
                    model=model_name,
                    input=[
                        {"role": "system", "content": SEARCH_INTENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=SearchIntentSchema,
                )
                if response.output_parsed is None:
                    raise ValueError("OpenAI 返回空响应")
                return response.output_parsed

            return SearchIntentSchema.model_validate(
                _request_chat_completion_json(
                    client,
                    model_name,
                    SEARCH_INTENT_SYSTEM_PROMPT,
                    user_prompt,
                    provider_name,
                )
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = f"搜索意图结构化输出无效（{type(exc).__name__}）"
        except Exception as exc:
            if provider_name == "gemini":
                last_error = _classify_gemini_error(exc, model_name)
            else:
                last_error = _classify_openai_error(exc, model_name)
        if attempt < 2:
            time.sleep(1.0)

    raise ArticleAnalysisError(f"搜索意图 AI 理解失败：{last_error}")


SOURCE_CREDIBILITY_SYSTEM_PROMPT = """你是一名新闻搜索初筛审查员。

你需要独立完成两项评分：
1. 搜索相关性（0-100）：不要直接给总分，分别评估四个维度，总分由程序汇总。
   - core_topic_match_score（0-40）：新闻核心主体是否就是用户搜索的行业、企业、事件或技术。
   - information_need_match_score（0-30）：是否回答用户具体想了解的“新方向、市场、政策、案例”等问题。
   - semantic_coverage_score（0-20）：正文是否实质讨论相关内容，不能因为只出现一两个相同词就给高分。
   - directness_score（0-10）：搜索主题是新闻的核心，还是只在背景中顺带提及。
   四维合计的解读：80-100 高度相关；60-79 明显相关；
   40-59 间接相关；0-39 基本无关。
2. score（0-25）：新闻原始信息来源的权威度。不分析正文写作质量。

不要把 Google News、Bing、RSS 等聚合页当作原始发布者。结合提供的来源名、
最终文章域名、标题和正文开头判断，并严格返回 0-25 分：

- 23-25：政府、国际公共机构、顶级通讯社，或方法透明的顶级独立研究机构
- 19-22：大型主流媒体、专业研究机构、知名企业官方一手报告
- 14-18：可验证的行业媒体、一般企业官网或有明确编辑责任的来源
- 8-13：网站身份或编辑标准无法充分确认；信息不足时保守评分
- 0-7：内容农场、个人博客、无原始出处的聚合站或明显可疑来源

企业官方材料具有一手性，但应考虑自述偏向。不得编造机构背景、奖项、受众规模
或编辑制度。reason 和 relevance_reason 必须简短说明可核对的判断依据。"""


def _build_source_credibility_prompt(
    article: dict[str, Any],
    original_query: str = "",
    search_intent: SearchIntentSchema | dict[str, Any] | None = None,
) -> str:
    """组装搜索相关性与来源权威度初筛输入。"""
    content_excerpt = str(article.get("content", "")).strip()[:1500]
    if isinstance(search_intent, SearchIntentSchema):
        intent_data = search_intent.model_dump()
    elif isinstance(search_intent, dict):
        intent_data = search_intent
    else:
        intent_data = fallback_search_intent(original_query or "当前搜索").model_dump()
    return f"""【用户原始搜索】{original_query}
【AI 理解的搜索意图】{intent_data.get('intent_summary', '')}
【目标主题】{' / '.join(intent_data.get('target_topics', []) or [])}
【相关性判断标准】{' / '.join(intent_data.get('relevance_criteria', []) or [])}

【来源名称】{article.get('source', '未知来源')}
【文章网址】{article.get('url', '')}
【新闻标题】{article.get('title', '')}
【正文开头】
{content_excerpt}

请只输出 score、reason、core_topic_match_score、
information_need_match_score、semantic_coverage_score、directness_score 和
relevance_reason 字段。不要另外输出相关性总分。"""


def analyze_source_credibility(
    article: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    original_query: str = "",
    search_intent: SearchIntentSchema | dict[str, Any] | None = None,
) -> SourceCredibilitySchema:
    """先于正文分析，评估搜索相关性与来源权威度。"""
    provider_name = (provider or get_ai_provider()).strip().lower()
    if provider_name not in {"openai", "gemini", "deepseek"}:
        raise ValueError(
            f"不支持的 AI_PROVIDER：{provider_name}，可选值为 openai、gemini 或 deepseek"
        )

    if provider_name == "gemini":
        model_name = model or get_gemini_model()
    else:
        model_name = model or get_openai_model()
    user_prompt = _build_source_credibility_prompt(
        article,
        original_query=original_query,
        search_intent=search_intent,
    )
    last_error = "AI 未返回有效的相关性与来源初筛结果"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider_name == "gemini":
                response = _get_gemini_client().models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(text=SOURCE_CREDIBILITY_SYSTEM_PROMPT),
                                types.Part(text=user_prompt),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SourceCredibilitySchema,
                        temperature=0.1,
                    ),
                )
                if not response.text:
                    raise ValueError("Gemini 返回空响应")
                return SourceCredibilitySchema.model_validate_json(response.text)

            client = _get_openai_client()
            if _supports_openai_responses_api(provider_name):
                response = client.responses.parse(
                    model=model_name,
                    input=[
                        {"role": "system", "content": SOURCE_CREDIBILITY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=SourceCredibilitySchema,
                )
                if response.output_parsed is None:
                    raise ValueError("OpenAI 返回空响应")
                return response.output_parsed

            return SourceCredibilitySchema.model_validate(
                _request_chat_completion_json(
                    client,
                    model_name,
                    SOURCE_CREDIBILITY_SYSTEM_PROMPT,
                    user_prompt,
                    provider_name,
                )
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = f"相关性与来源初筛输出无效（{type(exc).__name__}）"
        except Exception as exc:
            if provider_name == "gemini":
                last_error = _classify_gemini_error(exc, model_name)
            else:
                last_error = _classify_openai_error(exc, model_name)
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    raise ArticleAnalysisError(f"相关性与来源 AI 初筛失败：{last_error}")


def score_sources_with_ai(
    articles: list[dict[str, Any]],
    original_query: str = "",
    search_intent: SearchIntentSchema | dict[str, Any] | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[str]:
    """并行生成搜索相关性和来源 AI 分数。"""
    pending_indexes = [
        index
        for index, article in enumerate(articles)
        if not (
            isinstance(article.get("quality_pre"), QualitySummary)
            and article["quality_pre"].source_score_method == "ai"
            and article.get("search_relevance_scored") is True
        )
    ]
    if not pending_indexes:
        return []

    worker_count = min(get_ai_analysis_workers(), len(pending_indexes))
    errors: list[str] = []
    _notify_progress(
        progress_callback,
        f"AI 正在初筛 {len(pending_indexes)} 篇新闻的相关性与来源（{worker_count} 路）…",
        0.05,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                analyze_source_credibility,
                articles[index],
                original_query=original_query,
                search_intent=search_intent,
            ): index
            for index in pending_indexes
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            article = articles[index]
            try:
                result = future.result()
                pre_score = article.get("quality_pre")
                if not isinstance(pre_score, QualitySummary):
                    pre_score = score_article_pre_ai(
                        title=str(article.get("title", "")),
                        url=str(article.get("url", "")),
                        content=str(article.get("content", "")),
                        source=str(article.get("source", "")),
                        published_at=str(article.get("published_at", "")),
                        keyword="",
                        content_hash=str(article.get("content_hash", "")),
                    )
                    article["quality_pre"] = pre_score
                apply_ai_source_score(pre_score, result.score, result.reason)
                article["source_ai_scored"] = True
                article["search_relevance_score"] = result.relevance_score
                article["search_relevance_dimensions"] = (
                    result.relevance_dimension_scores()
                )
                article["search_relevance_reason"] = result.relevance_reason
                article["search_relevance_scored"] = True
                article["search_relevance_rule_version"] = (
                    SEARCH_RELEVANCE_RULE_VERSION
                )
                article.pop("source_ai_error", None)
            except Exception as exc:
                reason = str(exc)
                article["source_ai_scored"] = False
                article["source_ai_error"] = reason
                errors.append(f"{article.get('title', '无标题')}：{reason}")
            completed += 1
            _notify_progress(
                progress_callback,
                f"相关性与来源初筛已完成 {completed}/{len(pending_indexes)} 篇…",
                completed / len(pending_indexes),
            )
    articles.sort(
        key=lambda article: int(article.get("search_relevance_score", -1) or 0),
        reverse=True,
    )
    return errors


def _normalize_analysis_result(
    parsed: NewsCaseSchema,
    article_title: str,
    article_url: str,
    article_text: str,
) -> NewsCaseSchema:
    """锁定来源字段，并移除正文中无法验证的证据句。"""
    parsed.title = article_title
    parsed.url = article_url
    parsed.evidence_quotes = _keep_verifiable_evidence(
        parsed.evidence_quotes, article_text
    )
    return parsed


def analyze_article_with_openai(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
    model: str | None = None,
) -> NewsCaseSchema:
    """使用 OpenAI Responses API 对单篇文章做结构化分析。"""
    if not article_text.strip():
        raise ArticleAnalysisError("新闻正文为空")

    client = _get_openai_client()
    model_name = model or get_openai_model()
    system_prompt, user_prompt = _prepare_analysis_prompts(
        article_title,
        article_url,
        article_text,
        industry_keyword,
        topic,
    )

    last_error = "OpenAI 未返回有效结果"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "OpenAI 分析中 (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                article_title[:50],
            )
            response = client.responses.parse(
                model=model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=NewsCaseSchema,
            )
            parsed = response.output_parsed
            if parsed is None:
                last_error = "OpenAI 返回空响应或拒绝处理该内容"
                logger.warning(last_error)
                continue

            parsed = _normalize_analysis_result(
                parsed, article_title, article_url, article_text
            )
            logger.info(
                "分析完成: score=%d, bullets=%d — %s",
                parsed.relevance_score,
                len(parsed.bullet_points),
                article_title[:40],
            )
            return parsed
        except ValidationError:
            last_error = "OpenAI 结构化输出不符合字段要求"
            logger.error(last_error)
        except Exception as exc:
            error_msg = str(exc).lower()
            last_error = _classify_openai_error(exc, model_name)
            is_rate_limit = any(
                key in error_msg for key in ("429", "rate", "quota", "insufficient_quota")
            )
            delay = (
                RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if is_rate_limit
                else RETRY_BASE_DELAY
            )
            logger.warning("%s (attempt %d/%d)", last_error, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    logger.error("OpenAI 分析最终失败: %s — %s", article_title, last_error)
    raise ArticleAnalysisError(last_error)




def analyze_article_with_chat_completions(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
    model: str | None = None,
) -> NewsCaseSchema:
    """使用 OpenAI 兼容的 Chat Completions API 对单篇文章做结构化分析。

    适用于 DeepSeek、硅基流动等仅支持 Chat Completions 的第三方提供方。
    通过 response_format=json_object 强制返回 JSON，手动解析后
    经 Pydantic 校验确保字段完整。
    """
    if not article_text.strip():
        raise ArticleAnalysisError("新闻正文为空")

    client = _get_openai_client()
    model_name = model or get_openai_model()
    system_prompt, user_prompt = _prepare_analysis_prompts(
        article_title,
        article_url,
        article_text,
        industry_keyword,
        topic,
    )
    # JSON mode 硬性要求：system prompt 中必须包含 "json" 字样
    system_prompt = system_prompt + "\nRespond with a valid JSON object."

    last_error = "Chat Completions API 未返回有效结果"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Chat Completions 分析中 (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                article_title[:50],
            )
            request_options: dict[str, Any] = {}
            if get_ai_provider() == "deepseek":
                thinking_mode = os.getenv(
                    "DEEPSEEK_THINKING", "disabled"
                ).strip().lower()
                if thinking_mode not in {"enabled", "disabled"}:
                    thinking_mode = "disabled"
                request_options["extra_body"] = {
                    "thinking": {"type": thinking_mode}
                }

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                **request_options,
            )
            content = response.choices[0].message.content
            if not content:
                last_error = "Chat Completions API 返回空响应"
                logger.warning(last_error)
                continue

            parsed_data = json.loads(content)
            parsed = NewsCaseSchema(**parsed_data)
            parsed = _normalize_analysis_result(
                parsed, article_title, article_url, article_text
            )
            logger.info(
                "分析完成: score=%d, bullets=%d — %s",
                parsed.relevance_score,
                len(parsed.bullet_points),
                article_title[:40],
            )
            return parsed
        except (json.JSONDecodeError, ValidationError):
            last_error = "Chat Completions API 返回 JSON 格式不符合 Schema 要求"
            logger.error(last_error)
        except Exception as exc:
            error_msg = str(exc).lower()
            last_error = _classify_openai_error(exc, model_name)
            is_rate_limit = any(
                key in error_msg for key in ("429", "rate", "quota", "insufficient_quota")
            )
            delay = (
                RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if is_rate_limit
                else RETRY_BASE_DELAY
            )
            logger.warning("%s (attempt %d/%d)", last_error, attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    logger.error("Chat Completions 分析最终失败: %s — %s", article_title, last_error)
    raise ArticleAnalysisError(last_error)
def analyze_article_with_gemini(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
    model: str | None = None,
) -> NewsCaseSchema:
    """
    调用 Gemini 对单篇文章进行结构化案例分析。

    使用 response_mime_type="application/json" + response_schema=NewsCaseSchema
    确保 100% 返回符合 Schema 的 JSON。

    Args:
        article_title: 新闻标题
        article_url: 原文链接
        article_text: 清洗后的正文
        industry_keyword: 行业关键词
        model: Gemini 模型名，默认使用最新 Flash 别名

    Returns:
        NewsCaseSchema 实例

    Raises:
        ArticleAnalysisError: 正文为空或重试后仍无法获得有效结果
    """
    if not article_text.strip():
        raise ArticleAnalysisError("新闻正文为空")

    client = _get_gemini_client()
    model_name = model or get_gemini_model()
    system_prompt, user_prompt = _prepare_analysis_prompts(
        article_title,
        article_url,
        article_text,
        industry_keyword,
        topic,
    )

    last_error = "Gemini 未返回有效结果"
    for attempt in range(1, MAX_RETRIES + 1):
        raw_text = ""
        try:
            logger.info(
                "Gemini 分析中 (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                article_title[:50],
            )

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=system_prompt),
                            types.Part(text=user_prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    # 强制 JSON 输出
                    response_mime_type="application/json",
                    # 使用 Pydantic Schema 约束输出结构
                    response_schema=NewsCaseSchema,
                    temperature=0.2,  # 低温度保证分析稳定性
                ),
            )

            raw_text = response.text
            if not raw_text:
                last_error = "Gemini 返回空响应"
                logger.warning("Gemini 返回空响应")
                continue

            # 解析并校验 JSON
            parsed = NewsCaseSchema.model_validate_json(raw_text)

            # 确保 title/url 与输入一致（防止模型篡改）
            parsed = _normalize_analysis_result(
                parsed, article_title, article_url, article_text
            )

            logger.info(
                "分析完成: score=%d, bullets=%d — %s",
                parsed.relevance_score,
                len(parsed.bullet_points),
                article_title[:40],
            )
            return parsed

        except ValidationError as exc:
            last_error = "Gemini 结构化输出不符合字段要求"
            logger.error("Gemini 返回 JSON 校验失败: %s", type(exc).__name__)
            # 尝试手动解析兜底
            try:
                if raw_text:
                    data = json.loads(raw_text)
                    parsed = NewsCaseSchema.model_validate(data)
                    parsed = _normalize_analysis_result(
                        parsed, article_title, article_url, article_text
                    )
                    return parsed
            except Exception:
                pass

        except Exception as exc:
            error_msg = str(exc).lower()
            last_error = _classify_gemini_error(exc, model_name)
            # 识别限流 / 配额错误，使用指数退避重试
            if any(kw in error_msg for kw in ("429", "rate", "quota", "resource_exhausted")):
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini API 限流/配额不足，%ds 后重试 (attempt %d)",
                    delay,
                    attempt,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(delay)
            else:
                logger.error("Gemini API 调用失败: %s", last_error)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY)

    logger.error("Gemini 分析最终失败: %s — %s", article_title, last_error)
    raise ArticleAnalysisError(last_error)


def analyze_article(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> NewsCaseSchema:
    """按 AI_PROVIDER 将单篇分析路由到 OpenAI 或 Gemini。"""
    provider_name = (provider or get_ai_provider()).strip().lower()
    openai_analyzer = (
        analyze_article_with_openai
        if _supports_openai_responses_api(provider_name)
        else analyze_article_with_chat_completions
    )
    analyzers = {
        "openai": openai_analyzer,
        "deepseek": analyze_article_with_chat_completions,
        "gemini": analyze_article_with_gemini,
    }
    if provider_name not in analyzers:
        raise ValueError(
            f"不支持的 AI_PROVIDER：{provider_name}，可选值为 openai、gemini 或 deepseek"
        )
    return analyzers[provider_name](
        article_title=article_title,
        article_url=article_url,
        article_text=article_text,
        industry_keyword=industry_keyword,
        topic=topic,
        model=model,
    )


def _classify_topic_posthoc(
    result: NewsCaseSchema,
    keyword: str,
    topic_records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """后置主题归类：根据 AI 提取的企业/地区/指标与 28 个主题进行关键词匹配。

    返回最佳匹配的主题字典，若无明显匹配则返回 None。
    """
    if not topic_records:
        return None

    # 构建每篇文章的搜索文本（用于匹配）
    article_texts: list[str] = [keyword, result.summary]
    article_texts.extend(result.involved_companies)
    article_texts.extend(result.regions)
    article_texts.extend(result.metric_tags)
    combined = " ".join(article_texts).lower()

    best_topic: dict[str, Any] | None = None
    best_score = 0

    for topic in topic_records:
        if str(topic.get("topic_id", "")) == "CUSTOM":
            continue
        score = 0
        kw = str(topic.get("search_keywords", "")).lower()
        name = str(topic.get("topic_name", "")).lower()
        dim = str(topic.get("dimension", "")).lower()
        cat = str(topic.get("category", "")).lower()

        # 关键词匹配
        for token in kw.split():
            if len(token) >= 2 and token in combined:
                score += 3
        if len(name) >= 2 and name in combined:
            score += 5
        for token in dim.split():
            if len(token) >= 2 and token in combined:
                score += 2

        if score > best_score:
            best_score = score
            best_topic = dict(topic)

    # 最低匹配阈值
    if best_score < 5:
        return None
    return best_topic

def fetch_and_pre_score(
    industry_keyword: str,
    max_articles: int = 8,
    article_callback: Callable[[dict[str, Any], int, int], None] | None = None,
    english_keyword: str | None = None,
    additional_queries: list[str] | None = None,
    english_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """抓取新闻列表并执行预 AI 质量评分（纯算法），返回文章列表供前端展示。

    与 run_pipeline 的抓取阶段逻辑一致，但不进入 AI 分析。
    调用方拿到文章列表后可先展示卡片，再调用 run_pipeline 进行 AI 分析。

    Args:
        industry_keyword: 行业关键词
        max_articles: 单次最多处理文章数
        article_callback: 每找到一篇有效文章时的回调
        english_keyword: 可选英文检索词
        additional_queries: AI 意图理解生成的额外中文检索词
        english_queries: AI 意图理解生成的英文检索词

    Returns:
        已含 quality_pre 字段的文章列表
    """
    def score_and_notify(article: dict[str, Any], found: int, total: int) -> None:
        article["quality_pre"] = score_article_pre_ai(
            title=article["title"],
            url=article["url"],
            content=article["content"],
            source=str(article.get("source", "")),
            published_at=str(article.get("published_at", "")),
            keyword=industry_keyword,
            content_hash=str(article.get("content_hash", "")),
        )
        if article_callback is not None:
            article_callback(article, found, total)

    articles = fetch_and_extract_batch(
        industry_keyword,
        max_articles=max_articles,
        article_callback=score_and_notify,
        english_query=english_keyword,
        additional_queries=additional_queries,
        english_queries=english_queries,
    )

    # 兼容自定义抓取器或测试替身没有执行回调的情况。
    for article in articles:
        if "quality_pre" not in article:
            score_and_notify(article, 0, max_articles)

    return articles


def _analyze_articles_parallel(
    articles: list[dict[str, Any]],
    industry_keyword: str,
    topic: dict[str, Any],
    progress_callback: Callable[[str, float], None] | None,
) -> list[NewsCaseSchema | Exception | None]:
    """并发执行模型调用，结果按原文章顺序返回。"""
    total = len(articles)
    if total == 0:
        return []

    worker_count = min(get_ai_analysis_workers(), total)
    outcomes: list[NewsCaseSchema | Exception | None] = [None] * total
    _notify_progress(
        progress_callback,
        f"AI 正在并行分析 {total} 篇文章（{worker_count} 路）…",
        0.15,
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                analyze_article,
                article_title=article["title"],
                article_url=article["url"],
                article_text=article["content"],
                industry_keyword=industry_keyword,
                topic=topic,
            ): index
            for index, article in enumerate(articles)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except Exception as exc:
                outcomes[index] = exc
            completed += 1
            _notify_progress(
                progress_callback,
                f"AI 已完成 {completed}/{total} 篇分析…",
                0.15 + 0.7 * (completed / total),
            )

    return outcomes



def run_pipeline(
    industry_keyword: str,
    min_score: int = DEFAULT_MIN_SCORE,
    max_articles: int = 8,
    topic: dict[str, Any] | None = None,
    trigger_type: str = "manual",
    progress_callback: Callable[[str, float], None] | None = None,
    pre_fetched_articles: list[dict[str, Any]] | None = None,
    pre_screen_completed: bool = False,
) -> dict[str, Any]:
    """
    执行完整的「抓取 → 提炼 → 入库」工作流。

    Args:
        industry_keyword: 行业关键词
        min_score: 入库最低相关性分数
        max_articles: 单次最多处理文章数
        topic: 预设的研究主题字典（可选）
        trigger_type: 触发方式标识（manual / scheduled）
        progress_callback: 进度回调 (message, progress_float)
        pre_fetched_articles: 预先抓取的带 pre-AI 评分的文章列表，提供时跳过抓取阶段
        pre_screen_completed: 调用方是否已执行过相关性与来源 AI 初筛

    Returns:
        运行摘要字典，包含 AI、新闻入库、案例入库及错误统计
    """
    topic = topic or {}
    summary: dict[str, Any] = {
        "keyword": industry_keyword,
        "topic_id": topic.get("topic_id", "CUSTOM"),
        "topic_name": topic.get("topic_name", "自定义行业研究"),
        "trigger_type": trigger_type,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed": 0,
        "source_scored": 0,
        "source_failed": 0,
        "relevance_skipped": 0,
        "analyzed": 0,
        "analysis_failed": 0,
        "news_saved": 0,
        "refreshed": 0,
        "saved": 0,
        "unqualified": 0,
        "duplicates": 0,
        "write_failed": 0,
        "skipped": 0,
        "errors": [],
        "cases": [],
        "details": [],
    }
    case_detail_indexes: list[int] = []

    logger.info(
        "=== 开始 Industry News Agent 流水线 === topic=%s keyword=%s",
        summary["topic_id"],
        industry_keyword,
    )

    try:
        if pre_fetched_articles is None:
            _notify_progress(progress_callback, "正在搜索新闻并提取正文…", 0.05)
            try:
                articles = fetch_and_extract_batch(
                    industry_keyword, max_articles=max_articles
                )

                # 预 AI 质量评分（纯算法，零成本）
                for article in articles:
                    article["quality_pre"] = score_article_pre_ai(
                        title=article["title"],
                        url=article["url"],
                        content=article["content"],
                        source=str(article.get("source", "")),
                        published_at=str(article.get("published_at", "")),
                        keyword=industry_keyword,
                        content_hash=str(article.get("content_hash", "")),
                    )
            except Exception as exc:
                error = f"抓取失败：{type(exc).__name__}: {exc}"
                logger.error("抓取阶段异常: %s", type(exc).__name__)
                summary["errors"].append(error)
                return summary

            if not articles:
                summary["errors"].append("未获取到有效文章")
                logger.warning("流水线结束：无可用文章")
                return summary
        else:
            articles = pre_fetched_articles

        for article in articles:
            if not isinstance(article.get("quality_pre"), QualitySummary):
                article["quality_pre"] = score_article_pre_ai(
                    title=str(article.get("title", "")),
                    url=str(article.get("url", "")),
                    content=str(article.get("content", "")),
                    source=str(article.get("source", "")),
                    published_at=str(article.get("published_at", "")),
                    keyword=industry_keyword,
                    content_hash=str(article.get("content_hash", "")),
                )

        if not pre_screen_completed and any(
            article["quality_pre"].source_score_method != "ai"
            or article.get("search_relevance_scored") is not True
            for article in articles
        ):
            source_errors = score_sources_with_ai(
                articles,
                original_query=industry_keyword,
                search_intent=fallback_search_intent(industry_keyword),
                progress_callback=lambda message, value: _notify_progress(
                    progress_callback,
                    message,
                    0.05 + 0.1 * value,
                ),
            )
            if source_errors:
                logger.warning(
                    "%d 篇来源预评分失败，将由完整分析结果兜底",
                    len(source_errors),
                )

        summary["source_scored"] = sum(
            1
            for article in articles
            if article["quality_pre"].source_score_method == "ai"
        )
        summary["source_failed"] = len(articles) - summary["source_scored"]

        total_articles = len(articles)
        _notify_progress(
            progress_callback,
            f"已提取 {total_articles} 篇正文，开始相关新闻的 AI 正文分析…",
            0.15,
        )
        eligible_indexes = [
            index
            for index, article in enumerate(articles)
            if article.get("search_relevance_scored") is not True
            or int(article.get("search_relevance_score", 0) or 0)
            >= BODY_ANALYSIS_RELEVANCE_THRESHOLD
        ]
        eligible_articles = [articles[index] for index in eligible_indexes]
        eligible_outcomes = _analyze_articles_parallel(
            eligible_articles,
            industry_keyword,
            topic,
            progress_callback,
        )
        analysis_outcomes: list[NewsCaseSchema | Exception | None] = [
            None
        ] * total_articles
        for article_index, outcome in zip(
            eligible_indexes,
            eligible_outcomes,
            strict=False,
        ):
            analysis_outcomes[article_index] = outcome

        for index, article in enumerate(articles, start=1):
            summary["processed"] += 1
            detail = {
                "title": str(article.get("title", "")),
                "url": str(article.get("url", "")),
                "score": None,
                "analysis_status": "失败",
                "qualification_status": "-",
                "storage_status": "未写入",
                "reason": "",
            }
            if article.get("search_relevance_scored") is True:
                screening_score = int(
                    article.get("search_relevance_score", 0) or 0
                )
                detail["score"] = screening_score
                detail["relevance_reason"] = str(
                    article.get("search_relevance_reason", "")
                )
                if screening_score < BODY_ANALYSIS_RELEVANCE_THRESHOLD:
                    detail["analysis_status"] = "跳过"
                    detail["qualification_status"] = "相关性不足"
                    detail["reason"] = (
                        f"搜索相关性 {screening_score}/100，低于正文 AI "
                        f"分析门槛 {BODY_ANALYSIS_RELEVANCE_THRESHOLD}"
                    )
                    summary["relevance_skipped"] += 1
                    summary["skipped"] += 1
                    summary["details"].append(detail)
                    continue
            try:
                outcome = analysis_outcomes[index - 1]
                if isinstance(outcome, Exception):
                    raise outcome
                result = outcome
            except ValueError as exc:
                reason = str(exc)
                summary["analysis_failed"] += 1
                summary["skipped"] += 1
                summary["errors"].append(reason)
                detail["reason"] = reason
                summary["details"].append(detail)
                logger.error("致命配置错误，终止流水线")
                break
            except ArticleAnalysisError as exc:
                reason = str(exc)
                summary["analysis_failed"] += 1
                summary["skipped"] += 1
                summary["errors"].append(
                    f"AI 分析失败 [{detail['title'][:60]}]：{reason}"
                )
                detail["reason"] = reason
                summary["details"].append(detail)
                continue
            except Exception as exc:
                reason = f"未预期的分析异常：{type(exc).__name__}"
                summary["analysis_failed"] += 1
                summary["skipped"] += 1
                summary["errors"].append(
                    f"AI 分析失败 [{detail['title'][:60]}]：{reason}"
                )
                detail["reason"] = reason
                summary["details"].append(detail)
                logger.error("单篇分析异常: %s", type(exc).__name__)
                continue

            if result is None:
                reason = "AI 模型未返回可用分析结果"
                summary["analysis_failed"] += 1
                summary["skipped"] += 1
                summary["errors"].append(
                    f"AI 分析失败 [{detail['title'][:60]}]：{reason}"
                )
                detail["reason"] = reason
                summary["details"].append(detail)
                continue

            if article.get("search_relevance_scored") is True:
                result.relevance_score = int(
                    article.get("search_relevance_score", 0) or 0
                )
            else:
                article["search_relevance_score"] = result.relevance_score
                article["search_relevance_reason"] = "由正文 AI 分析补充"
                article["search_relevance_scored"] = True

            summary["analyzed"] += 1
            detail["score"] = result.relevance_score
            detail["analysis_status"] = "成功"
            detail["summary"] = result.summary

            # 后置主题归类（无预设主题时）
            resolved_topic = topic
            if not topic or topic.get("topic_id") == "CUSTOM":
                topics_df = load_topics(enabled_only=True)
                topic_records = topics_df.to_dict(orient="records")
                matched = _classify_topic_posthoc(
                    result, industry_keyword, topic_records
                )
                if matched:
                    resolved_topic = matched
                    detail["auto_topic"] = matched.get("topic_id", "")
                    logger.info(
                        "自动归类: %s → %s",
                        result.title[:30],
                        matched.get("topic_id"),
                    )

            # AI 后质量增强
            pre_score = article.get("quality_pre")
            ai_data = {
                "title": result.title,
                "content": article.get("content", ""),
                "bodyQuality": result.body_quality.model_dump(),
            }
            if not (
                isinstance(pre_score, QualitySummary)
                and pre_score.source_score_method == "ai"
            ):
                ai_data["sourceCredibilityScore"] = result.source_credibility_score
                ai_data["sourceCredibilityReason"] = result.source_credibility_reason
            if isinstance(pre_score, QualitySummary):
                quality = enrich_with_ai_result(pre_score, ai_data)
            else:
                quality = QualitySummary()
            if quality.source_score_method == "ai":
                article["source_ai_scored"] = True
            detail["quality_score"] = quality.adjusted_score
            detail["quality_label"] = quality.label
            recommendation_score = calculate_recommendation_score(
                result.relevance_score,
                quality.adjusted_score,
            )
            detail["recommendation_score"] = recommendation_score

            case_dict = {
                "discovered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "published_at": article.get("published_at", ""),
                "source": article.get("source", ""),
                "content_hash": article.get("content_hash", ""),
                "title": result.title,
                "url": result.url,
                "summary": result.summary,
                "bullet_points": result.bullet_points,
                "evidence_quotes": result.evidence_quotes,
                "involved_companies": result.involved_companies,
                "regions": result.regions,
                "metric_tags": result.metric_tags,
                "relevance_score": result.relevance_score,
                "industry_keyword": industry_keyword,
                "topic_id": resolved_topic.get("topic_id", summary["topic_id"]),
                "topic_name": resolved_topic.get("topic_name", summary["topic_name"]),
                "dimension": resolved_topic.get("dimension", topic.get("dimension", "自定义")),
                "category": resolved_topic.get("category", topic.get("category", "自定义")),
                "quality_score": quality.adjusted_score,
                "quality_details": {
                    "total_score": quality.total_score,
                    "adjusted_score": quality.adjusted_score,
                    "dimension_scores": quality.dimension_scores,
                    "dimension_reasons": quality.dimension_reasons,
                    "body_quality_score": sum(
                        quality.dimension_scores.get(key, 0)
                        for key in (
                            "evidence_quality",
                            "completeness",
                            "transparency",
                            "headline_body_consistency",
                            "balance",
                            "clarity",
                        )
                    ),
                    "penalties": [
                        {"reason": p.reason, "deduction": p.deduction}
                        for p in quality.penalties
                    ],
                    "label": quality.label,
                    "label_description": quality.label_description,
                    "rule_version": quality.rule_version,
                    "ai_confidence": quality.ai_confidence,
                    "source_score_method": quality.source_score_method,
                    "score_cap": quality.score_cap,
                    "quality_warnings": quality.quality_warnings,
                    "search_relevance_score": result.relevance_score,
                    "search_relevance_reason": str(
                        article.get("search_relevance_reason", "")
                    ),
                    "search_relevance_dimensions": dict(
                        article.get("search_relevance_dimensions", {}) or {}
                    ),
                    "search_relevance_rule_version": article.get(
                        "search_relevance_rule_version", ""
                    ),
                    "recommendation_score": recommendation_score,
                },
            }
            detail["quality_details"] = case_dict["quality_details"]
            summary["cases"].append(case_dict)

            qualification_reasons = _qualification_reasons(result, min_score)
            if qualification_reasons:
                summary["unqualified"] += 1
                summary["skipped"] += 1
                detail["qualification_status"] = "未达标"
                detail["reason"] = "；".join(qualification_reasons)
                logger.info(
                    "分数/量化案例/证据未达门槛，仅保留新闻池: %s",
                    result.title[:40],
                )
            else:
                detail["qualification_status"] = "达标"

            summary["details"].append(detail)
            case_detail_indexes.append(len(summary["details"]) - 1)

        if summary["cases"]:
            _notify_progress(progress_callback, "正在写入新闻池并检查重复…", 0.9)
            try:
                write_summary = append_cases_batch_with_summary(
                    summary["cases"], min_score=min_score
                )
                summary["news_saved"] = int(write_summary["news_inserted"])
                summary["refreshed"] = int(write_summary.get("refreshed", 0))
                summary["saved"] = int(write_summary["qualified_inserted"])
                summary["duplicates"] = int(write_summary["duplicates"])
                summary["write_failed"] = int(write_summary["write_failed"])

                storage_labels = {
                    "inserted": "已新增",
                    "refreshed": "已更新",
                    "duplicate": "重复",
                    "failed": "写入失败",
                }
                for detail_index, item in zip(
                    case_detail_indexes, write_summary["items"], strict=False
                ):
                    detail = summary["details"][detail_index]
                    detail["storage_status"] = storage_labels.get(
                        item["storage_status"], "未知"
                    )
                    if item.get("reason"):
                        detail["reason"] = "；".join(
                            part
                            for part in (detail["reason"], item["reason"])
                            if part
                        )

                if summary["write_failed"]:
                    summary["errors"].append(
                        f"有 {summary['write_failed']} 条分析结果未写入数据库，请查看逐篇明细"
                    )
            except Exception as exc:
                summary["write_failed"] = len(summary["cases"])
                summary["errors"].append(
                    f"入库失败：{type(exc).__name__}: {exc}"
                )
                for detail_index in case_detail_indexes:
                    summary["details"][detail_index]["storage_status"] = "写入失败"
                logger.error("入库异常: %s", type(exc).__name__)
        return summary
    finally:
        summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if summary["errors"] and (
            summary["processed"] == 0 or summary["analyzed"] == 0
        ):
            summary["status"] = "failed"
        elif summary["errors"] or summary["write_failed"]:
            summary["status"] = "partial"
        else:
            summary["status"] = "success"
        try:
            summary["run_id"] = record_task_run(summary)
        except Exception as exc:
            logger.error("写入任务日志失败: %s", exc, exc_info=True)
        progress_message = {
            "success": "任务完成",
            "partial": "任务部分完成，请查看错误和逐篇明细",
            "failed": "任务失败，请查看错误详情",
        }[summary["status"]]
        _notify_progress(progress_callback, progress_message, 1.0)
        logger.info(
            "=== 流水线完成 === processed=%d analyzed=%d news_saved=%d "
            "qualified_saved=%d skipped=%d errors=%d",
            summary["processed"],
            summary["analyzed"],
            summary["news_saved"],
            summary["saved"],
            summary["skipped"],
            len(summary["errors"]),
        )
