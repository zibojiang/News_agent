"""
agent.py — Industry News Agent 核心逻辑

使用 Google 官方 google-genai SDK 调用 Gemini 大模型，
通过 Structured Outputs（response_schema）强制返回规整 JSON，
从新闻正文中提炼带量化数据的商业案例。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from database import append_cases_batch, record_task_run
from scraper import fetch_and_extract_batch

# 加载 .env 环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# Gemini 模型名称（可通过环境变量覆盖）
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# API 调用重试配置
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # 秒，指数退避基数

# 定时任务默认相关性分数门槛
DEFAULT_MIN_SCORE = 70


class NewsCaseSchema(BaseModel):
    """
    新闻商业案例结构化输出 Schema。

    Gemini 将严格按照此 Pydantic 模型返回 JSON，
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

5. **证据可追溯**：evidence_quotes 只能摘录输入正文中真实存在的短句，
   不得补全、改写或编造数字。

6. 若正文无法提取有效量化案例，bullet_points 和 evidence_quotes 可为空数组，
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


def _build_user_prompt(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
) -> str:
    """组装发送给 Gemini 的用户侧 Prompt。"""
    topic = topic or {}
    return f"""【目标行业】{industry_keyword}

【研究主题】{topic.get('topic_id', 'CUSTOM')} {topic.get('topic_name', '自定义行业研究')}
【主题分类】{topic.get('dimension', '自定义')} / {topic.get('category', '自定义')}

【新闻标题】{article_title}

【原文链接】{article_url}

【新闻正文】
{article_text}

请输出 title、url、summary、bullet_points、evidence_quotes、
involved_companies、regions、metric_tags、relevance_score 字段。
其中 title 和 url 请直接使用上方提供的值。"""


def _keep_verifiable_evidence(quotes: list[str], article_text: str) -> list[str]:
    """只保留去除空白后能在输入正文中完整匹配的证据短句。"""
    normalized_article = re.sub(r"\s+", "", article_text)
    return [
        quote.strip()
        for quote in quotes
        if quote.strip() and re.sub(r"\s+", "", quote.strip()) in normalized_article
    ]


def analyze_article_with_gemini(
    article_title: str,
    article_url: str,
    article_text: str,
    industry_keyword: str,
    topic: dict[str, Any] | None = None,
    model: str | None = None,
) -> NewsCaseSchema | None:
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
        NewsCaseSchema 实例；失败时返回 None
    """
    if not article_text.strip():
        logger.warning("正文为空，跳过分析: %s", article_title)
        return None

    client = _get_gemini_client()
    model_name = model or DEFAULT_MODEL

    topic = topic or {}
    system_prompt = ANALYST_SYSTEM_PROMPT.format(
        industry_keyword=industry_keyword,
        topic_id=topic.get("topic_id", "CUSTOM"),
        topic_name=topic.get("topic_name", "自定义行业研究"),
    )
    user_prompt = _build_user_prompt(
        article_title, article_url, article_text, industry_keyword, topic
    )

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
                logger.warning("Gemini 返回空响应")
                continue

            # 解析并校验 JSON
            parsed = NewsCaseSchema.model_validate_json(raw_text)

            # 确保 title/url 与输入一致（防止模型篡改）
            parsed.title = article_title
            parsed.url = article_url
            parsed.evidence_quotes = _keep_verifiable_evidence(
                parsed.evidence_quotes, article_text
            )

            logger.info(
                "分析完成: score=%d, bullets=%d — %s",
                parsed.relevance_score,
                len(parsed.bullet_points),
                article_title[:40],
            )
            return parsed

        except ValidationError as exc:
            logger.error("Gemini 返回 JSON 校验失败: %s", exc)
            # 尝试手动解析兜底
            try:
                if raw_text:
                    data = json.loads(raw_text)
                    parsed = NewsCaseSchema.model_validate(data)
                    parsed.title = article_title
                    parsed.url = article_url
                    parsed.evidence_quotes = _keep_verifiable_evidence(
                        parsed.evidence_quotes, article_text
                    )
                    return parsed
            except Exception:
                pass

        except Exception as exc:
            error_msg = str(exc).lower()
            # 识别限流 / 配额错误，使用指数退避重试
            if any(kw in error_msg for kw in ("429", "rate", "quota", "resource_exhausted")):
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini API 限流/配额不足，%ds 后重试 (attempt %d): %s",
                    delay,
                    attempt,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.error("Gemini API 调用失败: %s", exc, exc_info=True)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY)

    logger.error("Gemini 分析最终失败: %s", article_title)
    return None


def run_pipeline(
    industry_keyword: str,
    min_score: int = DEFAULT_MIN_SCORE,
    max_articles: int = 8,
    topic: dict[str, Any] | None = None,
    trigger_type: str = "manual",
) -> dict[str, Any]:
    """
    执行完整的「抓取 → 提炼 → 入库」工作流。

    Args:
        industry_keyword: 行业关键词
        min_score: 入库最低相关性分数
        max_articles: 单次最多处理文章数

    Returns:
        运行摘要字典，包含 processed/saved/errors 等统计信息
    """
    topic = topic or {}
    summary: dict[str, Any] = {
        "keyword": industry_keyword,
        "topic_id": topic.get("topic_id", "CUSTOM"),
        "topic_name": topic.get("topic_name", "自定义行业研究"),
        "trigger_type": trigger_type,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed": 0,
        "saved": 0,
        "skipped": 0,
        "errors": [],
        "cases": [],
    }

    logger.info(
        "=== 开始 Industry News Agent 流水线 === topic=%s keyword=%s",
        summary["topic_id"],
        industry_keyword,
    )

    try:
        try:
            articles = fetch_and_extract_batch(
                industry_keyword, max_articles=max_articles
            )
        except Exception as exc:
            logger.error("抓取阶段异常: %s", exc, exc_info=True)
            summary["errors"].append(f"抓取失败: {exc}")
            return summary

        if not articles:
            summary["errors"].append("未获取到有效文章")
            logger.warning("流水线结束：无可用文章")
            return summary

        for article in articles:
            summary["processed"] += 1
            try:
                result = analyze_article_with_gemini(
                    article_title=article["title"],
                    article_url=article["url"],
                    article_text=article["content"],
                    industry_keyword=industry_keyword,
                    topic=topic,
                )
            except ValueError as exc:
                summary["errors"].append(str(exc))
                logger.error("致命错误，终止流水线: %s", exc)
                break
            except Exception as exc:
                summary["errors"].append(f"分析异常 [{article['title']}]: {exc}")
                logger.error("单篇分析异常: %s", exc, exc_info=True)
                continue

            if result is None:
                summary["skipped"] += 1
                continue

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
                "topic_id": summary["topic_id"],
                "topic_name": summary["topic_name"],
                "dimension": topic.get("dimension", "自定义"),
                "category": topic.get("category", "自定义"),
            }
            summary["cases"].append(case_dict)

            if (
                result.relevance_score < min_score
                or not result.bullet_points
                or not result.evidence_quotes
            ):
                summary["skipped"] += 1
                logger.info(
                    "分数/量化案例/证据未达门槛，仅保留新闻池: %s",
                    result.title[:40],
                )

        try:
            summary["saved"] = append_cases_batch(
                summary["cases"], min_score=min_score
            )
        except Exception as exc:
            summary["errors"].append(f"入库失败: {exc}")
            logger.error("入库异常: %s", exc, exc_info=True)
        return summary
    finally:
        summary["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if summary["errors"] and summary["processed"] == 0:
            summary["status"] = "failed"
        elif summary["errors"]:
            summary["status"] = "partial"
        else:
            summary["status"] = "success"
        try:
            summary["run_id"] = record_task_run(summary)
        except Exception as exc:
            logger.error("写入任务日志失败: %s", exc, exc_info=True)
        logger.info(
            "=== 流水线完成 === processed=%d saved=%d skipped=%d errors=%d",
            summary["processed"],
            summary["saved"],
            summary["skipped"],
            len(summary["errors"]),
        )
