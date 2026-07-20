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
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from database import append_cases_batch_with_summary, record_task_run
from scraper import fetch_and_extract_batch

# 加载 .env 环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# AI 提供方与模型名称（均可通过环境变量覆盖）
DEFAULT_AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").strip().lower()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip()

# API 调用重试配置
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # 秒，指数退避基数

# 定时任务默认相关性分数门槛
DEFAULT_MIN_SCORE = 70


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


def _get_openai_client() -> OpenAI:
    """初始化 OpenAI 客户端，不在日志中暴露密钥。"""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "未配置有效的 OPENAI_API_KEY，请在 .env 或 Streamlit Secrets 中设置"
        )
    return OpenAI(api_key=api_key)


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
    model_name = model or DEFAULT_OPENAI_MODEL
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
    model_name = model or DEFAULT_GEMINI_MODEL
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
    provider_name = (provider or DEFAULT_AI_PROVIDER).strip().lower()
    analyzers = {
        "openai": analyze_article_with_openai,
        "gemini": analyze_article_with_gemini,
    }
    if provider_name not in analyzers:
        raise ValueError(
            f"不支持的 AI_PROVIDER：{provider_name}，可选值为 openai 或 gemini"
        )
    return analyzers[provider_name](
        article_title=article_title,
        article_url=article_url,
        article_text=article_text,
        industry_keyword=industry_keyword,
        topic=topic,
        model=model,
    )


def run_pipeline(
    industry_keyword: str,
    min_score: int = DEFAULT_MIN_SCORE,
    max_articles: int = 8,
    topic: dict[str, Any] | None = None,
    trigger_type: str = "manual",
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """
    执行完整的「抓取 → 提炼 → 入库」工作流。

    Args:
        industry_keyword: 行业关键词
        min_score: 入库最低相关性分数
        max_articles: 单次最多处理文章数

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
        "analyzed": 0,
        "analysis_failed": 0,
        "news_saved": 0,
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
        _notify_progress(progress_callback, "正在搜索新闻并提取正文…", 0.05)
        try:
            articles = fetch_and_extract_batch(
                industry_keyword, max_articles=max_articles
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

        total_articles = len(articles)
        _notify_progress(
            progress_callback,
            f"已提取 {total_articles} 篇正文，开始 AI 分析…",
            0.15,
        )

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
            _notify_progress(
                progress_callback,
                f"AI 分析第 {index}/{total_articles} 篇：{detail['title'][:35]}",
                0.15 + 0.7 * ((index - 1) / total_articles),
            )

            try:
                result = analyze_article(
                    article_title=article["title"],
                    article_url=article["url"],
                    article_text=article["content"],
                    industry_keyword=industry_keyword,
                    topic=topic,
                )
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

            summary["analyzed"] += 1
            detail["score"] = result.relevance_score
            detail["analysis_status"] = "成功"
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
                summary["saved"] = int(write_summary["qualified_inserted"])
                summary["duplicates"] = int(write_summary["duplicates"])
                summary["write_failed"] = int(write_summary["write_failed"])

                storage_labels = {
                    "inserted": "已新增",
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
