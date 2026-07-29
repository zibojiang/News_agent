"""
app.py — 文旅行业情报 Agent Streamlit 前端

功能：研究主题采集、新闻池、量化案例库、人工审核、
主题配置、定时 worker 状态和任务日志。
"""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime
from time import perf_counter
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 本地开发时，Streamlit 每次重新运行都刷新 .env；Cloud 环境继续以
# Streamlit Secrets 注入的环境变量为准。
load_dotenv(
    override=os.getenv("DEPLOYMENT_MODE", "local").strip().lower() == "local"
)

from access_control import is_cloud_demo
from agent import (
    fetch_and_pre_score,
    get_ai_provider,
    get_gemini_model,
    get_openai_model,
    run_pipeline,
)
from database import (
    JSON_COLUMNS,
    format_json_list_for_display,
    get_scheduler_health,
    initialize_database,
    load_cases,
    load_last_search_state,
    load_task_runs,
    load_topics,
    save_last_search_state,
    update_case_review_status,
    update_topic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def _parse_quality_json(raw: Any) -> dict:
    """从 DB 的 quality_json 字段解析质量评分字典。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, str):
        try:
            import json as _json
            result = _json.loads(raw)
            if isinstance(result, list) and result:
                result = result[0] if isinstance(result[0], dict) else {}
            return result if isinstance(result, dict) else {}
        except (_json.JSONDecodeError, TypeError):
            return {}
    if isinstance(raw, dict):
        return raw
    return {}

def _quality_label_badge(label: str) -> str:
    """返回质量标签对应的 Emoji 标识。"""
    mapping = {
        "高质量": "🟢",
        "质量良好": "🔵",
        "优秀": "🟢",
        "良好": "🔵",
        "一般": "🟡",
        "偏低": "🟠",
        "较低": "🔴",
    }
    return mapping.get(label, "⚪") + " " + label

def _enrich_with_quality(df: pd.DataFrame) -> pd.DataFrame:
    """从 quality_json 提取 quality_score 和 quality_label 到 DataFrame。"""
    if df.empty or "quality_json" not in df.columns:
        return df
    df = df.copy()
    scores = []
    labels = []
    for _, row in df.iterrows():
        q = _parse_quality_json(row.get("quality_json"))
        scores.append(int(q.get("adjusted_score", q.get("total_score", 0))))
        labels.append(str(q.get("label", "")))
    df["quality_score"] = scores
    df["quality_label"] = labels
    return df

st.set_page_config(
    page_title="文旅新闻搜索",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #667085;
        --line: #e7eaf0;
        --brand: #2457d6;
        --brand-soft: #eef4ff;
    }
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .news-hero {
        padding: 2.2rem 2.4rem 1.8rem;
        border: 1px solid #dfe7fb;
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 12%, rgba(86, 126, 255, .18), transparent 28%),
            linear-gradient(135deg, #f8faff 0%, #ffffff 64%);
        box-shadow: 0 16px 42px rgba(35, 65, 125, .08);
        margin-bottom: 1rem;
    }
    .hero-kicker {
        color: var(--brand);
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .6rem;
    }
    .main-header {
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.12;
        font-weight: 800;
        color: var(--ink);
        letter-spacing: -.04em;
        margin: 0 0 .7rem;
    }
    .sub-header {color: var(--muted); font-size: 1rem; line-height: 1.7; margin: 0;}
    .section-title {font-size: 1.2rem; font-weight: 760; color: var(--ink); margin: .6rem 0 .15rem;}
    .section-note {color: var(--muted); font-size: .9rem; margin-bottom: .85rem;}
    .stDataFrame {border: 1px solid var(--line); border-radius: 12px; overflow: hidden;}
    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: .85rem 1rem;
        background: #fff;
    }
    [data-testid="stForm"] {
        border: 2px solid #5f7fe8;
        border-radius: 18px;
        padding: 1.25rem 1.25rem .7rem;
        background: #fff;
        box-shadow: 0 12px 30px rgba(36, 87, 214, .13);
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        min-height: 3.35rem;
        border: 2px solid #9db2ef;
        border-radius: 11px;
        background: #fbfcff;
        font-size: 1.03rem;
        padding-inline: 1rem;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input:focus {
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(36, 87, 214, .12);
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        min-height: 3.35rem;
        border-radius: 11px;
        font-weight: 750;
        font-size: 1rem;
    }
    .search-form-title {color: var(--ink); font-size: 1.28rem; font-weight: 800; margin-bottom: .15rem;}
    .search-form-note {color: var(--muted); font-size: .9rem; margin-bottom: .85rem;}
    .search-examples {color: #667085; font-size: .82rem; line-height: 1.6; margin: .45rem 0 .2rem;}
    .search-examples strong {color: #475467;}
    .article-found-status {color: #16803c; font-size: .92rem; font-weight: 760; margin-bottom: .35rem;}
    [data-testid="stSidebar"] {border-right: 1px solid var(--line);}
    div[data-testid="stExpander"] {border-color: var(--line); border-radius: 12px;}
    .footer-note {color: #98a2b3; font-size: .78rem; text-align: center; margin-top: 2rem;}
    @media (max-width: 700px) {
        .block-container {padding-top: 1rem;}
        .news-hero {padding: 1.45rem 1.2rem 1.1rem; border-radius: 16px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_database()


def _display_cases(df: pd.DataFrame) -> pd.DataFrame:
    """将 SQLite 原始字段转为适合表格显示的中文列。"""
    if df.empty:
        return df
    display_df = df.copy()
    for column in JSON_COLUMNS:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value, col=column: format_json_list_for_display(
                    value, numbered=(col in {"bullet_points", "evidence_quotes"})
                )
            )
    # 提取质量评分
    if "quality_json" in display_df.columns:
        quality_scores = []
        quality_labels = []
        for _, row in display_df.iterrows():
            q = _parse_quality_json(row.get("quality_json"))
            quality_scores.append(int(q.get("adjusted_score", q.get("total_score", 0))))
            quality_labels.append(_quality_label_badge(str(q.get("label", ""))))
        display_df["quality_score"] = quality_scores
        display_df["quality_label"] = quality_labels

    display_df["is_qualified"] = display_df["is_qualified"].map(
        {1: "已达标", 0: "未达标"}
    )
    return display_df.rename(
        columns={
            "id": "ID",
            "discovered_at": "发现时间",
            "published_at": "发布时间",
            "title": "新闻标题",
            "url": "原文链接",
            "source": "来源",
            "topic_id": "主题ID",
            "dimension": "维度",
            "category": "分类",
            "topic_name": "研究主题",
            "industry_keyword": "搜索词",
            "summary": "新闻摘要",
            "bullet_points": "量化案例",
            "evidence_quotes": "证据原文",
            "involved_companies": "涉及企业",
            "regions": "地区",
            "metric_tags": "指标标签",
            "relevance_score": "相关性",
            "is_qualified": "入库判定",
            "review_status": "审核状态",
            "quality_score": "报道质量",
            "quality_label": "质量评级",
        }
    )


def _table_config() -> dict[str, Any]:
    return {
        "原文链接": st.column_config.LinkColumn(
            "原文链接", display_text="🔗 查看原文"
        ),
        "相关性": st.column_config.ProgressColumn(
            "相关性", min_value=0, max_value=100, format="%d"
        ),
        "报道质量": st.column_config.ProgressColumn(
            "报道质量", min_value=0, max_value=100, format="%d"
        ),
        "新闻摘要": st.column_config.TextColumn("新闻摘要", width="large"),
        "量化案例": st.column_config.TextColumn("量化案例", width="large"),
        "证据原文": st.column_config.TextColumn("证据原文", width="large"),
    }


def _excel_bytes(df: pd.DataFrame) -> bytes:
    export_df = df.copy()
    for column in JSON_COLUMNS:
        if column in export_df.columns:
            export_df[column] = export_df[column].apply(
                lambda value, col=column: format_json_list_for_display(
                    value, numbered=(col in {"bullet_points", "evidence_quotes"})
                )
            )
    buffer = io.BytesIO()
    export_df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer.getvalue()


def _render_run_summary(summary: dict[str, Any]) -> None:
    """展示最近一次任务的分阶段统计和逐篇诊断信息。"""
    result_cols = st.columns(5)
    result_cols[0].metric("提取正文", summary.get("processed", 0))
    result_cols[1].metric("AI 成功", summary.get("analyzed", 0))
    result_cols[2].metric("新闻新增", summary.get("news_saved", 0))
    result_cols[3].metric("案例新增", summary.get("saved", 0))
    failure_count = int(summary.get("analysis_failed", 0)) + int(
        summary.get("write_failed", 0)
    )
    result_cols[4].metric("失败", failure_count)

    st.caption(
        f"未达案例门槛：{summary.get('unqualified', 0)} · "
        f"重复跳过：{summary.get('duplicates', 0)} · "
        f"数据库写入失败：{summary.get('write_failed', 0)}"
    )
    timing_parts = []
    if summary.get("search_seconds") is not None:
        timing_parts.append(f"新闻搜索 {summary['search_seconds']:.1f} 秒")
    if summary.get("analysis_seconds") is not None:
        timing_parts.append(f"AI 分析 {summary['analysis_seconds']:.1f} 秒")
    if timing_parts:
        st.caption("本次耗时：" + " · ".join(timing_parts))

    if summary.get("news_saved"):
        st.success(
            f"已新增 {summary['news_saved']} 条新闻到新闻池，"
            f"其中 {summary.get('saved', 0)} 条进入案例库。"
        )
    elif summary.get("processed") and not summary.get("analyzed"):
        st.error("已提取新闻正文，但 AI 分析全部失败，请查看错误详情。")
    elif summary.get("write_failed"):
        st.error("AI 分析已完成，但结果没有成功写入数据库。")
    elif summary.get("duplicates") and summary.get("duplicates") == summary.get("analyzed"):
        st.info("分析结果均已存在，本次没有重复写入新闻池。")
    elif summary.get("processed"):
        st.warning("任务已完成，但本次没有新增新闻，请查看逐篇明细。")

    errors = summary.get("errors", [])
    if errors:
        with st.expander(f"错误详情（{len(errors)}）", expanded=True):
            for error in errors:
                st.error(error)

    details = summary.get("details", [])
    if details:
        details_df = pd.DataFrame(details).rename(
            columns={
                "title": "新闻标题",
                "url": "原文链接",
                "score": "相关性",
                "analysis_status": "AI 状态",
                "qualification_status": "案例判定",
                "storage_status": "写入状态",
                "reason": "原因",
            }
        )
        st.markdown("#### 本次逐篇处理明细")
        st.dataframe(
            details_df,
            width="stretch",
            hide_index=True,
            column_config={
                "原文链接": st.column_config.LinkColumn(
                    "原文链接", display_text="🔗 查看原文"
                ),
                "相关性": st.column_config.ProgressColumn(
                    "相关性", min_value=0, max_value=100, format="%d"
                ),
                "原因": st.column_config.TextColumn("原因", width="large"),
            },
        )

def _render_score_breakdown(quality_details: dict) -> None:
    """在 expander 中展示文章质量评分的各维度明细。"""
    dim_scores = quality_details.get("dimension_scores", {})
    dim_reasons = quality_details.get("dimension_reasons", {})
    penalties = quality_details.get("penalties", [])
    total = quality_details.get("total_score", 0)
    adjusted = quality_details.get("adjusted_score", total)
    quality_label = quality_details.get("label", "")
    quality_warnings = quality_details.get("quality_warnings", [])

    dim_labels = {
        "source_credibility": "来源权威度",
        "content_density": "信息密度",
        "data_richness": "数据含量",
        "freshness": "时效性",
        "evidence_quality": "正文证据与可核验性",
        "completeness": "报道完整性",
        "transparency": "信源与方法透明度",
        "headline_body_consistency": "标题正文一致性",
        "balance": "客观与平衡性",
        "clarity": "清晰与连贯性",
    }
    dim_max_scores = {
        "source_credibility": 25,
        "content_density": 10,
        "data_richness": 10,
        "freshness": 5,
        "evidence_quality": 15,
        "completeness": 10,
        "transparency": 10,
        "headline_body_consistency": 5,
        "balance": 5,
        "clarity": 5,
    }
    items: list[str] = []
    for key, label in dim_labels.items():
        score = dim_scores.get(key, 0)
        reason = dim_reasons.get(key, "")
        if score > 0 or reason:
            items.append(f"{label}（{score}/{dim_max_scores[key]}）")
    total_deduction = sum(p.get("deduction", 0) for p in penalties)

    formula_parts = " + ".join(items) if items else "无维度数据"
    if total_deduction > 0:
        formula_parts += f" — 扣分（{total_deduction}分）"
    is_pre_score = quality_label == "预筛"
    if is_pre_score:
        heading = f"预筛基础分 {adjusted}/50"
    else:
        heading = f"综合质量 {adjusted}/100"
    if quality_label and not is_pre_score:
        heading += f" · {quality_label}"
    st.markdown(f"#### ⭐ {heading}")
    base_score = sum(
        int(dim_scores.get(key, 0) or 0)
        for key in (
            "source_credibility",
            "content_density",
            "data_richness",
            "freshness",
        )
    )
    if "evidence_quality" in dim_scores:
        body_score = int(
            quality_details.get(
                "body_quality_score",
                sum(
                    int(dim_scores.get(key, 0) or 0)
                    for key in (
                        "evidence_quality",
                        "completeness",
                        "transparency",
                        "headline_body_consistency",
                        "balance",
                        "clarity",
                    )
                ),
            )
            or 0
        )
        st.caption(f"基础维度 {base_score}/50 · AI 正文质量 {body_score}/50")
    else:
        st.caption(f"基础维度 {base_score}/50 · 等待 AI 正文质量评分")
    st.caption(f"计算方式：{formula_parts}")
    for warning in quality_warnings:
        st.warning(str(warning))

    for key, label in dim_labels.items():
        score = dim_scores.get(key, 0)
        reason = dim_reasons.get(key, "")
        if score > 0 or reason:
            st.markdown(f"**{label}：{score}/{dim_max_scores[key]} 分**")
            if reason:
                st.caption(reason)
    for penalty in penalties:
        st.caption(f"⚠️ 扣分 {penalty.get('deduction', 0)}：{penalty.get('reason', '')}")


def _quality_value(quality: Any, key: str, default: Any = None) -> Any:
    """同时兼容运行时 QualitySummary 和数据库恢复后的字典。"""
    if isinstance(quality, dict):
        return quality.get(key, default)
    return getattr(quality, key, default)


def _quality_state(quality: Any) -> dict[str, Any]:
    """将预筛评分精简为可持久化字典。"""
    penalties = []
    for penalty in _quality_value(quality, "penalties", []) or []:
        penalties.append(
            {
                "reason": _quality_value(penalty, "reason", ""),
                "deduction": _quality_value(penalty, "deduction", 0),
            }
        )
    return {
        "total_score": _quality_value(quality, "total_score", 0),
        "adjusted_score": _quality_value(quality, "adjusted_score", 0),
        "dimension_scores": dict(
            _quality_value(quality, "dimension_scores", {}) or {}
        ),
        "dimension_reasons": dict(
            _quality_value(quality, "dimension_reasons", {}) or {}
        ),
        "penalties": penalties,
        "label": _quality_value(quality, "label", ""),
        "score_cap": _quality_value(quality, "score_cap", 100),
        "quality_warnings": list(
            _quality_value(quality, "quality_warnings", []) or []
        ),
    }


def _persist_latest_search(
    keyword: str,
    articles: list[dict[str, Any]],
    results: dict[int, dict[str, Any]] | None = None,
    run_summary: dict[str, Any] | None = None,
) -> None:
    """保存恢复卡片所需的最小状态，不重复保存整篇正文。"""
    stored_articles = []
    for article in articles:
        stored_articles.append(
            {
                "title": str(article.get("title", "")),
                "url": str(article.get("url", "")),
                "source": str(article.get("source", "")),
                "published_at": str(article.get("published_at", "")),
                "language": str(article.get("language", "zh")),
                "content": str(article.get("content", ""))[:500],
                "quality_pre": _quality_state(article.get("quality_pre")),
            }
        )
    summary_state = {
        key: value
        for key, value in (run_summary or {}).items()
        if key != "cases"
    }
    state = {
        "keyword": keyword,
        "articles": stored_articles,
        "results": results or {},
        "run_summary": summary_state,
    }
    try:
        save_last_search_state(state)
    except Exception as exc:
        logger.warning("保存最近搜索状态失败: %s", type(exc).__name__)


def _restore_latest_search() -> None:
    """在新 Streamlit 会话中恢复最近一次搜索卡片。"""
    if st.session_state.get("latest_search_restored"):
        return
    st.session_state.latest_search_restored = True
    if st.session_state.get("fetched_articles"):
        return
    try:
        state = load_last_search_state()
    except Exception as exc:
        logger.warning("读取最近搜索状态失败: %s", type(exc).__name__)
        return
    if not state or not isinstance(state.get("articles"), list):
        return
    articles = state.get("articles", [])
    if not articles:
        return
    raw_results = state.get("results", {})
    results = {
        int(index): detail
        for index, detail in raw_results.items()
        if str(index).isdigit() and isinstance(detail, dict)
    } if isinstance(raw_results, dict) else {}
    keyword = str(state.get("keyword", ""))
    st.session_state.fetched_articles = articles
    st.session_state.fetched_keyword = keyword
    st.session_state.last_search_keyword = keyword
    if results:
        st.session_state.fetched_results = results
    if isinstance(state.get("run_summary"), dict) and state["run_summary"]:
        st.session_state.last_run_summary = state["run_summary"]


def _render_article_cards(
    articles: list[dict],
    keyword: str,
    results: dict[int, dict] | None = None,
) -> None:
    """渲染文章卡片列表。每张卡片含标题/来源/预评分/原文链接，分析完成后展示总评分和维度细分。

    Args:
        articles: fetch_and_pre_score 返回的文章列表
        keyword: 搜索关键词（仅作标注，不影响渲染）
        results: 已分析文章的结果字典，key 为文章序号(0-based)，value 为 run_pipeline 的 detail 字典
    """
    if not articles:
        return

    total_pre_score = sum(
        int(_quality_value(a.get("quality_pre"), "adjusted_score", 0) or 0)
        for a in articles
    )
    analyzed_count = len(results) if results else 0

    st.markdown(f"### 📰 已搜索到的文章（{len(articles)}篇）")
    if results:
        analyzed_score = sum(
            r.get("quality_score", 0) for r in (results or {}).values()
        )
        st.caption(
            f"已完成 AI 分析 {analyzed_count}/{len(articles)} 篇"
            f" | 预筛基础均分 {total_pre_score // len(articles)}/50"
            f" → 综合质量均分 {analyzed_score // max(analyzed_count, 1)}/100"
        )
    else:
        st.caption("新闻已搜索到，可以先浏览标题和原文；AI 分析会在后台继续。")

    for idx, article in enumerate(articles):
        pre = article.get("quality_pre")
        pre_score = int(
            _quality_value(
                pre,
                "adjusted_score",
                _quality_value(pre, "total_score", 0),
            ) or 0
        )
        pre_dims = dict(_quality_value(pre, "dimension_scores", {}) or {})

        result = (results or {}).get(idx)
        title = article.get("title", "无标题")
        source = article.get("source", "未知来源")
        url = article.get("url", "#")
        published = str(article.get("published_at", ""))[:16]
        language_label = " · 🌐 英文" if article.get("language") == "en" else ""
        source_score = int(pre_dims.get("source_credibility", 0) or 0)
        source_label = ""
        if source_score >= 22:
            source_label = " · 🛡️ 权威来源"
        elif source_score >= 15:
            source_label = " · 主流来源"

        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{title[:80]}**")
                st.caption(f"{source} · {published}{language_label}{source_label}")
                if url and url != "#":
                    st.markdown(f"[🔗 查看原文]({url})")
                summary_text = ""
                summary_label = ""
                if result and result.get("summary"):
                    summary_text = str(result["summary"])
                    summary_label = "AI 摘要"
                elif article.get("content"):
                    summary_text = " ".join(str(article["content"]).split())[:180]
                    if len(str(article["content"])) > 180:
                        summary_text += "…"
                    summary_label = "正文速览"
                if summary_text:
                    st.markdown(f"**📝 {summary_label}**")
                    st.write(summary_text)
            with cols[1]:
                if result:
                    relevance = result.get("score", 0) or 0
                    quality = result.get("quality_score", 0) or 0
                    label = result.get("quality_label", "")
                    st.metric("相关性", f"{relevance}分")
                    st.metric("综合质量", f"{quality}/100", label)
                    status = result.get("analysis_status", "?")
                    store = result.get("storage_status", "?")
                    st.caption(f"AI: {status} | 写入: {store}")
                else:
                    st.markdown(
                        '<div class="article-found-status">✓ 已搜索到</div>',
                        unsafe_allow_html=True,
                    )
                    st.metric("预筛基础分", f"{pre_score}/50")
                    st.caption("AI 分析中…")

            if result and result.get("analysis_status") == "成功":
                quality_score = result.get("quality_score", 0) or 0
                quality_label = result.get("quality_label", "")
                with st.expander(
                    f"⭐ 查看完整评分｜{quality_score}/100 · {quality_label}"
                ):
                    quality_json = result.get("quality_details")
                    if quality_json:
                        _render_score_breakdown(quality_json)
                    else:
                        st.caption("暂无评分明细")
                    reason = result.get("reason", "")
                    if reason:
                        st.caption(f"判定理由：{reason}")
            elif not result:
                with st.expander(f"⭐ 查看评分依据｜预筛 {pre_score}/50"):
                    if pre_dims:
                        _render_score_breakdown({
                            "total_score": _quality_value(pre, "total_score", pre_score),
                            "adjusted_score": pre_score,
                            "dimension_scores": pre_dims,
                            "dimension_reasons": dict(
                                _quality_value(pre, "dimension_reasons", {}) or {}
                            ),
                            "penalties": _quality_state(pre)["penalties"],
                            "label": _quality_value(pre, "label", ""),
                            "quality_warnings": _quality_value(
                                pre, "quality_warnings", []
                            ),
                        })
                    else:
                        st.caption("暂无评分明细")



def _set_active_view(view: str) -> None:
    st.session_state.active_view = view


def _render_news_pool() -> None:
    st.markdown('<p class="section-title">已分析新闻</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">按主题、审核状态或相关性快速缩小范围。</p>',
        unsafe_allow_html=True,
    )
    news_df = _enrich_with_quality(load_cases())
    filter_cols = st.columns([1, 1, 1, 1])
    topic_filter = filter_cols[0].selectbox(
        "主题筛选",
        ["全部"] + sorted(news_df["topic_name"].dropna().unique().tolist())
        if not news_df.empty
        else ["全部"],
        key="news_topic_filter",
    )
    status_filter = filter_cols[1].selectbox(
        "审核状态",
        ["全部", "待审核", "已确认", "已忽略", "低相关"],
        key="news_status_filter",
    )
    score_filter = filter_cols[2].number_input(
        "最低相关性", min_value=0, max_value=100, value=0, step=5
    )
    news_sort = filter_cols[3].selectbox(
        "排序方式",
        ["最新发布", "最早发布", "质量最高", "质量最低"],
        key="news_sort",
    )

    filtered_news = news_df.copy()
    if filtered_news.empty:
        st.info("还没有历史新闻。完成第一次搜索后，分析结果会自动出现在这里。")
        return

    if topic_filter != "全部":
        filtered_news = filtered_news[filtered_news["topic_name"] == topic_filter]
    if status_filter != "全部":
        filtered_news = filtered_news[filtered_news["review_status"] == status_filter]
    filtered_news = filtered_news[filtered_news["relevance_score"] >= score_filter]
    news_sort_map = {
        "最新发布": ("published_at", False),
        "最早发布": ("published_at", True),
        "质量最高": ("quality_score", False),
        "质量最低": ("quality_score", True),
    }
    sort_col, sort_asc = news_sort_map[news_sort]
    if sort_col in filtered_news.columns:
        filtered_news = filtered_news.sort_values(
            sort_col, ascending=sort_asc, na_position="last"
        )
    news_display = _display_cases(filtered_news)
    news_columns = [
        "发布时间", "新闻标题", "来源", "研究主题", "新闻摘要",
        "相关性", "入库判定", "审核状态", "原文链接",
    ]
    st.dataframe(
        news_display[[col for col in news_columns if col in news_display.columns]],
        width="stretch",
        hide_index=True,
        column_config=_table_config(),
    )
    st.caption(f"共 {len(filtered_news)} 条新闻")


def _render_case_library(allow_review: bool = False) -> None:
    st.markdown('<p class="section-title">量化案例</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">只展示达到入库门槛且包含可核验量化信息的新闻。</p>',
        unsafe_allow_html=True,
    )
    cases_df = _enrich_with_quality(load_cases(qualified_only=True))
    if cases_df.empty:
        st.info("暂无达标案例。可以降低搜索页的入库门槛，或尝试更具体的关键词。")
        return

    key_prefix = "review" if allow_review else "browse"
    case_filters = st.columns([2, 2, 2])
    case_topic = case_filters[0].selectbox(
        "案例主题",
        ["全部"] + sorted(cases_df["topic_name"].dropna().unique().tolist()),
        key=f"{key_prefix}_case_topic_filter",
    )
    case_status = case_filters[1].selectbox(
        "案例状态",
        ["全部", "待审核", "已确认", "已忽略"],
        key=f"{key_prefix}_case_status_filter",
    )
    company_query = case_filters[2].text_input(
        "企业关键词", key=f"{key_prefix}_company_query"
    )
    filtered_cases = cases_df.copy()
    if case_topic != "全部":
        filtered_cases = filtered_cases[filtered_cases["topic_name"] == case_topic]
    if case_status != "全部":
        filtered_cases = filtered_cases[filtered_cases["review_status"] == case_status]
    if company_query.strip():
        filtered_cases = filtered_cases[
            filtered_cases["involved_companies"].str.contains(
                company_query.strip(), case=False, na=False
            )
        ]

    case_display = _display_cases(filtered_cases)
    case_columns = [
        "发布时间", "新闻标题", "研究主题", "涉及企业", "地区",
        "量化案例", "证据原文", "相关性", "审核状态", "原文链接",
    ]
    st.dataframe(
        case_display[[col for col in case_columns if col in case_display.columns]],
        width="stretch",
        hide_index=True,
        column_config=_table_config(),
    )

    export_name = f"travel_intelligence_cases_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    if allow_review:
        case_options = {
            f"#{int(row.id)} · {str(row.title)[:55]}": int(row.id)
            for row in filtered_cases.itertuples()
        }
        if case_options:
            action_col, status_col, button_col = st.columns([5, 2, 1])
            selected_case_label = action_col.selectbox(
                "选择要审核的案例", list(case_options), key="review_case"
            )
            selected_status = status_col.selectbox(
                "更新为", ["已确认", "待审核", "已忽略"], key="review_status"
            )
            with button_col:
                st.write("")
                st.write("")
                if st.button("保存", width="stretch", key="save_review"):
                    update_case_review_status(
                        case_options[selected_case_label], selected_status
                    )
                    st.rerun()

    st.download_button(
        "📥 导出当前结果",
        data=_excel_bytes(filtered_cases),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _render_dashboard() -> None:
    all_news = load_cases()
    qualified_news = (
        all_news[all_news["is_qualified"] == 1] if not all_news.empty else all_news
    )
    pending_count = (
        int((qualified_news["review_status"] == "待审核").sum())
        if not qualified_news.empty
        else 0
    )
    covered_topics = int(all_news["topic_id"].nunique()) if not all_news.empty else 0
    metric_cols = st.columns(4)
    metric_cols[0].metric("新闻总量", len(all_news))
    metric_cols[1].metric("达标案例", len(qualified_news))
    metric_cols[2].metric("待审核", pending_count)
    metric_cols[3].metric("主题覆盖", f"{covered_topics} / 28")

    if all_news.empty:
        st.info("暂无数据。返回搜索首页完成第一次新闻搜索即可开始积累。")
        return

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### 主题覆盖")
        topic_counts = (
            all_news.groupby("topic_name").size().sort_values(ascending=False).head(12)
        )
        st.bar_chart(topic_counts, horizontal=True)
    with right:
        st.markdown("#### 最近发现")
        recent_cols = ["发现时间", "新闻标题", "研究主题", "相关性", "原文链接"]
        recent_df = _display_cases(_enrich_with_quality(all_news).head(8))
        st.dataframe(
            recent_df[[col for col in recent_cols if col in recent_df.columns]],
            width="stretch",
            hide_index=True,
            column_config=_table_config(),
        )


def _render_topic_management() -> None:
    st.markdown("#### 研究主题配置")
    st.caption(
        "控制周期采集使用的关键词、启用状态、入库门槛和篇数。"
        "修改只写入数据库，不会改动原始 Excel。"
    )
    editable_topics = load_topics(include_custom=False)
    editable_columns = [
        "topic_id", "dimension", "category", "topic_name", "report_frequency",
        "search_keywords", "enabled", "min_score", "max_articles",
        "collection_interval_hours",
    ]
    edited_topics = st.data_editor(
        editable_topics[editable_columns],
        width="stretch",
        hide_index=True,
        disabled=["topic_id", "dimension", "category", "topic_name", "report_frequency"],
        column_config={
            "topic_id": "主题 ID",
            "dimension": "维度",
            "category": "分类",
            "topic_name": "研究主题",
            "report_frequency": "报告频率",
            "search_keywords": st.column_config.TextColumn("新闻搜索词", width="large"),
            "enabled": st.column_config.CheckboxColumn("启用"),
            "min_score": st.column_config.NumberColumn(
                "入库分数", min_value=0, max_value=100
            ),
            "max_articles": st.column_config.NumberColumn(
                "单次篇数", min_value=1, max_value=30
            ),
            "collection_interval_hours": st.column_config.NumberColumn(
                "采集间隔/小时", min_value=1
            ),
        },
        key="topics_editor",
    )
    if st.button("💾 保存主题配置", type="primary"):
        for row in edited_topics.to_dict(orient="records"):
            update_topic(
                str(row["topic_id"]),
                {
                    "search_keywords": row["search_keywords"],
                    "enabled": row["enabled"],
                    "min_score": row["min_score"],
                    "max_articles": row["max_articles"],
                    "collection_interval_hours": row["collection_interval_hours"],
                },
            )
        st.success("主题配置已保存。")


def _render_task_center(cloud_demo: bool) -> None:
    st.markdown("#### 任务状态与运行日志")
    health_cols = st.columns(3)
    if cloud_demo:
        health_cols[0].metric("运行模式", "Cloud 展示")
        health_cols[1].metric("定时 Worker", "未启用")
        health_cols[2].metric("数据持久化", "临时")
        st.info("云端展示版支持手动搜索，不运行后台定时任务。")
    else:
        current_health = get_scheduler_health()
        health_cols[0].metric(
            "Worker 状态", "运行中" if current_health["running"] else "未运行"
        )
        health_cols[1].metric("最后心跳", current_health.get("last_heartbeat") or "-")
        health_cols[2].metric(
            "扫描间隔", f"{os.getenv('SCHEDULER_POLL_MINUTES', '30')} 分钟"
        )
        if not current_health["running"]:
            st.info("周期采集当前未运行；手动新闻搜索不受影响。")

    runs_df = load_task_runs(limit=100)
    if runs_df.empty:
        st.info("暂无任务运行记录。")
        return
    status_map = {"success": "成功", "partial": "部分成功", "failed": "失败"}
    trigger_map = {"manual": "手动", "scheduled": "定时"}
    display_runs = runs_df.copy()
    display_runs["status"] = display_runs["status"].map(status_map).fillna(
        display_runs["status"]
    )
    display_runs["trigger_type"] = display_runs["trigger_type"].map(
        trigger_map
    ).fillna(display_runs["trigger_type"])
    display_runs = display_runs.rename(
        columns={
            "id": "任务ID", "topic_id": "主题ID", "keyword": "搜索词",
            "trigger_type": "触发方式", "started_at": "开始时间",
            "finished_at": "结束时间", "status": "状态", "processed": "处理",
            "saved": "新增案例", "skipped": "跳过", "error_count": "错误数",
            "errors": "错误详情",
        }
    )
    st.dataframe(display_runs, width="stretch", hide_index=True)


def _render_search_page(cloud_demo: bool) -> None:
    _restore_latest_search()
    _, action_col = st.columns([12, 1])
    with action_col:
        st.button(
            "⚙️",
            key="open_management",
            help="打开管理台",
            on_click=_set_active_view,
            args=("management",),
        )
    st.markdown(
        """
        <div class="news-hero">
            <div class="hero-kicker">Travel Intelligence Search</div>
            <div class="main-header">找到值得读的文旅新闻</div>
            <p class="sub-header">输入一个公司、事件或行业关键词。系统会搜索真实新闻、分析相关性，<br>并把包含可核验数字的内容整理成案例。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("news_search_form"):
        st.markdown(
            """
            <div class="search-form-title">🔍 搜索新闻</div>
            <div class="search-form-note">输入企业、品牌、事件、地区或行业指标，查找相关新闻并自动分析。</div>
            """,
            unsafe_allow_html=True,
        )
        search_col, submit_col = st.columns([7, 2])
        with search_col:
            search_keyword = st.text_input(
                "搜索关键词",
                value=st.session_state.get("last_search_keyword", ""),
                placeholder="输入你想搜索的新闻关键词…",
                label_visibility="collapsed",
            )
        with submit_col:
            run_button = st.form_submit_button(
                "立即搜索", type="primary", width="stretch"
            )
        with st.expander("搜索设置"):
            option_cols = st.columns(2)
            max_articles = option_cols[0].slider(
                "文章数量", min_value=1, max_value=20, value=8, step=1,
                help="数量越多，分析时间和 AI 用量越高。",
            )
            min_score = option_cols[1].slider(
                "案例入库门槛", min_value=0, max_value=100, value=70, step=5,
                help="低于门槛的结果仍会保留在新闻池。",
            )
            english_keyword = st.text_input(
                "英文关键词（可选）",
                placeholder="例如 Fosun AI；留空时自动使用上方关键词",
                help="用于搜索 Reuters、BBC、Bloomberg 等英文新闻源。",
            )
        st.markdown(
            """
            <div class="search-examples">
                <strong>搜索示例：</strong>半导体产业研报 · AI 行业应用与商业化 ROI · AI 市场份额与竞争格局
            </div>
            """,
            unsafe_allow_html=True,
        )

    if cloud_demo:
        st.caption("当前是云端展示模式；应用重启后，临时数据可能重置。")

    if run_button:
        keyword = search_keyword.strip()
        if not keyword:
            st.error("请输入一个新闻关键词。")
        else:
            st.session_state.last_search_keyword = keyword
            st.session_state.pop("last_run_summary", None)
            st.session_state.pop("fetched_results", None)
            st.session_state.pop("fetched_articles", None)
            st.session_state.pop("pending_analysis", None)
            live_articles: list[dict[str, Any]] = []
            live_result_area = st.empty()

            def show_found_article(
                article: dict[str, Any], found: int, total: int
            ) -> None:
                live_articles.append(article)
                live_result_area.empty()
                with live_result_area.container():
                    _render_article_cards(live_articles, keyword)

            search_started = perf_counter()
            with st.spinner(f"正在搜索「{keyword}」相关的新闻…"):
                try:
                    articles = fetch_and_pre_score(
                        industry_keyword=keyword,
                        max_articles=int(max_articles),
                        article_callback=show_found_article,
                        english_keyword=english_keyword.strip() or None,
                    )
                except Exception as exc:
                    logger.error("搜索失败: %s", exc, exc_info=True)
                    st.error(f"搜索失败：{exc}")
                    articles = []
            search_seconds = perf_counter() - search_started

            if articles:
                st.session_state.fetched_articles = articles
                st.session_state.fetched_keyword = keyword
                _persist_latest_search(keyword, articles)
                st.session_state.pending_analysis = {
                    "keyword": keyword,
                    "min_score": int(min_score),
                    "max_articles": int(max_articles),
                    "search_seconds": search_seconds,
                }
                st.rerun()
            else:
                st.warning("没有找到可分析的文章。请尝试更具体或更常见的关键词。")

    if not run_button and st.session_state.get("fetched_articles"):
        articles = st.session_state.fetched_articles
        _render_article_cards(
            articles,
            st.session_state.get("fetched_keyword", ""),
            st.session_state.get("fetched_results"),
        )

        pending_analysis = st.session_state.pop("pending_analysis", None)
        if pending_analysis:
            st.markdown("### AI 正在并行分析")
            progress_bar = st.progress(0.0, text="已展示新闻，正在准备 AI 分析…")

            def update_progress(message: str, value: float) -> None:
                progress_bar.progress(value, text=message)

            try:
                analysis_started = perf_counter()
                run_summary = run_pipeline(
                    industry_keyword=pending_analysis["keyword"],
                    min_score=pending_analysis["min_score"],
                    max_articles=pending_analysis["max_articles"],
                    topic=None,
                    trigger_type="manual",
                    progress_callback=update_progress,
                    pre_fetched_articles=articles,
                )
                run_summary["search_seconds"] = pending_analysis["search_seconds"]
                run_summary["analysis_seconds"] = perf_counter() - analysis_started
                st.session_state.last_run_summary = run_summary
                result_map = {
                    index: detail
                    for index, detail in enumerate(run_summary.get("details", []))
                }
                st.session_state.fetched_results = result_map
                _persist_latest_search(
                    pending_analysis["keyword"],
                    articles,
                    result_map,
                    run_summary,
                )
                progress_bar.progress(1.0, text="分析完成")
                st.rerun()
            except ValueError as exc:
                st.error(f"配置错误：{exc}")
            except Exception as exc:
                logger.error("手动任务失败: %s", exc, exc_info=True)
                st.error(f"分析失败：{exc}")

    last_run_summary = st.session_state.get("last_run_summary")
    if last_run_summary:
        _render_run_summary(last_run_summary)

    st.divider()
    news_tab, cases_tab = st.tabs(["📰 新闻池", "📚 案例库"])
    with news_tab:
        _render_news_pool()
    with cases_tab:
        _render_case_library()


def _render_management_page(cloud_demo: bool) -> None:
    back_col, title_col = st.columns([1, 11])
    with back_col:
        st.button(
            "←",
            key="back_to_search",
            help="返回新闻搜索",
            on_click=_set_active_view,
            args=("search",),
        )
    with title_col:
        st.markdown('<p class="main-header" style="font-size:2rem">管理台</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sub-header">查看数据积累、审核案例、配置研究主题和排查任务运行。</p>',
            unsafe_allow_html=True,
        )
    st.info(
        "当前管理台为开放模式，无需管理员密码。任何能访问网页的人都可以审核案例和修改主题配置。",
        icon="ℹ️",
    )

    overview_tab, review_tab, topics_tab, tasks_tab = st.tabs(
        ["📊 数据概览", "✅ 案例审核", "🎯 主题配置", "⏱️ 任务日志"]
    )
    with overview_tab:
        _render_dashboard()
    with review_tab:
        _render_case_library(allow_review=True)
    with topics_tab:
        _render_topic_management()
    with tasks_tab:
        _render_task_center(cloud_demo)


cloud_demo = is_cloud_demo()
if "active_view" not in st.session_state:
    st.session_state.active_view = "search"

with st.sidebar:
    st.markdown("### 📰 文旅新闻搜索")
    st.caption("搜索真实新闻，沉淀量化案例。")
    st.button(
        "🔎 搜索首页",
        width="stretch",
        type="primary" if st.session_state.active_view == "search" else "secondary",
        on_click=_set_active_view,
        args=("search",),
    )
    st.button(
        "⚙️ 管理台",
        width="stretch",
        type="primary" if st.session_state.active_view == "management" else "secondary",
        on_click=_set_active_view,
        args=("management",),
    )
    st.divider()
    if cloud_demo:
        st.caption("Cloud 展示模式 · 数据可能随重启重置")
    else:
        scheduler_health = get_scheduler_health()
        st.caption(
            "定时采集：" + ("运行中" if scheduler_health["running"] else "未运行")
        )

if st.session_state.active_view == "management":
    _render_management_page(cloud_demo)
else:
    _render_search_page(cloud_demo)

active_provider = get_ai_provider()
active_model = (
    get_gemini_model() if active_provider == "gemini" else get_openai_model()
)
st.markdown(
    f'<p class="footer-note">AI：{active_provider} / {active_model} · '
    f'运行模式：{"cloud_demo" if cloud_demo else "local"}</p>',
    unsafe_allow_html=True,
)
