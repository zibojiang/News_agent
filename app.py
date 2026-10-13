"""
app.py — 文旅行业情报 Agent Streamlit 前端

功能：新闻搜索、当前会话的搜索历史、新闻池、量化案例库和人工审核。
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
from i18n import (
    get_language,
    init_language,
    intent_text,
    label_formatter,
    localize,
    localize_dataframe,
    t,
    toggle_language,
)

# 本地开发时，Streamlit 每次重新运行都刷新 .env；Cloud 环境继续以
# Streamlit Secrets 注入的环境变量为准。
load_dotenv(
    override=os.getenv("DEPLOYMENT_MODE", "local").strip().lower() == "local"
)

from access_control import is_cloud_demo
from agent import (
    BODY_ANALYSIS_RELEVANCE_THRESHOLD,
    SEARCH_RELEVANCE_DIMENSION_MAXIMA,
    SEARCH_RELEVANCE_DISPLAY_THRESHOLD,
    SEARCH_RELEVANCE_RULE_VERSION,
    SearchIntentSchema,
    analyze_search_intent,
    calculate_multi_intent_article_limit,
    calculate_recommendation_score,
    fallback_search_intent,
    fetch_and_pre_score,
    get_ai_provider,
    get_intent_model,
    merge_search_interpretations,
    run_pipeline,
    score_sources_with_ai,
)
from database import (
    JSON_COLUMNS,
    format_json_list_for_display,
    initialize_database,
    load_cases,
    load_last_search_state,
    save_last_search_state,
    update_case_review_status,
)
from quality_scorer import NEWS_QUALITY_RULE_VERSION
from result_table import (
    HIGH_QUALITY_DEFAULT_THRESHOLD,
    build_high_quality_results,
    csv_bytes,
    selected_export_rows,
)
from search_task import get_background_search, start_background_search
from scraper import article_matches_language
from session_workspace import session_workspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _workspace():
    return session_workspace(st.session_state)


def _db_path() -> str:
    return _workspace().db_path


def _owned_task(task_id: str) -> dict[str, Any] | None:
    return get_background_search(task_id, owner_id=_workspace().owner_id)


def _selected_news_language() -> str:
    """UI language is the sole policy; ignore obsolete saved dropdown choices."""
    return "en" if get_language() == "en" else "both"


def _visible_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or get_language() == "zh":
        return frame.copy()
    return frame.loc[
        [article_matches_language(row, "en") for row in frame.to_dict("records")]
    ].copy()


def _news_language_label(language: str) -> str:
    return {
        "en": t("英文原文", "English articles"),
        "zh": t("中文原文", "Chinese articles"),
        "both": t("中英双语", "Chinese and English articles"),
    }[language]


def _analyze_intent_for_session(query: str) -> SearchIntentSchema:
    """Reuse successful plans only within this user's current session."""
    key = (query.strip(), get_ai_provider(), get_intent_model(), os.getenv("OPENAI_BASE_URL", ""))
    cache = st.session_state.setdefault("intent_plan_cache", {})
    cached = cache.get(key)
    if cached and perf_counter() - cached["created_at"] < 900:
        return SearchIntentSchema.model_validate(cached["intent"])
    started = perf_counter()
    try:
        result = analyze_search_intent(query)
    finally:
        logger.info("搜索意图请求耗时 %.2fs", perf_counter() - started)
    cache[key] = {"created_at": perf_counter(), "intent": result.model_dump()}
    while len(cache) > 12:
        cache.pop(next(iter(cache)))
    return result


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
    return mapping.get(label, "⚪") + " " + localize(label)


BASE_QUALITY_DIMENSIONS = {
    "source_credibility": ("来源权威度", 25),
    "content_density": ("信息密度", 10),
    "data_richness": ("数据含量", 10),
    "freshness": ("时效性", 5),
}
BODY_QUALITY_DIMENSIONS = {
    "evidence_quality": ("正文证据与可核验性", 15),
    "completeness": ("报道完整性", 10),
    "transparency": ("信源与方法透明度", 10),
    "headline_body_consistency": ("标题正文一致性", 5),
    "balance": ("客观与平衡性", 5),
    "clarity": ("清晰与连贯性", 5),
}
SEARCH_RELEVANCE_DIMENSIONS = {
    "core_topic_match": ("核心主题匹配", 40),
    "information_need_match": ("信息需求匹配", 30),
    "semantic_coverage": ("语义内容覆盖", 20),
    "directness": ("直接相关程度", 10),
}


def _is_current_quality_details(
    quality_details: Any,
    *,
    require_body: bool,
) -> bool:
    """判断评分详情是否完整符合当前规则，避免混用旧版分数。"""
    if not isinstance(quality_details, dict):
        return False
    if quality_details.get("rule_version") != NEWS_QUALITY_RULE_VERSION:
        return False
    if quality_details.get("source_score_method") != "ai":
        return False
    dimension_scores = quality_details.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        return False
    required = set(BASE_QUALITY_DIMENSIONS)
    if require_body:
        required.update(BODY_QUALITY_DIMENSIONS)
    if not required.issubset(dimension_scores):
        return False
    maxima = {**BASE_QUALITY_DIMENSIONS, **BODY_QUALITY_DIMENSIONS}
    for key in required:
        value = dimension_scores.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not 0 <= value <= maxima[key][1]:
            return False
    return True


def _bounded_dimension_score(value: Any, maximum: int) -> int:
    """将展示分数限制在当前维度的合法范围内。"""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(maximum, score))

def _enrich_with_quality(df: pd.DataFrame) -> pd.DataFrame:
    """只提取当前规则的质量分，旧版评分不参与排序或展示。"""
    if df.empty or "quality_json" not in df.columns:
        return df
    df = df.copy()
    scores = []
    labels = []
    for _, row in df.iterrows():
        q = _parse_quality_json(row.get("quality_json"))
        if _is_current_quality_details(q, require_body=True):
            scores.append(
                _bounded_dimension_score(q.get("adjusted_score"), 100)
            )
            labels.append(str(q.get("label", "")))
        else:
            scores.append(None)
            labels.append("")
    df["quality_score"] = scores
    df["quality_label"] = labels
    return df

init_language()

st.set_page_config(
    page_title=t("文旅新闻搜索", "Travel News Intelligence"),
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
    .block-container {max-width: 1180px; padding-top: 5.5rem; padding-bottom: 3rem;}
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
    .st-key-news_search_form {
        border: 2px solid #5f7fe8;
        border-radius: 18px;
        padding: 1.25rem 1.25rem .7rem;
        background: #fff;
        box-shadow: 0 12px 30px rgba(36, 87, 214, .13);
    }
    .st-key-news_search_form [data-testid="stTextInput"] input {
        min-height: 3.35rem;
        border: 2px solid #9db2ef;
        border-radius: 11px;
        background: #fbfcff;
        font-size: 1.03rem;
        padding-inline: 1rem;
    }
    .st-key-news_search_form [data-testid="stTextInput"] input:focus {
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(36, 87, 214, .12);
    }
    .st-key-news_search_form [data-testid="stButton"] button {
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
        .block-container {padding-top: 5rem;}
        .news-hero {padding: 1.45rem 1.2rem 1.1rem; border-radius: 16px;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_database(_db_path())


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
            if _is_current_quality_details(q, require_body=True):
                quality_scores.append(
                    _bounded_dimension_score(q.get("adjusted_score"), 100)
                )
                quality_labels.append(
                    _quality_label_badge(str(q.get("label", "")))
                )
            else:
                quality_scores.append(None)
                quality_labels.append("")
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
    configs = {
        "原文链接": st.column_config.LinkColumn(
            t("原文链接", "Source link"), display_text=t("🔗 查看原文", "🔗 Read article")
        ),
        "相关性": st.column_config.ProgressColumn(
            t("相关性", "Relevance"), min_value=0, max_value=100, format="%d"
        ),
        "报道质量": st.column_config.ProgressColumn(
            t("报道质量", "Article quality"), min_value=0, max_value=100, format="%d"
        ),
        "新闻摘要": st.column_config.TextColumn(t("新闻摘要", "Summary"), width="large"),
        "量化案例": st.column_config.TextColumn(t("量化案例", "Quantified cases"), width="large"),
        "证据原文": st.column_config.TextColumn(t("证据原文", "Evidence quotes"), width="large"),
    }
    return {**configs, **{localize(key): value for key, value in configs.items()}}


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
    localize_dataframe(export_df).to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    return buffer.getvalue()


def _render_high_quality_results_table(summary: dict[str, Any]) -> None:
    """展示本次高质量新闻，允许编辑后下载，不覆盖原始评分数据。"""
    cases = summary.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return

    st.markdown(t("### 📊 本次高质量新闻数据表", "### 📊 Quality news from this search"))
    st.markdown(
        t(
            '<p class="section-note">筛选达到质量门槛的新闻；可直接修改业务字段，下载文件会使用修改后的内容。</p>',
            '<p class="section-note">Filter articles by quality, edit their details, and download the edited results.</p>',
        ),
        unsafe_allow_html=True,
    )
    run_key = str(summary.get("started_at") or "latest").replace(" ", "_")
    min_quality = st.slider(
        t("最低新闻质量", "Minimum article quality"),
        min_value=0,
        max_value=100,
        value=HIGH_QUALITY_DEFAULT_THRESHOLD,
        step=5,
        key=f"high_quality_threshold_{run_key}",
        help=t("默认展示 75 分及以上，即“质量良好”和“高质量”的新闻。", "Shows articles scoring 75 or higher by default, including good and high-quality articles."),
    )
    table_df = _visible_news_frame(build_high_quality_results(summary, min_quality=min_quality))
    if table_df.empty:
        st.info(t("本次没有新闻达到 {score} 分的质量门槛。", "No articles in this search reached the quality threshold of {score}.").format(score=min_quality))
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric(t("高质量新闻", "Quality articles"), len(table_df))
    metric_cols[1].metric(t("平均质量", "Average quality"), f"{table_df['新闻质量'].mean():.0f}/100")
    metric_cols[2].metric(t("平均相关性", "Average relevance"), f"{table_df['搜索相关性'].mean():.0f}/100")

    editable_columns = [
        "导出",
        "新闻标题",
        "新闻摘要",
        "研究主题",
        "涉及企业",
        "地区",
        "量化案例",
        "证据原文",
        "指标标签",
        "编辑备注",
    ]
    disabled_columns = [
        column for column in table_df.columns if column not in editable_columns
    ]
    editor_key = f"high_quality_editor_{run_key}_{min_quality}_{get_language()}"
    editor_data_key = f"{editor_key}_data"
    editor_saved_key = f"{editor_key}_saved"
    editor_language_key = f"{editor_key}_language"
    if (
        editor_data_key not in st.session_state
        or editor_key not in st.session_state
        or st.session_state.get(editor_language_key) != get_language()
    ):
        st.session_state[editor_data_key] = st.session_state.get(
            editor_saved_key, table_df
        ).copy()
        st.session_state[editor_language_key] = get_language()
        st.session_state.pop(editor_key, None)
    editor_display_df = st.session_state[editor_data_key].copy()
    if "质量评级" in editor_display_df:
        editor_display_df["质量评级"] = editor_display_df["质量评级"].map(localize)
    edited_df = st.data_editor(
        editor_display_df,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        row_height=72,
        disabled=disabled_columns,
        key=editor_key,
        column_config={
            **{column: st.column_config.Column(localize(column)) for column in table_df.columns},
            "导出": st.column_config.CheckboxColumn(
                t("导出", "Export"), help=t("取消勾选后，该行不会进入下载文件。", "Uncheck to exclude this row from the download.")
            ),
            "原文链接": st.column_config.LinkColumn(
                t("原文链接", "Source link"), display_text=t("🔗 查看原文", "🔗 Read article")
            ),
            "搜索相关性": st.column_config.ProgressColumn(
                t("搜索相关性", "Search relevance"), min_value=0, max_value=100, format="%d"
            ),
            "新闻质量": st.column_config.ProgressColumn(
                t("新闻质量", "Article quality"), min_value=0, max_value=100, format="%d"
            ),
            "推荐分": st.column_config.ProgressColumn(
                t("推荐分", "Recommendation score"), min_value=0, max_value=100, format="%d"
            ),
            "新闻标题": st.column_config.TextColumn(t("新闻标题", "Article title"), width="large"),
            "新闻摘要": st.column_config.TextColumn(t("新闻摘要", "Summary"), width="large"),
            "量化案例": st.column_config.TextColumn(t("量化案例", "Quantified cases"), width="large"),
            "证据原文": st.column_config.TextColumn(t("证据原文", "Evidence quotes"), width="large"),
            "编辑备注": st.column_config.TextColumn(t("编辑备注", "Editor notes"), width="medium"),
        },
    )
    if "质量评级" in edited_df:
        edited_df["质量评级"] = table_df["质量评级"]
    st.session_state[editor_saved_key] = edited_df.copy()
    export_df = selected_export_rows(edited_df)
    st.caption(
        t(
            "可编辑：标题、摘要、主题、企业、地区、量化案例、证据、标签和备注。分数、来源、时间及原文链接保持只读；编辑只用于本次下载，不覆盖数据库。",
            "Edit titles, summaries, topics, companies, regions, cases, evidence, tags, and notes. Scores, sources, dates, and links are read-only. Edits apply to this download without changing the database.",
        )
    )
    if export_df.empty:
        st.warning(t("请至少勾选一条新闻后再下载。", "Select at least one article to download."))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_cols = st.columns(2)
    download_cols[0].download_button(
        t("📥 下载 Excel", "📥 Download Excel"),
        data=_excel_bytes(export_df),
        file_name=f"high_quality_news_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    download_cols[1].download_button(
        t("📄 下载 CSV", "📄 Download CSV"),
        data=csv_bytes(localize_dataframe(export_df)),
        file_name=f"high_quality_news_{timestamp}.csv",
        mime="text/csv",
        width="stretch",
    )


def _render_run_summary(summary: dict[str, Any]) -> None:
    """展示最近一次任务的分阶段统计和逐篇诊断信息。"""
    result_cols = st.columns(5)
    result_cols[0].metric(t("提取正文", "Articles extracted"), summary.get("processed", 0))
    result_cols[1].metric(t("AI 成功", "AI analyses completed"), summary.get("analyzed", 0))
    result_cols[2].metric(t("新闻新增", "New articles"), summary.get("news_saved", 0))
    result_cols[3].metric(t("案例新增", "New cases"), summary.get("saved", 0))
    failure_count = int(summary.get("analysis_failed", 0)) + int(
        summary.get("write_failed", 0)
    )
    result_cols[4].metric(t("失败", "Failures"), failure_count)

    st.caption(
        t(
            "未达案例门槛：{unqualified} · 旧记录更新：{refreshed} · 重复跳过：{duplicates} · 数据库写入失败：{write_failed}",
            "Below case threshold: {unqualified} · Records updated: {refreshed} · Duplicates skipped: {duplicates} · Database write failures: {write_failed}",
        ).format(**{key: summary.get(key, 0) for key in ("unqualified", "refreshed", "duplicates", "write_failed")})
    )
    timing_parts = []
    if summary.get("search_seconds") is not None:
        timing_parts.append(t("新闻搜索 {seconds:.1f} 秒", "News search: {seconds:.1f}s").format(seconds=summary["search_seconds"]))
    if summary.get("analysis_seconds") is not None:
        timing_parts.append(t("AI 分析 {seconds:.1f} 秒", "AI analysis: {seconds:.1f}s").format(seconds=summary["analysis_seconds"]))
    if timing_parts:
        st.caption(t("本次耗时：", "Elapsed time: ") + " · ".join(timing_parts))

    if summary.get("news_saved"):
        st.success(
            t("已新增 {news} 条新闻到新闻池，其中 {cases} 条进入案例库。", "Added {news} articles to the news pool; {cases} qualified for the case library.").format(news=summary["news_saved"], cases=summary.get("saved", 0))
        )
    elif summary.get("refreshed"):
        st.success(
            t("已用当前评分规则更新 {count} 条已有新闻。", "Updated {count} existing articles using the current scoring rules.").format(count=summary["refreshed"])
        )
    elif summary.get("processed") and not summary.get("analyzed"):
        st.error(t("已提取新闻正文，但 AI 分析全部失败，请查看错误详情。", "Article text was extracted, but all AI analyses failed. See the error details."))
    elif summary.get("write_failed"):
        st.error(t("AI 分析已完成，但结果没有成功写入数据库。", "AI analysis completed, but the results could not be saved to the database."))
    elif summary.get("duplicates") and summary.get("duplicates") == summary.get("analyzed"):
        st.info(t("分析结果均已存在，本次没有重复写入新闻池。", "All results already exist. No duplicate articles were saved."))
    elif summary.get("processed"):
        st.warning(t("任务已完成，但本次没有新增新闻，请查看逐篇明细。", "The task completed without adding articles. See the article details."))

    errors = summary.get("errors", [])
    if errors:
        with st.expander(t("错误详情（{count}）", "Error details ({count})").format(count=len(errors)), expanded=True):
            for error in errors:
                st.error(localize(error))

    details = [
        detail for detail in summary.get("details", [])
        if article_matches_language(detail, _selected_news_language())
    ]
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
        st.markdown(t("#### 本次逐篇处理明细", "#### Article processing details"))
        st.dataframe(
            localize_dataframe(details_df),
            width="stretch",
            hide_index=True,
            column_config={
                localize("原文链接"): st.column_config.LinkColumn(
                    t("原文链接", "Source link"), display_text=t("🔗 查看原文", "🔗 Read article")
                ),
                localize("相关性"): st.column_config.ProgressColumn(
                    t("相关性", "Relevance"), min_value=0, max_value=100, format="%d"
                ),
                localize("原因"): st.column_config.TextColumn(t("原因", "Reason"), width="large"),
            },
        )
    _render_high_quality_results_table(summary)

def _render_score_breakdown(quality_details: dict) -> None:
    """在 expander 中展示文章质量评分的各维度明细。"""
    quality_label = quality_details.get("label", "")
    is_pre_score = quality_label == "预筛"
    if not _is_current_quality_details(
        quality_details,
        require_body=not is_pre_score,
    ):
        return

    dim_scores = quality_details.get("dimension_scores", {})
    dim_reasons = quality_details.get("dimension_reasons", {})
    penalties = quality_details.get("penalties", [])
    total = quality_details.get("total_score", 0)
    adjusted = quality_details.get("adjusted_score", total)
    quality_warnings = quality_details.get("quality_warnings", [])

    dimensions = dict(BASE_QUALITY_DIMENSIONS)
    dimensions["source_credibility"] = ("来源权威度（AI 评分）", 25)
    if not is_pre_score:
        dimensions.update(BODY_QUALITY_DIMENSIONS)
    bounded_scores = {
        key: _bounded_dimension_score(dim_scores.get(key), maximum)
        for key, (_, maximum) in dimensions.items()
    }
    items: list[str] = []
    for key, (label, maximum) in dimensions.items():
        score = bounded_scores[key]
        reason = dim_reasons.get(key, "")
        if score > 0 or reason:
            items.append(t("{label}（{score}/{maximum}）", "{label} ({score}/{maximum})").format(label=localize(label), score=score, maximum=maximum))
    total_deduction = sum(p.get("deduction", 0) for p in penalties)

    formula_parts = " + ".join(items) if items else t("无维度数据", "No dimension scores")
    if total_deduction > 0:
        formula_parts += t(" — 扣分（{points}分）", " − deductions ({points} points)").format(points=total_deduction)
    base_score = sum(
        bounded_scores[key] for key in BASE_QUALITY_DIMENSIONS
    )
    if is_pre_score:
        pre_score = max(0, min(50, int(adjusted or 0)))
        st.markdown(t("#### 基础分 {score}/50", "#### Base score {score}/50").format(score=pre_score))
        st.caption(t("基础评分已完成；AI 分析完成后会再生成 50 分正文质量分。", "Base scoring is complete. AI analysis will add up to 50 points for article content quality."))
    else:
        body_score = sum(
            bounded_scores[key] for key in BODY_QUALITY_DIMENSIONS
        )
        adjusted = max(0, min(100, int(adjusted or 0)))
        heading = t("综合质量 {score}/100", "Overall quality {score}/100").format(score=adjusted)
        if quality_label:
            heading += f" · {localize(quality_label)}"
        st.markdown(f"####  {heading}")
        equation = t("基础分 {base}/50 + AI 正文质量 {body}/50", "Base score {base}/50 + AI content quality {body}/50").format(base=base_score, body=body_score)
        if total_deduction > 0:
            equation += t(" - 扣分 {points}", " − deductions {points}").format(points=total_deduction)
        score_cap = max(
            0,
            min(100, int(quality_details.get("score_cap", 100) or 100)),
        )
        if score_cap < 100:
            equation += t("，规则封顶 {cap}", ", rule cap {cap}").format(cap=score_cap)
        st.markdown(
            t("**{equation} = 综合质量 {score}/100**", "**{equation} = overall quality {score}/100**").format(equation=equation, score=adjusted)
        )
    st.caption(t("计算方式：{formula}", "Calculation: {formula}").format(formula=formula_parts))
    for warning in quality_warnings:
        st.warning(localize(str(warning)))

    for key, (label, maximum) in dimensions.items():
        score = bounded_scores[key]
        reason = dim_reasons.get(key, "")
        if score > 0 or reason:
            st.markdown(t("**{label}：{score}/{maximum} 分**", "**{label}: {score}/{maximum} points**").format(label=localize(label), score=score, maximum=maximum))
            if reason:
                st.caption(localize(reason))
    for penalty in penalties:
        st.caption(t("⚠️ 扣分 {points}：{reason}", "⚠️ Deduction {points}: {reason}").format(points=penalty.get("deduction", 0), reason=localize(penalty.get("reason", ""))))


def _render_search_relevance(
    article: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> None:
    """展示独立于新闻质量分的搜索相关性初筛结果。"""
    raw_score = article.get("search_relevance_score")
    if raw_score is None and result:
        raw_score = result.get("score")
    if raw_score is None:
        return
    relevance = _bounded_dimension_score(raw_score, 100)
    reason = str(
        article.get("search_relevance_reason")
        or (result or {}).get("relevance_reason", "")
    )
    st.markdown(t("#### 🎯 搜索相关性 {score}/100", "#### 🎯 Search relevance {score}/100").format(score=relevance))
    dimensions = article.get("search_relevance_dimensions")
    if not isinstance(dimensions, dict) and result:
        quality_details = result.get("quality_details")
        if isinstance(quality_details, dict):
            dimensions = quality_details.get("search_relevance_dimensions")
    if isinstance(dimensions, dict):
        dimension_parts = []
        for key, (label, maximum) in SEARCH_RELEVANCE_DIMENSIONS.items():
            score = _bounded_dimension_score(dimensions.get(key), maximum)
            dimension_parts.append(f"{localize(label)} {score}/{maximum}")
            st.markdown(t("**{label}：{score}/{maximum} 分**", "**{label}: {score}/{maximum} points**").format(label=localize(label), score=score, maximum=maximum))
        st.caption(t("计算方式：", "Calculation: ") + " + ".join(dimension_parts))
    if reason:
        st.caption(localize(reason))
    st.caption(t("搜索相关性用于筛选和排序，不计入新闻质量基础分 50 分。", "Search relevance controls filtering and ranking. It is separate from the 50-point quality base score."))


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
        "rule_version": _quality_value(quality, "rule_version", ""),
        "source_score_method": _quality_value(
            quality, "source_score_method", ""
        ),
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
    search_intent: dict[str, Any] | None = None,
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
                "source_ai_scored": bool(article.get("source_ai_scored", False)),
                "source_ai_error": str(article.get("source_ai_error", "")),
                "search_relevance_score": article.get("search_relevance_score"),
                "search_relevance_dimensions": dict(
                    article.get("search_relevance_dimensions", {}) or {}
                ),
                "search_relevance_reason": str(
                    article.get("search_relevance_reason", "")
                ),
                "search_relevance_scored": bool(
                    article.get("search_relevance_scored", False)
                ),
                "search_relevance_rule_version": str(
                    article.get("search_relevance_rule_version", "")
                ),
            }
        )
    summary_state = dict(run_summary or {})
    state = {
        "keyword": keyword,
        "articles": stored_articles,
        "results": results or {},
        "run_summary": summary_state,
        "search_intent": search_intent or {},
    }
    try:
        save_last_search_state(state, db_path=_db_path())
    except Exception as exc:
        logger.warning("保存最近搜索状态失败: %s", type(exc).__name__)


def _completed_search_uses_current_scoring(
    articles: Any,
    results: Any,
) -> bool:
    """仅允许完整的当前规则搜索状态跨刷新恢复。"""
    if not isinstance(articles, list) or not articles:
        return False
    if not isinstance(results, dict) or not results:
        return False
    if any(not isinstance(article, dict) for article in articles):
        return False
    if any(
        article.get("search_relevance_scored") is not True
        or isinstance(article.get("search_relevance_score"), bool)
        or not isinstance(article.get("search_relevance_score"), (int, float))
        or not 0 <= article["search_relevance_score"] <= 100
        for article in articles
    ):
        return False
    for article in articles:
        if (
            article.get("search_relevance_rule_version")
            != SEARCH_RELEVANCE_RULE_VERSION
        ):
            return False
        dimensions = article.get("search_relevance_dimensions")
        if not isinstance(dimensions, dict):
            return False
        if set(SEARCH_RELEVANCE_DIMENSION_MAXIMA) - set(dimensions):
            return False
        for key, maximum in SEARCH_RELEVANCE_DIMENSION_MAXIMA.items():
            value = dimensions.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not 0 <= value <= maximum:
                return False
        if round(
            sum(dimensions[key] for key in SEARCH_RELEVANCE_DIMENSION_MAXIMA)
        ) != round(
            article["search_relevance_score"]
        ):
            return False
    if any(
        not _is_current_quality_details(
            _quality_state(article.get("quality_pre")),
            require_body=False,
        )
        for article in articles
    ):
        return False
    for detail in results.values():
        if not isinstance(detail, dict):
            return False
        if detail.get("analysis_status") == "成功" and not (
            _is_current_quality_details(
                detail.get("quality_details"),
                require_body=True,
            )
        ):
            return False
    return True


def _clear_search_session() -> None:
    """移除当前会话中的旧搜索结果，但保留搜索框关键词。"""
    for key in (
        "fetched_articles",
        "fetched_results",
        "fetched_keyword",
        "fetched_search_intent",
        "last_run_summary",
        "search_intent_review",
        "confirmed_search_request",
    ):
        st.session_state.pop(key, None)


def _restore_latest_search() -> None:
    """在新 Streamlit 会话中恢复最近一次搜索卡片。"""
    if st.session_state.get("fetched_articles"):
        if st.session_state.get("pending_analysis"):
            return
        if _completed_search_uses_current_scoring(
            st.session_state.get("fetched_articles"),
            st.session_state.get("fetched_results"),
        ):
            return
        _clear_search_session()
        try:
            save_last_search_state({}, db_path=_db_path())
        except Exception as exc:
            logger.warning("清理旧搜索状态失败: %s", type(exc).__name__)
        return
    if st.session_state.get("latest_search_restored"):
        return
    st.session_state.latest_search_restored = True
    try:
        state = load_last_search_state(db_path=_db_path())
    except Exception as exc:
        logger.warning("读取最近搜索状态失败: %s", type(exc).__name__)
        return
    if not state or not isinstance(state.get("articles"), list):
        return
    articles = state.get("articles", [])
    if not articles:
        return
    raw_results = state.get("results", {})
    if not _completed_search_uses_current_scoring(articles, raw_results):
        try:
            save_last_search_state({}, db_path=_db_path())
        except Exception as exc:
            logger.warning("清理旧搜索状态失败: %s", type(exc).__name__)
        return
    results = {
        int(index): detail
        for index, detail in raw_results.items()
        if str(index).isdigit() and isinstance(detail, dict)
    } if isinstance(raw_results, dict) else {}
    keyword = str(state.get("keyword", ""))
    st.session_state.fetched_articles = articles
    st.session_state.fetched_keyword = keyword
    st.session_state.last_search_keyword = keyword
    if isinstance(state.get("search_intent"), dict):
        st.session_state.fetched_search_intent = state["search_intent"]
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

    visible_indexes = [
        index for index, article in enumerate(articles)
        if article_matches_language(article, _selected_news_language())
    ]
    hidden_count = len(articles) - len(visible_indexes)
    if hidden_count:
        st.caption(t(
            "已隐藏 {count} 篇非英文文章；中文模式可查看中英文结果。",
            "Hidden {count} non-English articles. Switch to Chinese to view both languages.",
        ).format(count=hidden_count))
    if not visible_indexes:
        st.info(t(
            "本次结果没有英文文章。可切换到中文查看全部结果，或重新搜索英文新闻。",
            "No English articles in these results. Switch to Chinese to view all results, or start a new search for English news.",
        ))
        return
    # Filter only the rendered view; retain original results and score pairing.
    if results is not None:
        results = {
            new_index: results[old_index]
            for new_index, old_index in enumerate(visible_indexes)
            if old_index in results
        }
    articles = [articles[index] for index in visible_indexes]

    current_pre_scores = []
    for article in articles:
        pre_state = _quality_state(article.get("quality_pre"))
        if _is_current_quality_details(pre_state, require_body=False):
            pre_dimensions = pre_state.get("dimension_scores", {})
            current_pre_scores.append(
                sum(
                    _bounded_dimension_score(
                        pre_dimensions.get(key),
                        maximum,
                    )
                    for key, (_, maximum) in BASE_QUALITY_DIMENSIONS.items()
                )
            )
    analyzed_count = sum(
        1
        for result in (results or {}).values()
        if result.get("analysis_status") == "成功"
    )
    relevance_skipped_count = sum(
        1
        for result in (results or {}).values()
        if result.get("analysis_status") == "跳过"
    )

    st.markdown(t("### 📰 已搜索到的文章（{count}篇）", "### 📰 Articles found ({count})").format(count=len(articles)))
    intent = st.session_state.get("fetched_search_intent", {})
    if isinstance(intent, dict) and intent.get("intent_summary"):
        if intent.get("used_fallback"):
            st.warning(
                t(
                    "AI 搜索意图理解未成功，本次已降级为原始中文关键词搜索。英文检索词只会在 AI 成功生成或你手动填写时使用。",
                    "AI search intent analysis failed. This search uses the original Chinese keywords. English queries are used only when generated successfully or entered manually.",
                )
            )
            fallback_reason = str(intent.get("fallback_reason", "")).strip()
            if fallback_reason:
                st.caption(t("降级原因：{reason}", "Fallback reason: {reason}").format(reason=localize(fallback_reason)))
        summary_text = intent_text(intent, "intent_summary")
        if summary_text:
            st.info(t("🧠 AI 理解的搜索目标：{intent}", "🧠 AI interpretation of your search: {intent}").format(intent=summary_text))
        with st.expander(t("查看本次中英文检索策略", "View Chinese and English search strategy")):
            selected_intents = intent.get("selected_intents", []) or []
            target_topics = intent.get("target_topics", []) or []
            chinese_queries = intent.get("chinese_queries", []) or []
            english_queries = intent.get("english_queries", []) or []
            relevance_criteria = intent.get("relevance_criteria", []) or []
            if selected_intents:
                st.markdown(t("**已选择的细分意图：**", "**Selected search directions:**"))
                for selected_intent in selected_intents:
                    if isinstance(selected_intent, dict):
                        st.markdown(
                            f"- {intent_text(selected_intent, 'label')}: "
                            f"{intent_text(selected_intent, 'intent_summary')}"
                        )
            if target_topics:
                st.markdown(t("**目标主题：**", "**Target topics:** ") + " · ".join(map(str, target_topics)))
            if chinese_queries:
                st.markdown(t("**中文检索词：**", "**Chinese queries:** ") + " · ".join(map(str, chinese_queries)))
            if english_queries:
                st.markdown(t("**英文检索词：**", "**English queries:** ") + " · ".join(map(str, english_queries)))
            if relevance_criteria:
                st.markdown(t("**相关性判断标准：**", "**Relevance criteria:**"))
                for criterion in relevance_criteria:
                    st.markdown(f"- {criterion}")
    if results:
        current_final_scores = [
            _bounded_dimension_score(result.get("quality_score"), 100)
            for result in results.values()
            if _is_current_quality_details(
                result.get("quality_details"),
                require_body=True,
            )
        ]
        summary_parts = [t("已完成 AI 分析 {completed}/{total} 篇", "AI analysis completed: {completed}/{total}").format(completed=analyzed_count, total=len(articles))]
        if relevance_skipped_count:
            summary_parts.append(
                t("相关性不足跳过 {count} 篇", "Skipped for low relevance: {count}").format(count=relevance_skipped_count)
            )
        if current_pre_scores:
            summary_parts.append(
                t("基础均分 {score}/50", "Average base score: {score}/50").format(score=sum(current_pre_scores) // len(current_pre_scores))
            )
        if current_final_scores:
            summary_parts.append(
                t("综合质量均分 {score}/100", "Average overall quality: {score}/100").format(score=sum(current_final_scores) // len(current_final_scores))
            )
        st.caption(" · ".join(summary_parts))
    else:
        if len(current_pre_scores) == len(articles):
            st.caption(
                t(
                    "搜索相关性与来源权威度 AI 初筛已完成，基础分已生成；正文质量分析会继续进行。",
                    "AI relevance and source screening is complete. Base scores are ready; content quality analysis will continue.",
                )
            )
        else:
            st.caption(
                t(
                    "新闻已搜索到，可以先浏览标题和原文；AI 将先评估搜索相关性与来源权威度。",
                    "Articles are ready to browse. AI will first assess search relevance and source credibility.",
                )
            )

    for idx, article in enumerate(articles):
        pre = article.get("quality_pre")
        pre_state = _quality_state(pre)
        has_current_pre_score = _is_current_quality_details(
            pre_state,
            require_body=False,
        )
        is_current_pre_version = (
            pre_state.get("rule_version") == NEWS_QUALITY_RULE_VERSION
        )
        pre_score = _bounded_dimension_score(
            pre_state.get("adjusted_score"),
            50,
        )
        pre_dims = dict(_quality_value(pre, "dimension_scores", {}) or {})

        result = (results or {}).get(idx)
        quality_json = result.get("quality_details") if result else None
        has_current_final_score = _is_current_quality_details(
            quality_json,
            require_body=True,
        )
        title = article.get("title", t("无标题", "Untitled"))
        source = article.get("source", t("未知来源", "Unknown source"))
        if source == "未知来源":
            source = t("未知来源", "Unknown source")
        url = article.get("url", "#")
        published = str(article.get("published_at", ""))[:16]
        language_label = t(" · 🌐 英文", " · 🌐 English") if article.get("language") == "en" else ""
        source_score = int(pre_dims.get("source_credibility", 0) or 0)
        relevance = article.get("search_relevance_score")
        if relevance is None and result:
            relevance = result.get("score")
        source_label = ""
        if has_current_pre_score:
            if source_score >= 22:
                source_label = t(" · 🛡️ AI 评估：权威来源", " · 🛡️ AI assessment: authoritative source")
            elif source_score >= 15:
                source_label = t(" · AI 评估：主流来源", " · AI assessment: mainstream source")

        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{title[:80]}**")
                st.caption(f"{source} · {published}{language_label}{source_label}")
                if url and url != "#":
                    st.markdown(t("[🔗 查看原文]({url})", "[🔗 Read article]({url})").format(url=url))
                summary_text = ""
                summary_label = ""
                if result and result.get("summary"):
                    summary_text = str(result["summary"])
                    summary_label = t("AI 摘要", "AI summary")
                elif article.get("content"):
                    summary_text = " ".join(str(article["content"]).split())[:180]
                    if len(str(article["content"])) > 180:
                        summary_text += "…"
                    summary_label = t("正文速览", "Article preview")
                if summary_text:
                    st.markdown(f"**📝 {summary_label}**")
                    st.write(summary_text)
            with cols[1]:
                if result:
                    relevance_score = _bounded_dimension_score(relevance, 100)
                    st.metric(t("搜索相关性", "Search relevance"), f"{relevance_score}/100")
                    if has_current_final_score:
                        quality = _bounded_dimension_score(
                            result.get("quality_score"),
                            100,
                        )
                        label = result.get("quality_label", "")
                        st.metric(t("综合质量", "Overall quality"), f"{quality}/100", localize(label))
                        recommendation = _bounded_dimension_score(
                            result.get("recommendation_score"),
                            100,
                        )
                        st.metric(t("推荐分", "Recommendation score"), f"{recommendation}/100")
                    elif has_current_pre_score:
                        st.metric(t("基础分", "Base score"), f"{pre_score}/50")
                        st.caption(t("正文 AI 分析未完成，保留基础评分", "Content analysis is incomplete; showing the base score."))
                    status = result.get("analysis_status", "?")
                    store = result.get("storage_status", "?")
                    st.caption(t("AI: {status} | 写入: {store}", "AI: {status} | Saved: {store}").format(status=localize(status), store=localize(store)))
                else:
                    st.markdown(
                        t('<div class="article-found-status">✓ 已搜索到</div>', '<div class="article-found-status">✓ Article found</div>'),
                        unsafe_allow_html=True,
                    )
                    if has_current_pre_score:
                        st.metric(t("基础分", "Base score"), f"{pre_score}/50")
                        st.caption(t("AI 正文质量分析中…", "AI is assessing content quality…"))
                        if relevance is not None:
                            st.metric(
                                t("搜索相关性", "Search relevance"),
                                f"{_bounded_dimension_score(relevance, 100)}/100",
                            )
                    elif is_current_pre_version:
                        st.metric(t("基础分", "Base score"), t("计算中", "Calculating"))
                        st.caption(t("AI 正在评估搜索相关性与来源权威度…", "AI is assessing search relevance and source credibility…"))

            if (
                result
                and result.get("analysis_status") == "成功"
                and has_current_final_score
            ):
                quality_score = _bounded_dimension_score(
                    result.get("quality_score"),
                    100,
                )
                quality_label = result.get("quality_label", "")
                expander_label = (
                    t(" 查看完整评分｜{score}/100 · {label}", "View full score | {score}/100 · {label}").format(score=quality_score, label=localize(quality_label))
                )
                with st.expander(expander_label):
                    _render_search_relevance(article, result)
                    recommendation = _bounded_dimension_score(
                        result.get("recommendation_score"),
                        100,
                    )
                    st.markdown(t("**推荐分 {score}/100**", "**Recommendation score {score}/100**").format(score=recommendation))
                    st.caption(t("推荐分 = 搜索相关性 × 70% + 综合质量 × 30%", "Recommendation score = search relevance × 70% + overall quality × 30%"))
                    if isinstance(quality_json, dict):
                        _render_score_breakdown(quality_json)
                    else:
                        st.caption(t("暂无评分明细", "Score breakdown unavailable"))
                    reason = result.get("reason", "")
                    if reason:
                        st.caption(t("判定理由：{reason}", "Decision reason: {reason}").format(reason=localize(reason)))
            elif has_current_pre_score:
                with st.expander(t(" 查看基础评分｜{score}/50", "View base score | {score}/50").format(score=pre_score)):
                    _render_search_relevance(article, result)
                    if pre_dims:
                        _render_score_breakdown(pre_state)
                    else:
                        st.caption(t("暂无评分明细", "Score breakdown unavailable"))



def _set_active_view(view: str) -> None:
    st.session_state.active_view = view


def _active_search_task_id() -> str:
    """Task identity is server-side only; shared URLs never restore private data."""
    if "search_task" in st.query_params:
        del st.query_params["search_task"]
    return str(st.session_state.get("active_search_task_id") or "").strip()


def _clear_active_search_task() -> None:
    st.session_state.pop("active_search_task_id", None)
    if "search_task" in st.query_params:
        del st.query_params["search_task"]


def _apply_completed_search_task(snapshot: dict[str, Any]) -> None:
    """将后台结果接回当前会话，并按现有格式持久化。"""
    articles = list(snapshot.get("articles") or [])
    if not articles:
        st.session_state.search_task_notice = str(
            snapshot.get("message") or "没有找到可分析的文章。"
        )
        diagnostics = dict(snapshot.get("search_diagnostics") or {})
        if diagnostics and not diagnostics.get("extracted"):
            st.session_state.search_task_diagnostics = diagnostics
        _clear_active_search_task()
        return
    results = {
        int(index): detail
        for index, detail in dict(snapshot.get("results") or {}).items()
        if isinstance(detail, dict)
    }
    keyword = str(snapshot.get("keyword") or "")
    search_intent = dict(snapshot.get("search_intent") or {})
    run_summary = dict(snapshot.get("run_summary") or {})
    st.session_state.fetched_articles = articles
    st.session_state.fetched_keyword = keyword
    st.session_state.fetched_search_intent = search_intent
    st.session_state.fetched_results = results
    st.session_state.last_run_summary = run_summary
    _persist_latest_search(
        keyword,
        articles,
        results,
        run_summary,
        search_intent=search_intent,
    )
    _clear_active_search_task()


@st.fragment(run_every=2)
def _render_active_search_task(task_id: str) -> None:
    """自动轮询后台搜索；切换我的记录不会销毁该 fragment 之外的任务。"""
    snapshot = _owned_task(task_id)
    if snapshot is None:
        st.error(t("后台搜索任务已不存在，可能是应用进程重新启动。", "This background task is no longer available. The application may have restarted."))
        if st.button(t("清除失效任务", "Clear unavailable task"), key=f"clear_missing_task_{task_id}"):
            _clear_active_search_task()
            st.rerun()
        return
    status = str(snapshot.get("status") or "running")
    if status == "completed":
        _apply_completed_search_task(snapshot)
        st.rerun()
    if status == "failed":
        st.error(t("后台搜索失败：{error}", "Background search failed: {error}").format(error=localize(snapshot.get('error') or snapshot.get('message'))))
        if st.button(t("关闭失败任务", "Dismiss failed task"), key=f"clear_failed_task_{task_id}"):
            _clear_active_search_task()
            st.rerun()
        return

    stage_labels = {
        "queued": t("任务排队", "Queued"),
        "search": t("搜索与正文提取", "Search and content extraction"),
        "source": t("相关性与来源初筛", "Relevance and source screening"),
        "body": t("正文质量分析", "Content quality analysis"),
    }
    stage = str(snapshot.get("stage") or "queued")
    st.markdown(t("### ⏳ 后台任务 · {stage}", "### ⏳ Background task · {stage}").format(stage=stage_labels.get(stage, stage)))
    st.progress(
        float(snapshot.get("progress", 0.0) or 0.0),
        text=localize(str(snapshot.get("message") or "后台任务运行中…")),
    )
    st.caption(t("你可以进入“我的记录”；搜索和 AI 分析会继续运行。", "You can open My history while search and AI analysis continue."))
    articles = list(snapshot.get("articles") or [])
    if articles:
        _render_article_cards(
            articles,
            str(snapshot.get("keyword") or ""),
            snapshot.get("results"),
        )


def _render_background_task_banner() -> None:
    """在我的记录中提示仍在运行或已完成的用户搜索。"""
    task_id = _active_search_task_id()
    if not task_id:
        return
    snapshot = _owned_task(task_id)
    if snapshot is None:
        st.warning(t("先前的搜索任务已失效，请返回搜索页重新发起。", "The previous task is no longer available. Return to search to start again."))
        return
    status = str(snapshot.get("status") or "running")
    if status == "running":
        st.info(
            t("后台搜索仍在运行：{message} 你可以继续查看我的记录。", "Search is still running: {message} You can keep browsing My history.").format(message=localize(snapshot.get('message', '')))
        )
        st.progress(float(snapshot.get("progress", 0.0) or 0.0))
    elif status == "completed":
        st.success(t("后台搜索和 AI 分析已经完成，返回搜索页即可查看结果。", "Search and AI analysis are complete. Return to search to view the results."))
    else:
        st.error(t("后台搜索失败：{error}", "Background search failed: {error}").format(error=localize(snapshot.get('error', ''))))
    st.button(
        t("返回搜索查看任务", "Return to search"),
        key="management_open_search_task",
        on_click=_set_active_view,
        args=("search",),
    )


def _render_news_pool() -> None:
    st.markdown(t("<p class=\"section-title\">已分析新闻</p>", "<p class=\"section-title\">Analyzed news</p>"), unsafe_allow_html=True)
    st.markdown(
        t("<p class=\"section-note\">按主题、审核状态或相关性快速缩小范围。</p>", "<p class=\"section-note\">Filter by topic, review status, or relevance.</p>"),
        unsafe_allow_html=True,
    )
    news_df = _visible_news_frame(_enrich_with_quality(load_cases(db_path=_db_path())))
    filter_cols = st.columns([1, 1, 1, 1])
    topic_filter = filter_cols[0].selectbox(
        t("主题筛选", "Topic filter"),
        ["全部"] + sorted(news_df["topic_name"].dropna().unique().tolist())
        if not news_df.empty
        else ["全部"],
        key="news_topic_filter",
        format_func=label_formatter(),
    )
    status_filter = filter_cols[1].selectbox(
        t("审核状态", "Review status"),
        ["全部", "待审核", "已确认", "已忽略", "低相关"],
        key="news_status_filter",
        format_func=label_formatter(),
    )
    score_filter = filter_cols[2].number_input(
        t("最低相关性", "Minimum relevance"), min_value=0, max_value=100, value=0, step=5,
        key="news_score_filter",
    )
    news_sort = filter_cols[3].selectbox(
        t("排序方式", "Sort by"),
        ["最新发布", "最早发布", "质量最高", "质量最低"],
        key="news_sort",
        format_func=label_formatter(),
    )

    filtered_news = news_df.copy()
    if filtered_news.empty:
        st.info(t("还没有历史新闻。完成第一次搜索后，分析结果会自动出现在这里。", "No articles yet. Results will appear here after your first search."))
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
        localize_dataframe(news_display[[col for col in news_columns if col in news_display.columns]]),
        width="stretch",
        hide_index=True,
        column_config=_table_config(),
    )
    st.caption(t("共 {count} 条新闻", "{count} articles").format(count=len(filtered_news)))


def _render_case_library(allow_review: bool = False) -> None:
    st.markdown(t("<p class=\"section-title\">量化案例</p>", "<p class=\"section-title\">Quantified cases</p>"), unsafe_allow_html=True)
    st.markdown(
        t("<p class=\"section-note\">只展示达到入库门槛且包含可核验量化信息的新闻。</p>", "<p class=\"section-note\">News that meets the inclusion threshold and contains verifiable quantitative information.</p>"),
        unsafe_allow_html=True,
    )
    cases_df = _visible_news_frame(_enrich_with_quality(load_cases(qualified_only=True, db_path=_db_path())))
    if cases_df.empty:
        st.info(t("暂无达标案例。可以降低搜索页的入库门槛，或尝试更具体的关键词。", "No qualified cases yet. Try a lower inclusion threshold or more specific keywords."))
        return

    key_prefix = "review" if allow_review else "browse"
    case_filters = st.columns([2, 2, 2])
    case_topic = case_filters[0].selectbox(
        t("案例主题", "Case topic"),
        ["全部"] + sorted(cases_df["topic_name"].dropna().unique().tolist()),
        key=f"{key_prefix}_case_topic_filter",
        format_func=label_formatter(),
    )
    case_status = case_filters[1].selectbox(
        t("案例状态", "Case status"),
        ["全部", "待审核", "已确认", "已忽略"],
        key=f"{key_prefix}_case_status_filter",
        format_func=label_formatter(),
    )
    company_query = case_filters[2].text_input(
        t("企业关键词", "Company keyword"), key=f"{key_prefix}_company_query"
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
        localize_dataframe(case_display[[col for col in case_columns if col in case_display.columns]]),
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
                t("选择要审核的案例", "Select a case to review"), list(case_options), key="review_case"
            )
            selected_status = status_col.selectbox(
                t("更新为", "Set status"), ["已确认", "待审核", "已忽略"], key="review_status",
                format_func=label_formatter(),
            )
            with button_col:
                st.write("")
                st.write("")
                if st.button(t("保存", "Save"), width="stretch", key="save_review"):
                    update_case_review_status(
                        case_options[selected_case_label], selected_status, db_path=_db_path()
                    )
                    st.rerun()

    st.download_button(
        t("📥 导出当前结果", "📥 Export current results"),
        data=_excel_bytes(filtered_cases),
        file_name=export_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _confirmed_intent_from_review(
    intent: dict[str, Any],
    selected_indexes: int | list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """将用户多选的细分方向合并为可直接搜索的最终意图。"""
    return merge_search_interpretations(intent, selected_indexes)


def _render_search_intent_review(review: dict[str, Any]) -> None:
    """渲染搜索意图确认卡，确认前不访问新闻源。"""
    intent = review.get("intent", {})
    if not isinstance(intent, dict):
        return
    if get_language() == "en" and (
        not intent.get("intent_summary_en")
        or (intent.get("clarification_question") and not intent.get("clarification_question_en"))
        or any(
            not item.get("label_en") or not item.get("description_en")
            for item in intent.get("interpretations", []) if isinstance(item, dict)
        )
    ):
        st.info("This earlier search plan has no English version. Submit your keywords again to generate a bilingual plan.")
        return
    revision = int(review.get("revision", 0) or 0)
    scope_labels = {
        "broad": t("范围较广", "Broad scope"),
        "focused": t("方向适中，可能有歧义", "Focused, with possible ambiguity"),
        "specific": t("目标较明确", "Specific intent"),
    }

    st.markdown(t("### 🧠 搜索前，请确认我的理解", "### 🧠 Confirm your search intent"))
    with st.container(border=True):
        if review.get("fallback_reason"):
            st.warning(
                t("AI 本次未返回可用的搜索意图格式，已按关键词结构生成可确认的备用方案。", "AI could not return a usable search plan. A fallback based on your keywords is ready to confirm.")
            )
            st.caption(t("已保留原始关键词。你可以直接确认，也可以补充更具体的搜索目标。", "Your original keywords are preserved. Confirm below or refine your request."))
        scope = str(intent.get("scope_level", "specific"))
        st.caption(t("搜索范围：{scope}", "Search scope: {scope}").format(scope=scope_labels.get(scope, t('待确认', 'To confirm'))))
        st.caption(t("新闻语言：{language}", "Article language: {language}").format(
            language=_news_language_label(_selected_news_language()),
        ))
        st.markdown(
            t("**我理解您想查找：**\n\n{intent}", "**You want to find:**\n\n{intent}").format(intent=intent_text(intent, "intent_summary"))
        )

        interpretations = intent.get("interpretations", [])
        selected_indexes: list[int] = []
        available_options: list[tuple[int, dict[str, Any]]] = []
        if isinstance(interpretations, list) and interpretations:
            question = str(
                intent_text(intent, "clarification_question")
                or t("请选择一个或多个希望检索的细分方向。", "Select one or more directions to explore.")
            )
            st.markdown(f"**{question}**")
            available_options = [
                (index, item)
                for index, item in enumerate(interpretations)
                if isinstance(item, dict)
            ][:4]
            try:
                recommended = int(
                    intent.get("recommended_interpretation_index", 0) or 0
                )
            except (TypeError, ValueError):
                recommended = 0
            available_indexes = {index for index, _ in available_options}
            if recommended not in available_indexes and available_options:
                recommended = available_options[0][0]
            for row_start in range(0, len(available_options), 2):
                option_columns = st.columns(2)
                for offset, (index, item) in enumerate(
                    available_options[row_start : row_start + 2]
                ):
                    position = row_start + offset
                    with option_columns[offset]:
                        selected = st.checkbox(
                            intent_text(item, "label") or t("方向 {number}", "Direction {number}").format(number=position + 1),
                            value=index == recommended,
                            help=intent_text(item, "description"),
                            key=f"intent_option_{revision}_{index}",
                        )
                        description = intent_text(item, "description").strip()
                        if description:
                            st.caption(description)
                        if selected:
                            selected_indexes.append(index)
            st.caption(t("以上方向可单选或多选。", "Select one or more directions."))
        elif intent.get("clarification_question"):
            st.markdown(f"**{intent_text(intent, 'clarification_question')}**")

        custom_intent = st.text_area(
            t("或者，直接输入你真正想搜索的内容", "Or describe exactly what you want to find"),
            placeholder=(
                t("例如：我想搜索中国企业在海外销售的产品，以及国内供应链如何支持这些产品研发。", "For example: Find products that Chinese companies sell overseas and how domestic supply chains support their R&D.")
            ),
            key=f"intent_custom_{revision}",
        )
        custom_query = custom_intent.strip()
        if custom_query:
            st.caption(t("将以你输入的内容为准，直接生成检索词并开始搜索。", "Your description will be used to generate queries and start the search."))

        selected_count = 1 if custom_query else max(1, len(selected_indexes))
        articles_per_intent = int(review.get("max_articles", 8))
        planned_articles = calculate_multi_intent_article_limit(
            articles_per_intent,
            selected_count,
        )
        if not custom_query and selected_count > 1:
            uncapped_articles = articles_per_intent * selected_count
            quantity_text = (
                t("已选择 {count} 个方向：按每方向 {per_direction} 篇扩大共享结果池，本次最多处理 {total} 篇", "{count} directions selected, with {per_direction} articles per direction: up to {total} articles in the shared result pool.").format(count=selected_count, per_direction=articles_per_intent, total=planned_articles)
            )
            if planned_articles < uncapped_articles:
                quantity_text += t("（已达到单次 40 篇上限）", " (40-article limit reached)")
            st.info(quantity_text)
            st.caption(
                t("多个方向共享本次结果池；相关性筛选后，不保证每个方向最终都正好返回相同数量。", "Selected directions share one result pool. Relevance filtering may leave a different number of articles for each direction.")
            )

        confirm_clicked = st.button(
            t("✓ 确认并开始搜索", "✓ Confirm and search"),
            type="primary",
            width="stretch",
            key=f"confirm_intent_{revision}",
            disabled=(
                bool(available_options)
                and not selected_indexes
                and not custom_query
            ),
        )

    if confirm_clicked:
        request_query = str(review.get("original_query", ""))
        fallback_reason = str(review.get("fallback_reason", ""))
        intent_to_confirm = intent
        indexes_to_confirm: list[int] | None = selected_indexes
        if custom_query:
            request_query = custom_query
            try:
                with st.spinner(t("AI 正在生成检索词…", "AI is generating search queries…"), show_time=True):
                    custom_model = _analyze_intent_for_session(custom_query)
                fallback_reason = ""
            except Exception as exc:
                logger.warning("AI 理解自定义搜索内容失败: %s", exc)
                custom_model = fallback_search_intent(
                    custom_query,
                    english_query=(
                        str(review.get("english_keyword", "")) or None
                    ),
                )
                fallback_reason = str(exc) or type(exc).__name__
            intent_to_confirm = custom_model.model_dump()
            intent_to_confirm["interpretations"] = []
            intent_to_confirm["needs_clarification"] = False
            intent_to_confirm["clarification_question"] = ""
            intent_to_confirm["clarification_question_en"] = ""
            indexes_to_confirm = None

        confirmed = _confirmed_intent_from_review(
            intent_to_confirm,
            indexes_to_confirm,
        )
        confirmed["used_fallback"] = bool(fallback_reason)
        confirmed["fallback_reason"] = fallback_reason
        selected_count = int(confirmed.get("selected_intent_count", 1) or 1)
        articles_per_intent = int(review.get("max_articles", 8))
        planned_articles = calculate_multi_intent_article_limit(
            articles_per_intent,
            selected_count,
        )
        st.session_state.confirmed_search_request = {
            "original_query": request_query,
            "intent": confirmed,
            "max_articles": planned_articles,
            "articles_per_intent": articles_per_intent,
            "selected_intent_count": selected_count,
            "min_score": int(review.get("min_score", 70)),
            "english_keyword": str(review.get("english_keyword", "")),
            "news_language": _selected_news_language(),
            "output_language": get_language(),
        }
        st.session_state.pop("search_intent_review", None)
        st.session_state.last_search_keyword = request_query
        st.rerun()


def _render_search_page(cloud_demo: bool) -> None:
    active_task_id = _active_search_task_id()
    active_task = (
        _owned_task(active_task_id) if active_task_id else None
    )
    if not active_task_id:
        _restore_latest_search()
    task_notice = str(st.session_state.pop("search_task_notice", "")).strip()
    if task_notice:
        st.warning(localize(task_notice))
    search_diagnostics = st.session_state.pop("search_task_diagnostics", None)
    if search_diagnostics:
        st.caption(t(
            "找到 {candidates} 条新闻链接，未能取得可分析的正文。链接解析失败 {unresolved} 条；正文不可用 {unavailable} 条；正文过短 {too_short} 条；其他抓取失败 {errors} 条。",
            "Found {candidates} news links, but no usable article text. Unresolved links: {unresolved}; unavailable text: {unavailable}; text too short: {too_short}; other extraction failures: {errors}.",
        ).format(**search_diagnostics))
        if search_diagnostics.get("fallback_used"):
            st.caption(t("已尝试 Bing 备用新闻源。", "The Bing fallback news source was also tried."))
        if search_diagnostics.get("language_mismatch"):
            st.caption(t(
                "另有 {count} 篇非英文原文已跳过。中文模式可搜索中英文文章。",
                "Skipped {count} non-English articles. Switch to Chinese to search both languages.",
            ).format(count=search_diagnostics["language_mismatch"]))
        st.info(t(
            "本次未进入新闻评分和正文分析，因此这两个步骤没有调用 AI。搜索意图确认是单独的步骤。可以调整关键词后重试。",
            "News scoring and content analysis did not run, so neither step called the AI. Search intent confirmation is a separate step. Try adjusting your keywords and searching again.",
        ))
    st.markdown(
        t("\n        <div class=\"news-hero\">\n            <div class=\"hero-kicker\">Travel Intelligence Search</div>\n            <div class=\"main-header\">找到值得读的文旅新闻</div>\n            <p class=\"sub-header\">输入一个公司、事件或行业关键词。系统会搜索真实新闻、分析相关性，<br>并把包含可核验数字的内容整理成案例。</p>\n        </div>\n        ", "\n        <div class=\"news-hero\">\n            <div class=\"hero-kicker\">Travel Intelligence Search</div>\n            <div class=\"main-header\">Find travel news worth reading</div>\n            <p class=\"sub-header\">Enter a company, event, or industry keyword. Search real news, assess relevance,<br>and turn verifiable figures into useful cases.</p>\n        </div>\n        "),
        unsafe_allow_html=True,
    )

    with st.container(key="news_search_form"):
        st.markdown(
            t("\n            <div class=\"search-form-title\">🔍 搜索新闻</div>\n            <div class=\"search-form-note\">输入企业、品牌、事件、地区或行业指标；AI 先与你确认检索目标，再开始搜索。</div>\n            ", "\n            <div class=\"search-form-title\">🔍 Search news</div>\n            <div class=\"search-form-note\">Search a company, brand, event, region, or industry metric. Confirm the search intent with AI before searching.</div>\n            "),
            unsafe_allow_html=True,
        )
        search_col, submit_col = st.columns([7, 2])
        with search_col:
            search_keyword = st.text_input(
                t("搜索关键词", "Search keywords"),
                value=st.session_state.get("last_search_keyword", ""),
                placeholder=t("输入你想搜索的新闻关键词…", "Enter the news keywords you want to search…"),
                label_visibility="collapsed",
                key="search_keyword",
            )
        with submit_col:
            run_button = st.button(
                t("下一步：确认目标", "Next: confirm intent"),
                key="search_submit",
                type="primary",
                width="stretch",
                disabled=bool(
                    active_task
                    and active_task.get("status") == "running"
                ),
            )
        with st.expander(t("搜索设置", "Search settings")):
            st.caption(t(
                "新闻语言：中文 + 英文。切换到英文模式后，只检索和显示英文新闻。",
                "Article language: English only. Chinese mode includes both Chinese and English articles.",
            ))
            option_cols = st.columns(2)
            max_articles = option_cols[0].slider(
                t("每个检索方向的文章数量", "Articles per direction"),
                min_value=1,
                max_value=20,
                value=8,
                step=1,
                key="max_articles",
                help=(
                    t("选择多个细分意图时，文章数量会按方向数增加，单次最多处理 40 篇。数量越多，分析时间和 AI 用量越高。", "Selecting multiple directions increases the article limit, up to 40 per search. More articles take longer and use more AI capacity.")
                ),
            )
            min_score = option_cols[1].slider(
                t("案例入库门槛", "Case inclusion threshold"), min_value=0, max_value=100, value=70, step=5,
                key="min_score",
                help=t("低于门槛的结果仍会保留在新闻池。", "Results below this threshold are still kept in the news pool."),
            )
            english_keyword = st.text_input(
                t("英文关键词（可选）", "English keywords (optional)"),
                key="english_keyword",
                placeholder=t("例如 overseas products China supply chain R&D", "For example: overseas products China supply chain R&D"),
                help=t("用于搜索 Reuters、BBC、Bloomberg 等英文新闻源。", "Search English-language sources such as Reuters, BBC, and Bloomberg."),
            )
        st.markdown(
            t("\n            <div class=\"search-examples\">\n                <strong>搜索示例：</strong>半导体产业研报 · AI 行业应用与商业化 ROI · AI 市场份额与竞争格局\n            </div>\n            ", "\n            <div class=\"search-examples\">\n                <strong>Try searching: </strong>Semiconductor industry research · AI adoption and ROI · AI market share and competition\n            </div>\n            "),
            unsafe_allow_html=True,
        )

    if cloud_demo:
        st.caption(t("无需登录；搜索记录仅在当前浏览器会话内保留。", "No sign-in required. Search history is kept only for this browser session."))

    if run_button:
        keyword = search_keyword.strip()
        if not keyword:
            st.error(t("请输入一个新闻关键词。", "Enter a news keyword."))
        else:
            st.session_state.last_search_keyword = keyword
            st.session_state.pop("last_run_summary", None)
            st.session_state.pop("fetched_results", None)
            st.session_state.pop("fetched_articles", None)
            st.session_state.pop("fetched_search_intent", None)
            st.session_state.pop("pending_analysis", None)
            st.session_state.pop("confirmed_search_request", None)

            try:
                with st.spinner(t("🧠 AI 正在理解你想了解的新闻方向…", "🧠 AI is interpreting your search intent…"), show_time=True):
                    search_intent_model = _analyze_intent_for_session(keyword)
                intent_fallback_reason = ""
            except Exception as exc:
                logger.warning("AI 搜索意图理解失败，使用原始查询: %s", exc)
                search_intent_model = fallback_search_intent(
                    keyword,
                    english_query=english_keyword.strip() or None,
                )
                intent_fallback_reason = str(exc) or type(exc).__name__
            st.session_state.search_intent_review = {
                "original_query": keyword,
                "intent": search_intent_model.model_dump(),
                "fallback_reason": intent_fallback_reason,
                "max_articles": int(max_articles),
                "min_score": int(min_score),
                "english_keyword": english_keyword.strip(),
                "revision": 0,
            }
            st.rerun()

    intent_review = st.session_state.get("search_intent_review")
    if isinstance(intent_review, dict):
        _render_search_intent_review(intent_review)

    confirmed_request = st.session_state.pop("confirmed_search_request", None)
    if isinstance(confirmed_request, dict):
        search_intent = confirmed_request.get("intent", {})
        if not isinstance(search_intent, dict):
            search_intent = {}
        confirmed_keyword = str(
            search_intent.get("intent_summary")
            or confirmed_request.get("original_query", "")
        ).strip()
        chinese_queries = list(search_intent.get("chinese_queries", []) or [])
        primary_query = (
            str(chinese_queries[0]).strip()
            if chinese_queries
            else confirmed_keyword
        )
        max_articles_value = int(confirmed_request.get("max_articles", 8))
        min_score_value = int(confirmed_request.get("min_score", 70))
        manual_english_keyword = str(
            confirmed_request.get("english_keyword", "")
        ).strip()
        news_language = _selected_news_language()
        if news_language == "en":
            english_queries = list(search_intent.get("english_queries") or [])
            primary_query = manual_english_keyword or (
                str(english_queries[0]).strip() if english_queries
                else str(confirmed_request.get("original_query") or confirmed_keyword)
            )
        st.session_state.fetched_search_intent = search_intent
        task_id = start_background_search(
            {
                "primary_query": primary_query,
                "confirmed_keyword": confirmed_keyword,
                "original_query": confirmed_request.get("original_query", ""),
                "max_articles": max_articles_value,
                "min_score": min_score_value,
                "english_keyword": manual_english_keyword,
                "chinese_queries": chinese_queries,
                "english_queries": list(
                    search_intent.get("english_queries", []) or []
                ),
                "search_intent": search_intent,
                "news_language": news_language,
                "output_language": get_language(),
            },
            workspace=_workspace(),
        )
        st.session_state.setdefault("search_history", []).append(task_id)
        st.session_state.active_search_task_id = task_id
        st.rerun()

    active_task_id = _active_search_task_id()
    if active_task_id:
        _render_active_search_task(active_task_id)
        return

    if not run_button and st.session_state.get("fetched_articles"):
        articles = st.session_state.fetched_articles
        _render_article_cards(
            articles,
            st.session_state.get("fetched_keyword", ""),
            st.session_state.get("fetched_results"),
        )

        pending_analysis = st.session_state.get("pending_analysis")
        if pending_analysis:
            stage = pending_analysis.get("stage", "source")
            if stage == "source":
                st.markdown(t("### AI 正在初筛搜索相关性与来源", "### AI is screening relevance and sources"))
                progress_text = t("新闻已展示，正在生成相关性和来源 AI 评分…", "Articles are ready. AI is scoring relevance and sources…")
            else:
                st.markdown(t("### AI 正在分析新闻正文", "### AI is analyzing article content"))
                progress_text = t("基础分已生成，正在分析正文质量…", "Base scores are ready. Analyzing content quality…")
            progress_bar = st.progress(0.0, text=progress_text)

            def update_progress(message: str, value: float) -> None:
                progress_bar.progress(value, text=localize(message))

            try:
                if stage == "source":
                    source_errors = score_sources_with_ai(
                        articles,
                        original_query=pending_analysis["keyword"],
                        search_intent=pending_analysis.get("search_intent"),
                        progress_callback=update_progress,
                    )
                    visible_articles = [
                        article
                        for article in articles
                        if article.get("search_relevance_scored") is not True
                        or int(article.get("search_relevance_score", 0) or 0)
                        >= SEARCH_RELEVANCE_DISPLAY_THRESHOLD
                    ]
                    filtered_count = len(articles) - len(visible_articles)
                    articles[:] = visible_articles
                    pending_analysis["relevance_filtered"] = filtered_count
                    if not articles:
                        st.session_state.pop("pending_analysis", None)
                        st.session_state.fetched_articles = []
                        save_last_search_state({}, db_path=_db_path())
                        progress_bar.progress(1.0, text=t("搜索相关性初筛完成", "Relevance screening complete"))
                        st.warning(
                            t("本次找到的候选文章与搜索意图的相关性均低于 {threshold} 分，已不作为有效结果展示。", "All candidate articles scored below {threshold} for relevance and have been excluded from the results.").format(threshold=SEARCH_RELEVANCE_DISPLAY_THRESHOLD)
                        )
                        return
                    st.session_state.fetched_articles = articles
                    pending_analysis["stage"] = "body"
                    pending_analysis["source_errors"] = source_errors
                    st.session_state.pending_analysis = pending_analysis
                    _persist_latest_search(
                        pending_analysis["keyword"],
                        articles,
                        search_intent=pending_analysis.get("search_intent"),
                    )
                    progress_bar.progress(
                        1.0,
                        text=t("相关性与来源初筛完成，基础分已生成", "Relevance and source screening complete. Base scores are ready."),
                    )
                    st.rerun()

                st.session_state.pop("pending_analysis", None)
                analysis_started = perf_counter()
                run_summary = run_pipeline(
                    industry_keyword=pending_analysis["keyword"],
                    min_score=pending_analysis["min_score"],
                    max_articles=pending_analysis["max_articles"],
                    topic=None,
                    trigger_type="manual",
                    progress_callback=update_progress,
                    pre_fetched_articles=articles,
                    pre_screen_completed=True,
                    db_path=_db_path(),
                )
                run_summary["search_seconds"] = pending_analysis["search_seconds"]
                run_summary["analysis_seconds"] = perf_counter() - analysis_started
                run_summary["relevance_filtered"] = pending_analysis.get(
                    "relevance_filtered", 0
                )
                details = list(run_summary.get("details", []))
                ranked_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
                for index, article in enumerate(articles):
                    if index < len(details):
                        detail = details[index]
                    else:
                        detail = {
                            "title": str(article.get("title", "")),
                            "url": str(article.get("url", "")),
                            "score": article.get("search_relevance_score"),
                            "analysis_status": "未完成",
                            "qualification_status": "-",
                            "storage_status": "未写入",
                            "reason": "分析任务提前结束",
                        }
                    ranked_pairs.append((article, detail))

                def ranking_score(
                    pair: tuple[dict[str, Any], dict[str, Any]],
                ) -> float:
                    _, detail = pair
                    recommendation = detail.get("recommendation_score")
                    if isinstance(recommendation, (int, float)) and not isinstance(
                        recommendation, bool
                    ):
                        return float(recommendation)
                    relevance = detail.get("score", 0)
                    if isinstance(relevance, (int, float)) and not isinstance(
                        relevance, bool
                    ):
                        return float(calculate_recommendation_score(relevance, 0))
                    return -1.0

                ranked_pairs.sort(key=ranking_score, reverse=True)
                articles[:] = [article for article, _ in ranked_pairs]
                run_summary["details"] = [detail for _, detail in ranked_pairs]
                st.session_state.last_run_summary = run_summary
                result_map = {
                    index: detail
                    for index, detail in enumerate(run_summary.get("details", []))
                }
                st.session_state.fetched_articles = articles
                st.session_state.fetched_results = result_map
                _persist_latest_search(
                    pending_analysis["keyword"],
                    articles,
                    result_map,
                    run_summary,
                    search_intent=pending_analysis.get("search_intent"),
                )
                progress_bar.progress(1.0, text=t("分析完成", "Analysis complete"))
                st.rerun()
            except ValueError as exc:
                st.error(t("配置错误：{error}", "Configuration error: {error}").format(error=localize(str(exc))))
            except Exception as exc:
                logger.error("手动任务失败: %s", exc, exc_info=True)
                st.error(t("分析失败：{error}", "Analysis failed: {error}").format(error=localize(str(exc))))

    last_run_summary = st.session_state.get("last_run_summary")
    if last_run_summary:
        _render_run_summary(last_run_summary)

    st.divider()
    news_tab, cases_tab = st.tabs([t("📰 我的新闻", "📰 My news"), t("📚 我的案例", "📚 My cases")])
    with news_tab:
        _render_news_pool()
    with cases_tab:
        _render_case_library()


def _render_history_page() -> None:
    st.button(
        t("← 返回搜索", "← Back to search"), key="back_to_search",
        on_click=_set_active_view, args=("search",),
    )
    st.markdown(t("## 我的记录", "## My history"))
    st.caption(t(
        "仅显示当前浏览器会话的记录。刷新、新开标签页或关闭后重新打开可能开始新会话；重要结果请及时导出。",
        "Only this browser session's records are shown. Reloading, opening another tab, or reopening the page may start a new session. Export any results you want to keep.",
    ))
    _render_background_task_banner()
    history_tab, news_tab, cases_tab = st.tabs([
        t("🔎 搜索记录", "🔎 Search history"),
        t("📰 我的新闻", "📰 My news"),
        t("📚 我的案例", "📚 My cases"),
    ])
    with history_tab:
        task_ids = st.session_state.get("search_history", [])
        if not task_ids:
            st.info(t("本次会话还没有搜索记录。", "No searches in this session yet."))
        statuses = {
            "running": t("进行中", "In progress"),
            "completed": t("已完成", "Completed"),
            "failed": t("失败", "Failed"),
        }
        for task_id in reversed(task_ids):
            snapshot = _owned_task(task_id)
            if snapshot is None:
                continue
            with st.container(border=True):
                st.write(snapshot.get("original_query") or snapshot.get("keyword", ""))
                st.caption(f"{snapshot.get('created_at', '')} · {statuses.get(snapshot['status'], snapshot['status'])}")
                st.caption(localize(snapshot.get("message", "")))
                if snapshot["status"] == "completed" and snapshot.get("articles"):
                    if st.button(
                        t("查看结果", "View results"), key=f"history_result_{task_id}",
                        disabled=bool(_active_search_task_id()),
                    ):
                        _apply_completed_search_task(snapshot)
                        _set_active_view("search")
                        st.rerun()
    with news_tab:
        _render_news_pool()
    with cases_tab:
        _render_case_library(allow_review=True)


cloud_demo = is_cloud_demo()
if st.session_state.get("active_view") not in {"search", "history"}:
    st.session_state.active_view = "search"

with st.sidebar:
    st.markdown(t("### 📰 文旅新闻搜索", "### 📰 Travel News Search"))
    st.caption(t("搜索真实新闻，沉淀量化案例。", "Find real news. Build evidence-backed cases."))
    st.button(
        t("🔎 搜索首页", "🔎 Search"), key="sidebar_search", width="stretch",
        type="primary" if st.session_state.active_view == "search" else "secondary",
        on_click=_set_active_view, args=("search",),
    )
    st.button(
        t("📋 我的记录", "📋 My history"), key="sidebar_history", width="stretch",
        type="primary" if st.session_state.active_view == "history" else "secondary",
        on_click=_set_active_view, args=("history",),
    )
    st.divider()
    st.caption(t("无需登录 · 记录仅限当前会话", "No sign-in · Records are private to this session"))

with st.container(key="page_toolbar", horizontal=True, horizontal_alignment="right"):
    st.button(
        t("📋 我的记录", "📋 My history"), key="open_history", width="content",
        help=t("查看本次会话的搜索、新闻和案例", "View this session's searches, news, and cases"),
        on_click=_set_active_view, args=("history",),
    )
    st.button(
        "中/ENG", key="language_toggle", on_click=toggle_language,
        help=t("当前为中文，点击切换英文", "Currently English; click to switch to Chinese"),
        width="content",
    )

if st.session_state.active_view == "history":
    _render_history_page()
else:
    _render_search_page(cloud_demo)

st.caption(t(
    "本次会话的搜索与其他访客隔离。关闭页面前，可将所需结果导出保存。",
    "Your session's searches are separate from other visitors. Export results you want to keep before leaving.",
))
st.markdown('<footer class="footer-note">Zibo Jiang · Fosun</footer>', unsafe_allow_html=True)
