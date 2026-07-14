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
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from access_control import (
    admin_password_configured,
    is_cloud_demo,
    local_admin_without_password,
    verify_admin_password,
)
from agent import run_pipeline
from database import (
    JSON_COLUMNS,
    format_json_list_for_display,
    get_scheduler_health,
    initialize_database,
    load_cases,
    load_task_runs,
    load_topics,
    update_case_review_status,
    update_topic,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="文旅行业情报 Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {font-size: 2rem; font-weight: 750; color: #172033; margin-bottom: .15rem;}
    .sub-header {color: #667085; font-size: .95rem; margin-bottom: 1rem;}
    .stDataFrame {border: 1px solid #eaecf0; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_database()


def _render_admin_access() -> bool:
    """渲染管理员解锁区，返回当前会话是否有管理权限。"""
    if local_admin_without_password():
        return True

    st.markdown("### 🔐 管理员")
    if st.session_state.get("admin_authenticated", False):
        st.success("管理操作已解锁")
        if st.button("退出管理模式", width="stretch"):
            st.session_state.admin_authenticated = False
            st.rerun()
        return True

    if not admin_password_configured():
        st.warning("Cloud Secrets 未配置 ADMIN_PASSWORD，管理操作已锁定。")
        return False

    with st.form("admin_login_form", clear_on_submit=True):
        candidate = st.text_input("管理员口令", type="password")
        submitted = st.form_submit_button("解锁管理操作", width="stretch")
    if submitted:
        if verify_admin_password(candidate):
            st.session_state.admin_authenticated = True
            st.rerun()
        st.error("口令不正确")
    return False


def _topic_label(topic: dict[str, Any]) -> str:
    return f"{topic['topic_id']} · {topic['topic_name']}"


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


topics_df = load_topics(enabled_only=True)
topic_records = topics_df.to_dict(orient="records")
topic_options = {_topic_label(topic): topic for topic in topic_records}
cloud_demo = is_cloud_demo()

with st.sidebar:
    is_admin = _render_admin_access()
    if not is_admin:
        st.caption("当前为公开只读模式；可查看数据和原文链接。")
    st.divider()
    st.markdown("### 🎯 手动研究")
    selected_label = st.selectbox(
        "研究主题", options=list(topic_options), disabled=not is_admin
    )
    selected_topic = topic_options[selected_label]

    search_keyword = st.text_input(
        "新闻搜索词",
        value=str(selected_topic.get("search_keywords", "")),
        help="可在单次任务中临时调整，不会改写主题配置。",
        disabled=not is_admin,
    )
    min_score = st.slider(
        "案例入库门槛",
        min_value=0,
        max_value=100,
        value=int(selected_topic.get("min_score", 70)),
        step=5,
        disabled=not is_admin,
    )
    max_articles = st.number_input(
        "最大文章数",
        min_value=1,
        max_value=30,
        value=int(selected_topic.get("max_articles", 8)),
        disabled=not is_admin,
    )
    run_button = st.button(
        "🚀 抓取并分析",
        type="primary",
        width="stretch",
        disabled=not is_admin,
        help="需要管理员权限" if not is_admin else None,
    )

    st.divider()
    if cloud_demo:
        st.markdown("### ☁️ Cloud 展示模式")
        st.info("网页可公开访问")
        st.caption("Community Cloud 不运行本地定时 worker。")
    else:
        st.markdown("### ⏱️ 定时 worker")
        scheduler_health = get_scheduler_health()
        if scheduler_health["running"]:
            st.success("运行中")
        else:
            st.warning("未运行")
        if scheduler_health.get("last_heartbeat"):
            st.caption(f"最后心跳：{scheduler_health['last_heartbeat']}")
        st.caption("启动命令：`python3 worker.py`")

    st.divider()
    st.caption("文旅行业情报 Agent · MVP v1")

st.markdown('<p class="main-header">📰 文旅行业情报 Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">28 个研究主题 · 真实新闻链接 · 量化案例提炼 · 结构化导出</p>',
    unsafe_allow_html=True,
)

if cloud_demo:
    st.info(
        "当前为 Streamlit Community Cloud 快速展示版。"
        "云端 SQLite 数据可能在应用重启或重新部署后重置；"
        "已确认数据请及时导出 Excel。"
    )

if run_button:
    if not search_keyword.strip():
        st.error("请输入新闻搜索词")
    else:
        with st.spinner(f"正在处理「{selected_label}」，请稍候…"):
            try:
                run_summary = run_pipeline(
                    industry_keyword=search_keyword.strip(),
                    min_score=min_score,
                    max_articles=int(max_articles),
                    topic=selected_topic,
                    trigger_type="manual",
                )
                result_cols = st.columns(4)
                result_cols[0].metric("分析文章", run_summary.get("processed", 0))
                result_cols[1].metric("新增案例", run_summary.get("saved", 0))
                result_cols[2].metric("低分/无数据", run_summary.get("skipped", 0))
                result_cols[3].metric("错误", len(run_summary.get("errors", [])))
                if run_summary.get("saved"):
                    st.success(f"已新增 {run_summary['saved']} 条待审核量化案例。")
                elif run_summary.get("processed"):
                    st.info("分析完成，本次没有新的达标案例。")
                for error in run_summary.get("errors", []):
                    st.error(error)
            except ValueError as exc:
                st.error(f"配置错误：{exc}")
            except Exception as exc:
                logger.error("手动任务失败: %s", exc, exc_info=True)
                st.error(f"运行异常：{exc}")

dashboard_tab, news_tab, cases_tab, topics_tab, tasks_tab = st.tabs(
    ["📊 仪表盘", "📰 新闻池", "📚 案例库", "🎯 主题管理", "⏱️ 任务中心"]
)

with dashboard_tab:
    all_news = load_cases()
    qualified_news = all_news[all_news["is_qualified"] == 1] if not all_news.empty else all_news
    metric_cols = st.columns(4)
    metric_cols[0].metric("新闻池", len(all_news))
    metric_cols[1].metric("达标案例", len(qualified_news))
    pending_count = (
        int((qualified_news["review_status"] == "待审核").sum())
        if not qualified_news.empty
        else 0
    )
    metric_cols[2].metric("待审核", pending_count)
    covered_topics = int(all_news["topic_id"].nunique()) if not all_news.empty else 0
    metric_cols[3].metric("已覆盖主题", f"{covered_topics} / 28")

    if all_news.empty:
        st.info("暂无数据。请在左侧选择主题并运行第一次采集。")
    else:
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
            recent_df = _display_cases(all_news.head(8))
            st.dataframe(
                recent_df[[col for col in recent_cols if col in recent_df.columns]],
                width="stretch",
                hide_index=True,
                column_config=_table_config(),
            )

with news_tab:
    st.markdown("#### 所有已分析新闻")
    news_df = load_cases()
    filter_cols = st.columns([2, 2, 1])
    topic_filter = filter_cols[0].selectbox(
        "主题筛选",
        ["全部"] + sorted(news_df["topic_name"].dropna().unique().tolist())
        if not news_df.empty
        else ["全部"],
        key="news_topic_filter",
    )
    status_filter = filter_cols[1].selectbox(
        "审核状态", ["全部", "待审核", "已确认", "已忽略", "低相关"]
    )
    score_filter = filter_cols[2].number_input(
        "最低相关性", min_value=0, max_value=100, value=0, step=5
    )
    filtered_news = news_df.copy()
    if not filtered_news.empty:
        if topic_filter != "全部":
            filtered_news = filtered_news[filtered_news["topic_name"] == topic_filter]
        if status_filter != "全部":
            filtered_news = filtered_news[filtered_news["review_status"] == status_filter]
        filtered_news = filtered_news[filtered_news["relevance_score"] >= score_filter]
        news_display = _display_cases(filtered_news)
        news_columns = [
            "ID", "发布时间", "新闻标题", "来源", "研究主题", "新闻摘要",
            "相关性", "入库判定", "审核状态", "原文链接",
        ]
        st.dataframe(
            news_display[[col for col in news_columns if col in news_display.columns]],
            width="stretch",
            hide_index=True,
            column_config=_table_config(),
        )
        st.caption(f"共 {len(filtered_news)} 条新闻")
    else:
        st.info("新闻池暂无数据。")

with cases_tab:
    st.markdown("#### 量化案例数据表")
    cases_df = load_cases(qualified_only=True)
    if cases_df.empty:
        st.info("暂无达到门槛且包含量化信息的案例。")
    else:
        case_filters = st.columns([2, 2, 2])
        case_topic = case_filters[0].selectbox(
            "案例主题",
            ["全部"] + sorted(cases_df["topic_name"].dropna().unique().tolist()),
            key="case_topic_filter",
        )
        case_status = case_filters[1].selectbox(
            "案例状态", ["全部", "待审核", "已确认", "已忽略"], key="case_status_filter"
        )
        company_query = case_filters[2].text_input("企业关键词")
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
            "ID", "发布时间", "新闻标题", "研究主题", "涉及企业", "地区",
            "量化案例", "证据原文", "相关性", "审核状态", "原文链接",
        ]
        st.dataframe(
            case_display[[col for col in case_columns if col in case_display.columns]],
            width="stretch",
            hide_index=True,
            column_config=_table_config(),
        )

        case_options = {
            f"#{int(row.id)} · {str(row.title)[:55]}": int(row.id)
            for row in filtered_cases.itertuples()
        }
        export_name = f"travel_intelligence_cases_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        if is_admin and case_options:
            action_col, status_col, button_col, export_col = st.columns([4, 2, 1, 2])
            selected_case_label = action_col.selectbox(
                "选择要审核的案例", list(case_options), key="review_case"
            )
            selected_status = status_col.selectbox(
                "更新为", ["已确认", "待审核", "已忽略"], key="review_status"
            )
            with button_col:
                st.write("")
                st.write("")
                if st.button("保存", width="stretch"):
                    update_case_review_status(
                        case_options[selected_case_label], selected_status
                    )
                    st.rerun()
            with export_col:
                st.write("")
                st.write("")
                st.download_button(
                    "📥 导出当前结果",
                    data=_excel_bytes(filtered_cases),
                    file_name=export_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                )
        else:
            if not is_admin:
                st.caption("公开模式可查看和导出；案例审核需管理员权限。")
            st.download_button(
                "📥 导出当前结果",
                data=_excel_bytes(filtered_cases),
                file_name=export_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

with topics_tab:
    st.markdown("#### 研究主题配置")
    st.caption("主题来自项目中的 Excel；此处修改只写入 SQLite，不改动原始 Excel。")
    editable_topics = load_topics(include_custom=False)
    editable_columns = [
        "topic_id", "dimension", "category", "topic_name", "report_frequency",
        "search_keywords", "enabled", "min_score", "max_articles", "collection_interval_hours",
    ]
    if is_admin:
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
                "min_score": st.column_config.NumberColumn("入库分数", min_value=0, max_value=100),
                "max_articles": st.column_config.NumberColumn("单次篇数", min_value=1, max_value=30),
                "collection_interval_hours": st.column_config.NumberColumn("采集间隔/小时", min_value=1),
            },
            key="topics_editor",
        )
        if st.button("💾 保存主题配置"):
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
            st.rerun()
    else:
        st.info("公开模式仅展示研究主题，编辑需管理员权限。")
        st.dataframe(
            editable_topics[editable_columns], width="stretch", hide_index=True
        )

with tasks_tab:
    st.markdown("#### 定时任务与运行日志")
    health_cols = st.columns(3)
    if cloud_demo:
        health_cols[0].metric("运行模式", "Cloud 展示")
        health_cols[1].metric("定时 Worker", "未启用")
        health_cols[2].metric("数据持久化", "临时")
        st.info("展示版支持管理员手动抓取，不运行后台定时 worker。")
    else:
        current_health = get_scheduler_health()
        health_cols[0].metric("Worker 状态", "运行中" if current_health["running"] else "未运行")
        health_cols[1].metric("最后心跳", current_health.get("last_heartbeat") or "-")
        health_cols[2].metric("扫描间隔", f"{os.getenv('SCHEDULER_POLL_MINUTES', '30')} 分钟")
        if not current_health["running"]:
            st.info("在另一个终端执行 `python3 worker.py` 即可启动本地周期采集。")

    runs_df = load_task_runs(limit=100)
    if runs_df.empty:
        st.info("暂无任务运行记录。")
    else:
        status_map = {"success": "成功", "partial": "部分成功", "failed": "失败"}
        trigger_map = {"manual": "手动", "scheduled": "定时"}
        display_runs = runs_df.copy()
        display_runs["status"] = display_runs["status"].map(status_map).fillna(display_runs["status"])
        display_runs["trigger_type"] = (
            display_runs["trigger_type"].map(trigger_map).fillna(display_runs["trigger_type"])
        )
        display_runs = display_runs.rename(
            columns={
                "id": "任务ID", "topic_id": "主题ID", "keyword": "搜索词",
                "trigger_type": "触发方式", "started_at": "开始时间", "finished_at": "结束时间",
                "status": "状态", "processed": "处理", "saved": "新增案例",
                "skipped": "跳过", "error_count": "错误数", "errors": "错误详情",
            }
        )
        if not is_admin:
            public_columns = [
                "任务ID", "主题ID", "触发方式", "开始时间", "结束时间",
                "状态", "处理", "新增案例", "跳过", "错误数",
            ]
            display_runs = display_runs[
                [column for column in public_columns if column in display_runs.columns]
            ]
        st.dataframe(display_runs, width="stretch", hide_index=True)

st.markdown("---")
st.caption(
    f"数据库：`{os.getenv('DATABASE_PATH', 'data/industry_news.db')}` · "
    f"Gemini：`{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}` · "
    f"模式：`{'cloud_demo' if cloud_demo else 'local'}` · "
    "原始主题 Excel 不会被程序修改"
)
