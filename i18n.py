"""Session-scoped UI translations; stored news and research content stay unchanged."""

from __future__ import annotations

import json
import re
from functools import partial
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    import pandas as pd


def init_language() -> None:
    """Default to English, restoring an explicit language from the URL on reload."""
    if st.session_state.get("ui_language") not in {"en", "zh"}:
        requested = st.query_params.get("lang", "en")
        st.session_state["ui_language"] = requested if requested in {"en", "zh"} else "en"


def get_language() -> str:
    return "zh" if st.session_state.get("ui_language") == "zh" else "en"


def toggle_language() -> None:
    language = "zh" if get_language() == "en" else "en"
    st.session_state["ui_language"] = language
    st.query_params["lang"] = language


def t(chinese: str, english: str) -> str:
    """Choose a UI string at render time, never at module import time."""
    return chinese if get_language() == "zh" else english


def intent_text(data: dict[str, Any], field: str) -> str:
    """Choose generated intent copy without changing the underlying search plan."""
    return str(data.get(field if get_language() == "zh" else f"{field}_en") or "")


# Stable database/status values are translated only at the presentation boundary.
_LABELS = {
    "全部": "All", "待审核": "Pending review", "已确认": "Confirmed",
    "已忽略": "Ignored", "低相关": "Low relevance", "已达标": "Qualified",
    "未达标": "Not qualified", "达标": "Qualified", "相关性不足": "Low relevance",
    "成功": "Succeeded", "部分成功": "Partially succeeded", "失败": "Failed",
    "未完成": "Incomplete", "跳过": "Skipped", "未写入": "Not saved",
    "已新增": "Added", "已更新": "Updated", "已刷新": "Updated",
    "重复跳过": "Duplicate skipped", "重复": "Duplicate", "写入失败": "Save failed",
    "未知": "Unknown", "手动": "Manual", "定时": "Scheduled",
    "运行中": "Running", "未运行": "Not running", "未启用": "Disabled",
    "高质量": "High quality", "质量良好": "Good quality", "优秀": "Excellent",
    "良好": "Good", "一般": "Fair", "偏低": "Below average", "较低": "Low",
    "预筛": "Preliminary", "最新发布": "Newest first", "最早发布": "Oldest first",
    "质量最高": "Highest quality", "质量最低": "Lowest quality",
    "来源权威度": "Source credibility", "信息密度": "Information density",
    "来源权威度（AI 评分）": "Source credibility (AI assessment)",
    "由正文 AI 分析补充": "Added during AI body analysis",
    "数据含量": "Data richness", "时效性": "Freshness",
    "正文证据与可核验性": "Evidence and verifiability", "报道完整性": "Completeness",
    "信源与方法透明度": "Source and method transparency",
    "标题正文一致性": "Headline consistency", "客观与平衡性": "Objectivity and balance",
    "清晰与连贯性": "Clarity and coherence", "核心主题匹配": "Core topic match",
    "信息需求匹配": "Information need match", "语义内容覆盖": "Semantic coverage",
    "直接相关程度": "Direct relevance",
    "导出": "Export", "发现时间": "Discovered", "发布时间": "Published",
    "新闻标题": "Headline", "原文链接": "Source link", "来源": "Source",
    "主题ID": "Topic ID", "主题 ID": "Topic ID", "维度": "Dimension",
    "分类": "Category", "研究主题": "Research topic", "搜索词": "Search query",
    "新闻摘要": "Summary", "量化案例": "Quantitative cases", "证据原文": "Evidence quotes",
    "涉及企业": "Companies", "地区": "Regions", "指标标签": "Metric tags",
    "相关性": "Relevance", "搜索相关性": "Search relevance", "报道质量": "News quality",
    "新闻质量": "News quality", "推荐分": "Recommendation score", "质量评级": "Quality rating",
    "入库判定": "Qualification", "审核状态": "Review status", "编辑备注": "Notes",
    "AI 状态": "AI status", "案例判定": "Case qualification", "写入状态": "Save status",
    "原因": "Reason", "任务ID": "Task ID", "触发方式": "Trigger",
    "开始时间": "Started", "结束时间": "Finished", "状态": "Status",
    "处理": "Processed", "新增案例": "New cases", "错误数": "Errors", "错误详情": "Error details",
    "正文为空": "Article body is empty", "无法分句": "No sentences could be identified",
    "几乎不含量化数据": "Contains little quantitative data",
    "无法验证发布时间，保守评分": "Publication date could not be verified; scored conservatively",
    "发布时间超过 1 年": "Published more than one year ago",
    "标题含煽动性词汇": "Headline contains sensational wording",
    "正文不足 100 字": "Article body contains fewer than 100 characters",
    "来源权威度低于 15/25，不能判为高质量新闻": "Source credibility is below 15/25; not eligible for high quality",
    "AI 正文质量低于 35/50，不能判为高质量新闻": "Body quality is below 35/50; not eligible for high quality",
    "标题与正文严重不一致；综合分暂时封顶 59 分": "Headline substantially contradicts the article; overall score capped at 59",
    "正文存在影响核心结论、但缺乏证据支撑的严重断言": "The article contains serious unsupported claims affecting its conclusions",
    "AI 未提供评分依据": "AI did not provide a scoring rationale",
    "未提供具体依据": "No specific rationale provided",
    "AI 根据原始发布者与官网信息评估": "AI assessment based on the original publisher and official website",
    "来源可靠、正文证据充分，达到高质量新闻标准": "Reliable source and strong evidence; meets the high-quality standard",
    "报道整体可靠，但仍有部分维度可以补强": "Generally reliable reporting with some room for improvement",
    "报道质量一般，部分维度存在不足": "Fair reporting with weaknesses in some dimensions",
    "报道质量偏低，建议交叉验证": "Below-average quality; cross-check with other sources",
    "报道质量较低，信息不足或来源不明": "Low quality, insufficient information, or unclear sourcing",
    "当前为 50 分基础分，AI 正文分析完成后生成 100 分综合质量分": "Preliminary score out of 50; overall quality out of 100 follows body analysis",
    "来源权威度已由 AI 评估；正文分析完成后生成 100 分综合质量分": "Source assessed by AI; overall quality out of 100 follows body analysis",
    "正在搜索中英文新闻并提取正文…": "Searching Chinese and English news and extracting articles…",
    "正在搜索新闻并提取正文…": "Searching news and extracting articles…",
    "没有找到可分析的文章": "No articles available for analysis",
    "新闻已展示，正在计算相关性和来源权威度…": "Articles found; assessing relevance and source credibility…",
    "候选新闻均未达到搜索相关性展示门槛": "No articles met the relevance threshold",
    "相关性初筛完成，正在分析新闻正文质量…": "Relevance screening complete; analyzing article quality…",
    "搜索和 AI 分析已完成": "Search and AI analysis complete",
    "搜索任务失败": "Search failed", "后台搜索运行中…": "Search running in the background…",
    "后台任务运行中…": "Task running in the background…",
    "没有找到可分析的文章。": "No articles available for analysis.",
    "分析任务提前结束": "Analysis ended early",
    "未提取到量化案例": "No quantitative cases extracted",
    "没有可在正文中验证的证据原文": "No evidence quotes could be verified in the article",
    "未获取到有效文章": "No valid articles found",
    "AI 模型未返回可用分析结果": "The AI model did not return a usable analysis",
    "找到重复记录，但更新当前评分失败": "A duplicate record was found, but its scores could not be updated",
    "数据库未接受该记录，请查看运行日志": "The database rejected this record; see the run log",
    "正在写入新闻池并检查重复…": "Saving news and checking for duplicates…",
    "等待开始": "Waiting to start", "任务已创建": "Task created",
    "任务完成": "Task complete",
    "任务部分完成，请查看错误和逐篇明细": "Task partially completed; see errors and article details",
    "任务失败，请查看错误详情": "Task failed; see error details",
}

_PATTERNS = [
    (r"搜索相关性 (\d+)/100，低于正文 AI 分析门槛 (\d+)", r"Search relevance \1/100 is below the body-analysis threshold of \2"),
    (r"相关性 (\d+) 低于门槛 (\d+)", r"Relevance \1 is below the threshold of \2"),
    (r"有 (\d+) 条分析结果未写入数据库，请查看逐篇明细", r"\1 analysis results could not be saved; see article details"),
    (r"AI 正在初筛 (\d+) 篇新闻的相关性与来源（(\d+) 路）…", r"Screening relevance and sources for \1 articles (\2 in parallel)…"),
    (r"相关性与来源初筛已完成 (\d+)/(\d+) 篇…", r"Relevance and source screening: \1/\2 articles complete…"),
    (r"AI 正在并行分析 (\d+) 篇文章（(\d+) 路）…", r"Analyzing \1 articles (\2 in parallel)…"),
    (r"AI 已完成 (\d+)/(\d+) 篇分析…", r"AI analysis: \1/\2 articles complete…"),
    (r"已提取 (\d+) 篇正文，开始相关新闻的 AI 正文分析…", r"Extracted \1 articles; starting AI analysis of relevant articles…"),
    (r"已搜索到 (\d+)/(\d+) 篇有效新闻…", r"Found \1/\2 valid articles…"),
    (r"正文仅 (\d+) 字，硬跳过", r"Only \1 characters extracted; skipped"),
    (r"正文 (\d+) 字", r"Article length: \1 characters"),
    (r"段落结构好（(\d+) 段，均长 (\d+) 字）\+2", r"Well-structured paragraphs (\1 paragraphs, \2 characters on average) +2"),
    (r"含数据句子占比 (\d+%)", r"Sentences containing data: \1"),
    (r"([\d-]+) 个月内发布", r"Published within \1 months"),
    (r"标题含 (\d+) 个煽动性词汇", r"Headline contains \1 sensational terms"),
    (r"正文仅提取到 (\d+) 字，可能提取不完整；综合分暂时封顶 59 分", r"Only \1 characters extracted; the article may be incomplete. Overall score capped at 59"),
]


def localize(value: Any, *, language: str | None = None) -> Any:
    """Translate known system text, preserving unrecognized/user-authored content."""
    if (language or get_language()) == "zh" or not isinstance(value, str):
        return value
    if value in _LABELS:
        return _LABELS[value]
    for badge in ("🟢", "🔵", "🟡", "🟠", "🔴", "⚪"):
        if value.startswith(badge + " "):
            return badge + " " + localize(value[len(badge) + 1:], language="en")
    for pattern, replacement in _PATTERNS:
        if re.fullmatch(pattern, value):
            return re.sub(pattern, replacement, value)
    if "; " in value:
        return "; ".join(localize(part, language="en") for part in value.split("; "))
    if value.startswith("AI 评估："):
        return "AI assessment: " + localize(value.removeprefix("AI 评估："), language="en")
    if value.endswith("；综合分暂时封顶 59 分"):
        return localize(value.removesuffix("；综合分暂时封顶 59 分"), language="en") + "; overall score capped at 59"
    if "；" in value:
        parts = value.split("；")
        translated = [localize(part, language="en") for part in parts]
        return "; ".join(translated) if translated != parts else value
    return value


def label_formatter():
    """Freeze the render language for callbacks invoked outside a script context."""
    return partial(localize, language=get_language())


_STATUS_COLUMNS = {
    "quality_label", "review_status", "analysis_status", "qualification_status",
    "storage_status", "质量评级", "审核状态", "入库判定", "AI 状态", "案例判定",
    "写入状态", "状态", "触发方式", "原因", "status", "trigger_type",
    "错误详情", "errors",
}


def localize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a localized display/export copy without changing source data or IDs."""
    display = df.copy()
    if get_language() == "en":
        for column in display.columns:
            if column in {"错误详情", "errors"}:
                display[column] = display[column].map(_localize_errors)
            elif column in _STATUS_COLUMNS:
                display[column] = display[column].map(localize)
        display = display.rename(columns=localize)
    return display


def _localize_errors(value: Any) -> Any:
    """Task logs store error lists as JSON; translate their known messages only."""
    if isinstance(value, str):
        try:
            errors = json.loads(value)
        except (ValueError, TypeError):
            return localize(value)
        if isinstance(errors, list) and all(isinstance(error, str) for error in errors):
            return json.dumps([localize(error) for error in errors], ensure_ascii=False)
    return value
