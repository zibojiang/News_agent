"""将本次新闻分析结果转换为可编辑、可导出的高质量新闻表。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent import calculate_recommendation_score
from database import format_json_list_for_display


HIGH_QUALITY_DEFAULT_THRESHOLD = 75

HIGH_QUALITY_TABLE_COLUMNS = [
    "导出",
    "发布时间",
    "新闻标题",
    "来源",
    "新闻摘要",
    "搜索相关性",
    "新闻质量",
    "推荐分",
    "质量评级",
    "研究主题",
    "涉及企业",
    "地区",
    "量化案例",
    "证据原文",
    "指标标签",
    "原文链接",
    "编辑备注",
]


def _score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _display_list(value: Any, *, numbered: bool = False) -> str:
    return format_json_list_for_display(value, numbered=numbered)


def build_high_quality_results(
    summary: dict[str, Any],
    min_quality: int = HIGH_QUALITY_DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """从一次运行摘要中提取达到质量门槛的新闻，并按推荐分排序。"""
    rows: list[dict[str, Any]] = []
    threshold = _score(min_quality)
    cases = summary.get("cases", [])
    if not isinstance(cases, list):
        cases = []

    for case in cases:
        if not isinstance(case, dict):
            continue
        quality_details = case.get("quality_details")
        if not isinstance(quality_details, dict):
            quality_details = {}
        quality_score = _score(
            case.get("quality_score", quality_details.get("adjusted_score"))
        )
        if quality_score < threshold:
            continue
        relevance_score = _score(case.get("relevance_score"))
        recommendation_value = quality_details.get("recommendation_score")
        if isinstance(recommendation_value, bool) or not isinstance(
            recommendation_value, (int, float)
        ):
            recommendation_score = calculate_recommendation_score(
                relevance_score,
                quality_score,
            )
        else:
            recommendation_score = _score(recommendation_value)

        rows.append(
            {
                "导出": True,
                "发布时间": str(case.get("published_at") or ""),
                "新闻标题": str(case.get("title") or ""),
                "来源": str(case.get("source") or ""),
                "新闻摘要": str(case.get("summary") or ""),
                "搜索相关性": relevance_score,
                "新闻质量": quality_score,
                "推荐分": recommendation_score,
                "质量评级": str(quality_details.get("label") or ""),
                "研究主题": str(case.get("topic_name") or ""),
                "涉及企业": _display_list(case.get("involved_companies")),
                "地区": _display_list(case.get("regions")),
                "量化案例": _display_list(
                    case.get("bullet_points"), numbered=True
                ),
                "证据原文": _display_list(
                    case.get("evidence_quotes"), numbered=True
                ),
                "指标标签": _display_list(case.get("metric_tags")),
                "原文链接": str(case.get("url") or ""),
                "编辑备注": "",
            }
        )

    if not rows:
        return pd.DataFrame(columns=HIGH_QUALITY_TABLE_COLUMNS)
    return (
        pd.DataFrame(rows, columns=HIGH_QUALITY_TABLE_COLUMNS)
        .sort_values(
            ["推荐分", "新闻质量", "搜索相关性"],
            ascending=[False, False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def selected_export_rows(edited_df: pd.DataFrame) -> pd.DataFrame:
    """仅导出用户勾选的行，并移除页面专用选择列。"""
    if edited_df.empty:
        return edited_df.drop(columns=["导出"], errors="ignore").copy()
    if "导出" not in edited_df.columns:
        return edited_df.copy()
    selected = edited_df[edited_df["导出"].fillna(False).astype(bool)]
    return selected.drop(columns=["导出"], errors="ignore").reset_index(drop=True)


def csv_bytes(df: pd.DataFrame) -> bytes:
    """生成 Excel 友好的 UTF-8 BOM CSV。"""
    return df.to_csv(index=False).encode("utf-8-sig")
