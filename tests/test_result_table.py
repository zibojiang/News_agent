from __future__ import annotations

import unittest

from result_table import (
    build_high_quality_results,
    csv_bytes,
    selected_export_rows,
)


class ResultTableTestCase(unittest.TestCase):
    def _case(
        self,
        title: str,
        quality_score: int,
        relevance_score: int,
        recommendation_score: int | None = None,
    ) -> dict:
        quality_details = {
            "adjusted_score": quality_score,
            "label": "高质量" if quality_score >= 85 else "质量良好",
        }
        if recommendation_score is not None:
            quality_details["recommendation_score"] = recommendation_score
        return {
            "published_at": "2026-07-30 08:00:00",
            "title": title,
            "url": f"https://example.com/{title}",
            "source": "测试媒体",
            "summary": f"{title}摘要",
            "bullet_points": ["营收同比增长 20%"],
            "evidence_quotes": ["营收同比增长 20%"],
            "involved_companies": ["测试企业"],
            "regions": ["中国"],
            "metric_tags": ["营收"],
            "topic_name": "AI产业",
            "relevance_score": relevance_score,
            "quality_score": quality_score,
            "quality_details": quality_details,
        }

    def test_filters_by_quality_and_sorts_by_recommendation(self) -> None:
        summary = {
            "cases": [
                self._case("质量不足", 74, 99),
                self._case("质量较高", 90, 80, 87),
                self._case("推荐优先", 80, 95, 91),
            ]
        }

        result = build_high_quality_results(summary, min_quality=75)

        self.assertEqual(result["新闻标题"].tolist(), ["推荐优先", "质量较高"])
        self.assertEqual(result["新闻质量"].tolist(), [80, 90])
        self.assertTrue(result["导出"].all())
        self.assertEqual(result.iloc[0]["量化案例"], "1. 营收同比增长 20%")

    def test_selected_rows_use_edited_values_for_export(self) -> None:
        table = build_high_quality_results(
            {
                "cases": [
                    self._case("保留", 85, 90),
                    self._case("取消", 86, 88),
                ]
            }
        )
        table.loc[0, "新闻摘要"] = "人工修改后的摘要"
        table.loc[1, "导出"] = False

        selected = selected_export_rows(table)

        self.assertNotIn("导出", selected.columns)
        self.assertEqual(selected["新闻标题"].tolist(), ["保留"])
        self.assertEqual(selected.iloc[0]["新闻摘要"], "人工修改后的摘要")

    def test_csv_export_is_excel_friendly_utf8(self) -> None:
        table = build_high_quality_results(
            {"cases": [self._case("中文新闻", 85, 90)]}
        )
        exported = csv_bytes(selected_export_rows(table))

        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        self.assertIn("中文新闻", exported.decode("utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
