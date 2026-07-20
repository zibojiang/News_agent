from __future__ import annotations

import unittest
from unittest.mock import patch

from agent import (
    ArticleAnalysisError,
    NewsCaseSchema,
    _keep_verifiable_evidence,
    analyze_article,
    run_pipeline,
)


class AgentEvidenceTestCase(unittest.TestCase):
    def test_routes_analysis_to_openai(self) -> None:
        expected = NewsCaseSchema(
            title="测试新闻",
            url="https://example.com/news/1",
            summary="测试摘要",
            bullet_points=[],
            evidence_quotes=[],
            involved_companies=[],
            regions=[],
            metric_tags=[],
            relevance_score=50,
        )
        with patch("agent.analyze_article_with_openai", return_value=expected) as call:
            result = analyze_article(
                "测试新闻",
                "https://example.com/news/1",
                "测试正文",
                "酒店",
                provider="openai",
            )
        self.assertIs(result, expected)
        call.assert_called_once()

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "AI_PROVIDER"):
            analyze_article(
                "测试新闻",
                "https://example.com/news/1",
                "测试正文",
                "酒店",
                provider="unknown",
            )

    def test_keeps_only_quotes_present_in_article(self) -> None:
        article = "万豪 2025 年营收达 255 亿美元，同比增长 8%。"
        quotes = [
            "营收达 255 亿美元",
            "净利润同比增长 30%",
            "  ",
        ]
        self.assertEqual(
            _keep_verifiable_evidence(quotes, article),
            ["营收达 255 亿美元"],
        )

    def test_pipeline_keeps_valid_low_score_result_in_news_pool(self) -> None:
        article = {
            "title": "测试新闻",
            "url": "https://example.com/news/1",
            "content": "测试企业营收同比增长 20%。",
            "content_hash": "hash-one",
            "source": "测试媒体",
            "published_at": "2026-07-20 10:00:00",
        }
        analysis = NewsCaseSchema(
            title=article["title"],
            url=article["url"],
            summary="测试摘要",
            bullet_points=["测试企业营收同比增长 20%"],
            evidence_quotes=["营收同比增长 20%"],
            involved_companies=["测试企业"],
            regions=["中国"],
            metric_tags=["营收"],
            relevance_score=60,
        )
        write_summary = {
            "news_inserted": 1,
            "qualified_inserted": 0,
            "duplicates": 0,
            "write_failed": 0,
            "items": [
                {
                    "title": article["title"],
                    "url": article["url"],
                    "storage_status": "inserted",
                    "reason": "",
                }
            ],
        }

        with (
            patch("agent.fetch_and_extract_batch", return_value=[article]),
            patch("agent.analyze_article", return_value=analysis),
            patch(
                "agent.append_cases_batch_with_summary",
                return_value=write_summary,
            ),
            patch("agent.record_task_run", return_value=1),
        ):
            summary = run_pipeline("酒店营收", min_score=70, max_articles=1)

        self.assertEqual(summary["analyzed"], 1)
        self.assertEqual(summary["news_saved"], 1)
        self.assertEqual(summary["saved"], 0)
        self.assertEqual(summary["unqualified"], 1)
        self.assertEqual(summary["details"][0]["storage_status"], "已新增")

    def test_pipeline_surfaces_article_analysis_failure(self) -> None:
        article = {
            "title": "测试新闻",
            "url": "https://example.com/news/1",
            "content": "测试正文",
            "content_hash": "hash-one",
        }

        with (
            patch("agent.fetch_and_extract_batch", return_value=[article]),
            patch(
                "agent.analyze_article",
                side_effect=ArticleAnalysisError("OpenAI 配额不足（429）"),
            ),
            patch("agent.append_cases_batch_with_summary") as append_batch,
            patch("agent.record_task_run", return_value=1),
        ):
            summary = run_pipeline("酒店营收", min_score=70, max_articles=1)

        append_batch.assert_not_called()
        self.assertEqual(summary["analyzed"], 0)
        self.assertEqual(summary["analysis_failed"], 1)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("OpenAI 配额不足", summary["errors"][0])


if __name__ == "__main__":
    unittest.main()
