from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import (
    ArticleAnalysisError,
    BodyQualitySchema,
    NewsCaseSchema,
    _keep_verifiable_evidence,
    analyze_article,
    analyze_article_with_chat_completions,
    get_ai_analysis_workers,
    get_ai_provider,
    get_gemini_model,
    get_openai_model,
    run_pipeline,
)


def _body_quality() -> BodyQualitySchema:
    return BodyQualitySchema(
        evidence_score=12,
        evidence_reason="关键数据在正文中有明确支撑",
        completeness_score=8,
        completeness_reason="主体和事件背景较完整",
        transparency_score=8,
        transparency_reason="引用来源和数据口径明确",
        headline_body_consistency_score=5,
        headline_body_consistency_reason="标题准确反映正文",
        balance_score=4,
        balance_reason="表述客观并说明局限",
        clarity_score=5,
        clarity_reason="结构清晰，逻辑连贯",
        has_serious_unsupported_claims=False,
        unsupported_claims_reason="",
    )


class AgentEvidenceTestCase(unittest.TestCase):
    def test_ai_settings_are_read_dynamically(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AI_PROVIDER": "gemini",
                "GEMINI_MODEL": "gemini-test-model",
                "OPENAI_MODEL": "openai-test-model",
            },
        ):
            self.assertEqual(get_ai_provider(), "gemini")
            self.assertEqual(get_gemini_model(), "gemini-test-model")
            self.assertEqual(get_openai_model(), "openai-test-model")

    def test_ai_worker_setting_is_bounded(self) -> None:
        with patch.dict(os.environ, {"AI_ANALYSIS_WORKERS": "20"}):
            self.assertEqual(get_ai_analysis_workers(), 6)
        with patch.dict(os.environ, {"AI_ANALYSIS_WORKERS": "invalid"}):
            self.assertEqual(get_ai_analysis_workers(), 3)

    def test_deepseek_disables_thinking_for_structured_extraction(self) -> None:
        analysis = NewsCaseSchema(
            title="测试新闻",
            url="https://example.com/news/1",
            summary="测试摘要",
            bullet_points=["营收同比增长 20%"],
            evidence_quotes=["营收同比增长 20%"],
            involved_companies=["测试企业"],
            regions=["中国"],
            metric_tags=["营收"],
            relevance_score=85,
            source_credibility_score=20,
            source_credibility_reason="可验证的企业官网",
            body_quality=_body_quality(),
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=analysis.model_dump_json())
                )
            ]
        )
        with (
            patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "deepseek",
                    "DEEPSEEK_THINKING": "disabled",
                },
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            analyze_article_with_chat_completions(
                article_title=analysis.title,
                article_url=analysis.url,
                article_text="测试企业营收同比增长 20%。",
                industry_keyword="酒店营收",
            )

        request = get_client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "disabled"}}
        )

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
            source_credibility_score=10,
            source_credibility_reason="来源信息有限",
            body_quality=_body_quality(),
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
            source_credibility_score=18,
            source_credibility_reason="可验证的行业媒体",
            body_quality=_body_quality(),
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
        self.assertEqual(summary["details"][0]["summary"], "测试摘要")
        self.assertIn("dimension_scores", summary["details"][0]["quality_details"])
        self.assertEqual(
            summary["details"][0]["quality_details"]["body_quality_score"],
            42,
        )

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

    def test_pipeline_accepts_pre_fetched_articles(self) -> None:
        with (
            patch("agent.fetch_and_extract_batch") as fetch_batch,
            patch("agent.record_task_run", return_value=1),
        ):
            summary = run_pipeline(
                "酒店营收",
                pre_fetched_articles=[],
            )

        fetch_batch.assert_not_called()
        self.assertEqual(summary["processed"], 0)

    def test_parallel_analysis_preserves_article_order(self) -> None:
        articles = [
            {
                "title": title,
                "url": f"https://example.com/news/{index}",
                "content": f"{title}营收同比增长 20%。",
                "content_hash": f"hash-{index}",
            }
            for index, title in enumerate(["第一篇", "第二篇"], start=1)
        ]

        def analyze_side_effect(**kwargs: object) -> NewsCaseSchema:
            title = str(kwargs["article_title"])
            if title == "第一篇":
                time.sleep(0.02)
            return NewsCaseSchema(
                title=title,
                url=str(kwargs["article_url"]),
                summary="测试摘要",
                bullet_points=["营收同比增长 20%"],
                evidence_quotes=["营收同比增长 20%"],
                involved_companies=[],
                regions=[],
                metric_tags=["营收"],
                relevance_score=85,
                source_credibility_score=20,
                source_credibility_reason="可验证的主流来源",
                body_quality=_body_quality(),
            )

        write_summary = {
            "news_inserted": 2,
            "qualified_inserted": 2,
            "duplicates": 0,
            "write_failed": 0,
            "items": [
                {"storage_status": "inserted", "reason": ""},
                {"storage_status": "inserted", "reason": ""},
            ],
        }
        progress_messages: list[str] = []
        with (
            patch.dict(os.environ, {"AI_ANALYSIS_WORKERS": "2"}),
            patch("agent.analyze_article", side_effect=analyze_side_effect),
            patch(
                "agent.append_cases_batch_with_summary",
                return_value=write_summary,
            ),
            patch("agent.record_task_run", return_value=1),
        ):
            summary = run_pipeline(
                "酒店营收",
                topic={"topic_id": "S1.1", "topic_name": "测试主题"},
                pre_fetched_articles=articles,
                progress_callback=lambda message, _: progress_messages.append(message),
            )

        self.assertEqual(
            [detail["title"] for detail in summary["details"]],
            ["第一篇", "第二篇"],
        )
        self.assertTrue(any("2 路" in message for message in progress_messages))


if __name__ == "__main__":
    unittest.main()
