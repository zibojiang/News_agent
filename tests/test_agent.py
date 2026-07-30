from __future__ import annotations

import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import (
    ArticleAnalysisError,
    BodyQualitySchema,
    NewsCaseSchema,
    SEARCH_RELEVANCE_RULE_VERSION,
    SearchIntentSchema,
    SourceCredibilitySchema,
    _keep_verifiable_evidence,
    analyze_article,
    analyze_article_with_chat_completions,
    analyze_search_intent,
    calculate_recommendation_score,
    fallback_search_intent,
    get_ai_analysis_workers,
    get_ai_provider,
    get_gemini_model,
    get_openai_model,
    run_pipeline,
    score_sources_with_ai,
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

    def test_search_intent_generates_chinese_and_english_queries(self) -> None:
        intent = SearchIntentSchema(
            intent_summary="了解数字集成电路产业的新技术和产业化方向",
            target_topics=["数字集成电路", "技术趋势", "产业化"],
            chinese_queries=["数字集成电路 产业 新方向"],
            english_queries=["digital integrated circuit industry trends"],
            relevance_criteria=["新闻实质讨论新技术或产业化方向"],
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=intent.model_dump_json())
                )
            ]
        )
        with (
            patch.dict(
                os.environ,
                {"AI_PROVIDER": "deepseek", "DEEPSEEK_THINKING": "disabled"},
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            result = analyze_search_intent(
                "数字集成电路产业有哪些新方向"
            )

        self.assertEqual(result.chinese_queries, intent.chinese_queries)
        self.assertEqual(result.english_queries, intent.english_queries)
        request = get_client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", request["messages"][0]["content"])

    def test_search_intent_normalizes_compatible_api_field_types(self) -> None:
        malformed_intent = {
            "intent_summary": "对比美国和中国 AI 产业的发展",
            "target_topics": "美国AI产业与中国AI产业",
            "chinese_queries": "美国AI产业 中国AI产业 对比",
            "english_queries": "US China AI industry comparison",
            "relevance_criteria": "新闻实质比较两国 AI 产业",
            "scope_level": "目标较明确",
            "needs_clarification": "false",
            "clarification_question": None,
            "interpretations": [],
            "recommended_interpretation_index": None,
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(malformed_intent, ensure_ascii=False)
                    )
                )
            ]
        )
        with (
            patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "deepseek",
                    "OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            result = analyze_search_intent("美国AI产业与中国AI产业对比")

        self.assertEqual(result.scope_level, "specific")
        self.assertFalse(result.needs_clarification)
        self.assertEqual(result.target_topics, ["美国AI产业与中国AI产业"])
        self.assertEqual(
            result.english_queries,
            ["US China AI industry comparison"],
        )

    def test_search_intent_retries_with_format_feedback(self) -> None:
        invalid_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"message":"missing fields"}')
                )
            ]
        )
        valid_intent = SearchIntentSchema(
            intent_summary="对比美国和中国 AI 产业的发展",
            target_topics=["美国AI产业", "中国AI产业"],
            chinese_queries=["美国 中国 AI 产业 对比"],
            english_queries=["US China AI industry comparison"],
            relevance_criteria=["新闻实质比较两国 AI 产业"],
        )
        valid_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=valid_intent.model_dump_json())
                )
            ]
        )
        with (
            patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "deepseek",
                    "OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
            patch("agent.time.sleep"),
        ):
            get_client.return_value.chat.completions.create.side_effect = [
                invalid_response,
                valid_response,
            ]
            result = analyze_search_intent("美国AI产业与中国AI产业对比")

        self.assertEqual(result.intent_summary, valid_intent.intent_summary)
        calls = get_client.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("格式修正", calls[1].kwargs["messages"][1]["content"])

    def test_search_intent_fallback_does_not_fake_english_translation(self) -> None:
        result = fallback_search_intent("海外产品+中国供应链研发")

        self.assertEqual(result.target_topics, ["海外产品", "中国供应链研发"])
        self.assertEqual(result.chinese_queries, ["海外产品 中国供应链研发"])
        self.assertEqual(result.english_queries, [])
        self.assertTrue(result.needs_clarification)
        self.assertEqual(result.scope_level, "focused")
        self.assertIn("海外产品、中国供应链研发", result.relevance_criteria[0])

    def test_broad_ai_fallback_offers_concrete_interpretations(self) -> None:
        result = fallback_search_intent("AI")

        self.assertEqual(result.scope_level, "broad")
        self.assertTrue(result.needs_clarification)
        self.assertGreaterEqual(len(result.interpretations), 3)
        self.assertEqual(result.interpretations[1].label, "AI 应用与商业化")
        self.assertTrue(result.interpretations[1].english_queries)

    def test_custom_openai_endpoint_uses_chat_completions_for_intent(self) -> None:
        intent = SearchIntentSchema(
            intent_summary="了解海外产品与中国供应链研发的协同",
            target_topics=["overseas products", "China supply chain R&D"],
            chinese_queries=["海外产品 中国供应链 研发"],
            english_queries=["overseas products China supply chain R&D"],
            relevance_criteria=["新闻同时讨论海外产品和中国研发供应链"],
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=intent.model_dump_json())
                )
            ]
        )
        with (
            patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "openai",
                    "OPENAI_BASE_URL": "https://example-compatible-api.test/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            result = analyze_search_intent("海外产品+中国供应链研发")

        self.assertEqual(result.english_queries, intent.english_queries)
        get_client.return_value.responses.parse.assert_not_called()
        request = get_client.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", request)

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
                    "OPENAI_BASE_URL": "https://api.deepseek.com",
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

    def test_siliconflow_does_not_receive_deepseek_official_options(self) -> None:
        analysis = NewsCaseSchema(
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
                    "OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            analyze_article_with_chat_completions(
                article_title=analysis.title,
                article_url=analysis.url,
                article_text="测试正文。",
                industry_keyword="AI产业",
            )

        request = get_client.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("extra_body", request)

    def test_article_analysis_normalizes_compatible_api_field_types(self) -> None:
        malformed_analysis = {
            "title": "模型返回的标题",
            "url": "https://wrong.example.com",
            "summary": "文章介绍中国品牌通过供应链和本地化渠道拓展海外市场。",
            "bulletPoints": "海外门店数量增长 20%",
            "evidenceQuotes": "海外门店数量增长 20%",
            "involvedCompanies": "测试企业",
            "regions": "东南亚",
            "metricTags": "门店数量",
            "relevanceScore": "88",
            "sourceCredibilityScore": "21",
            "sourceCredibilityReason": "具有明确编辑责任的主流媒体",
            "bodyQuality": {
                "evidenceScore": "12",
                "completenessScore": "8",
                "transparencyScore": "7",
                "headlineBodyConsistencyScore": "5",
                "balanceScore": "4",
                "clarityScore": "5",
                "hasSeriousUnsupportedClaims": "false",
            },
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(malformed_analysis, ensure_ascii=False)
                    )
                )
            ]
        )
        article_text = "测试企业海外门店数量增长 20%，并持续建设本地供应链。"
        with (
            patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "deepseek",
                    "OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
        ):
            get_client.return_value.chat.completions.create.return_value = response
            result = analyze_article_with_chat_completions(
                article_title="国货品牌出海",
                article_url="https://example.com/news/export",
                article_text=article_text,
                industry_keyword="中国品牌出海",
            )

        self.assertEqual(result.title, "国货品牌出海")
        self.assertEqual(result.url, "https://example.com/news/export")
        self.assertEqual(result.bullet_points, ["海外门店数量增长 20%"])
        self.assertEqual(result.evidence_quotes, ["海外门店数量增长 20%"])
        self.assertEqual(result.relevance_score, 88)
        self.assertEqual(result.body_quality.evidence_score, 12)
        self.assertEqual(result.body_quality.evidence_reason, "AI 未提供具体依据")
        request = get_client.return_value.chat.completions.create.call_args.kwargs
        self.assertIn("JSON Schema", request["messages"][0]["content"])

    def test_article_analysis_retries_with_schema_feedback(self) -> None:
        invalid_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"summary":"缺少正文质量"}')
                )
            ]
        )
        analysis = NewsCaseSchema(
            title="测试新闻",
            url="https://example.com/news/retry",
            summary="第二次返回完整的正文分析结果。",
            bullet_points=[],
            evidence_quotes=[],
            involved_companies=["测试企业"],
            regions=["中国"],
            metric_tags=[],
            relevance_score=75,
            source_credibility_score=20,
            source_credibility_reason="可验证的主流媒体",
            body_quality=_body_quality(),
        )
        valid_response = SimpleNamespace(
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
                    "OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
                },
            ),
            patch("agent._get_openai_client") as get_client,
            patch("agent.time.sleep"),
        ):
            get_client.return_value.chat.completions.create.side_effect = [
                invalid_response,
                valid_response,
            ]
            result = analyze_article_with_chat_completions(
                article_title=analysis.title,
                article_url=analysis.url,
                article_text="测试企业正在推进中国品牌出海。",
                industry_keyword="中国品牌出海",
            )

        self.assertEqual(result.summary, analysis.summary)
        calls = get_client.return_value.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIn("格式修正", calls[1].kwargs["messages"][1]["content"])
        self.assertIn("body_quality", calls[1].kwargs["messages"][1]["content"])

    def test_source_ai_score_is_applied_before_body_analysis(self) -> None:
        article = {
            "title": "测试新闻",
            "url": "https://example.com/news/1",
            "content": "测试正文",
            "source": "测试媒体",
            "published_at": "2026-07-29 10:00:00",
        }
        with patch(
            "agent.analyze_source_credibility",
            return_value=SourceCredibilitySchema(
                score=19,
                reason="具有明确编辑责任的行业媒体",
                core_topic_match_score=35,
                information_need_match_score=26,
                semantic_coverage_score=17,
                directness_score=8,
                relevance_reason="新闻核心事件与搜索主题直接相关",
            ),
        ):
            errors = score_sources_with_ai([article], original_query="测试主题")

        self.assertEqual(errors, [])
        quality = article["quality_pre"]
        self.assertEqual(quality.dimension_scores["source_credibility"], 19)
        self.assertEqual(quality.source_score_method, "ai")
        self.assertIn("AI 评估", quality.dimension_reasons["source_credibility"])
        self.assertEqual(article["search_relevance_score"], 86)
        self.assertEqual(
            article["search_relevance_dimensions"],
            {
                "core_topic_match": 35,
                "information_need_match": 26,
                "semantic_coverage": 17,
                "directness": 8,
            },
        )
        self.assertEqual(
            article["search_relevance_rule_version"],
            SEARCH_RELEVANCE_RULE_VERSION,
        )
        self.assertTrue(article["search_relevance_scored"])

    def test_recommendation_score_prioritizes_search_relevance(self) -> None:
        self.assertEqual(calculate_recommendation_score(88, 85), 87)
        self.assertEqual(calculate_recommendation_score(100, 0), 70)
        self.assertEqual(calculate_recommendation_score(0, 100), 30)

    def test_pipeline_skips_body_ai_when_search_relevance_is_low(self) -> None:
        article = {
            "title": "间接相关新闻",
            "url": "https://example.com/news/low-relevance",
            "content": "这是一篇与用户问题只有间接关系的新闻正文。" * 20,
            "content_hash": "hash-low-relevance",
            "source": "测试媒体",
            "published_at": "2026-07-20 10:00:00",
            "search_relevance_score": 55,
            "search_relevance_reason": "仅提及外围主题",
            "search_relevance_scored": True,
        }

        with (
            patch("agent.score_sources_with_ai") as score_sources,
            patch("agent.analyze_article") as analyze,
            patch("agent.append_cases_batch_with_summary") as append_batch,
            patch("agent.record_task_run", return_value=1),
        ):
            summary = run_pipeline(
                "数字集成电路产业新方向",
                pre_fetched_articles=[article],
                pre_screen_completed=True,
            )

        score_sources.assert_not_called()
        analyze.assert_not_called()
        append_batch.assert_not_called()
        self.assertEqual(summary["relevance_skipped"], 1)
        self.assertEqual(summary["analyzed"], 0)
        self.assertEqual(summary["details"][0]["analysis_status"], "跳过")

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
        with (
            patch.dict(os.environ, {"OPENAI_BASE_URL": ""}),
            patch("agent.analyze_article_with_openai", return_value=expected) as call,
        ):
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
            patch("agent.score_sources_with_ai", return_value=[]),
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
            patch("agent.score_sources_with_ai", return_value=[]),
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
            patch("agent.score_sources_with_ai", return_value=[]),
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
