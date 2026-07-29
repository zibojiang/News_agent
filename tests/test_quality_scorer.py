
import unittest
import sys
sys.path.insert(0, "/Users/mr.jiang/Desktop/新闻数据爬取网页")
from quality_scorer import (
    compute_source_credibility,
    compute_keyword_relevance,
    compute_content_density,
    compute_data_richness,
    compute_freshness,
    compute_penalties,
    compute_quality_label,
    score_article_pre_ai,
    enrich_with_ai_result,
    QualitySummary,
    QualityPenalty,
    NEWS_QUALITY_RULE_VERSION,
)
from datetime import datetime, timezone, timedelta


class TestSourceCredibility(unittest.TestCase):
    def test_tier1_source(self):
        score, reason = compute_source_credibility("财联社", "https://www.cls.cn/detail/123")
        self.assertGreaterEqual(score, 20)
        self.assertIn("财联社", reason)

    def test_english_authoritative_source(self):
        score, reason = compute_source_credibility(
            "Reuters", "https://www.reuters.com/world/example"
        )
        self.assertEqual(score, 25)
        self.assertIn("Reuters", reason)

    def test_research_institution_official_report(self):
        score, reason = compute_source_credibility(
            "Deloitte",
            "https://www.deloitte.com/global/en/issues/generative-ai/report.html",
        )
        self.assertEqual(score, 24)
        self.assertIn("权威研究/咨询机构", reason)

    def test_known_official_domain_works_when_source_name_is_missing(self):
        score, reason = compute_source_credibility(
            "未知来源",
            "https://insights.deloitte.com/report/ai",
        )
        self.assertEqual(score, 24)
        self.assertIn("deloitte.com", reason)

    def test_lookalike_domain_does_not_gain_official_score(self):
        score, reason = compute_source_credibility(
            "未知来源",
            "https://fake-deloitte.com/report/ai",
        )
        self.assertEqual(score, 10)
        self.assertIn("中性基础分", reason)

    def test_unknown_source(self):
        score, reason = compute_source_credibility("未知小站", "https://example.com")
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 25)
        self.assertIn("中性基础分", reason)

    def test_gov_domain(self):
        score, reason = compute_source_credibility("某政务网", "https://www.gov.cn/xxx")
        self.assertIn("政府域名", reason)

    def test_blog_penalty(self):
        score, reason = compute_source_credibility("某人博客", "https://example.blog.com")
        self.assertIn("自媒体域名", reason)

    def test_score_range(self):
        for s in ["新华社", "财联社", "", "unknown"]:
            score, _ = compute_source_credibility(s, "")
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 25)


class TestKeywordRelevance(unittest.TestCase):
    def test_exact_match(self):
        score, reason = compute_keyword_relevance("酒店行业并购报告", "酒店行业并购")
        self.assertEqual(score, 30)
        self.assertIn("完整出现在标题", reason)

    def test_partial_match(self):
        score, reason = compute_keyword_relevance("万达酒店发展拟收购", "酒店行业并购")
        self.assertGreaterEqual(score, 6)

    def test_no_match(self):
        score, reason = compute_keyword_relevance("财经早报：A股三大指数收跌", "酒店行业并购")
        self.assertLessEqual(score, 12)

    def test_empty_keyword(self):
        score, reason = compute_keyword_relevance("标题", "")
        self.assertEqual(score, 3)


class TestContentDensity(unittest.TestCase):
    def test_rich_content(self):
        content = "这是一篇很长的文章。" * 150  # ~1200 chars
        score, _ = compute_content_density(content)
        self.assertGreaterEqual(score, 6)

    def test_short_content_cutoff(self):
        content = "太短了"
        score, _ = compute_content_density(content)
        self.assertEqual(score, 0)
        self.assertIn("硬跳过", _)


class TestDataRichness(unittest.TestCase):
    def test_data_dense(self):
        content = "营收45.6亿元。同比增长23%。客流量突破850万人次。" * 30
        score, _ = compute_data_richness(content)
        self.assertGreaterEqual(score, 5)

    def test_no_data(self):
        content = "没有数字的文章内容。" * 30
        score, _ = compute_data_richness(content)
        self.assertLessEqual(score, 5)


class TestFreshness(unittest.TestCase):
    def test_today(self):
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        score, _ = compute_freshness(dt)
        self.assertGreaterEqual(score, 8)

    def test_old_article(self):
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S%z")
        score, _ = compute_freshness(old)
        self.assertLessEqual(score, 4)

    def test_invalid_date(self):
        score, _ = compute_freshness("这是一段无效日期")
        self.assertEqual(score, 3)


class TestPenalties(unittest.TestCase):
    def test_sensational_title(self):
        title = "震惊！这简直炸裂了！"
        penalties = compute_penalties(title, "这是一篇正常的新闻文章内容。包含了足够多的文字用于测试用途。文章的主题是关于酒店行业的季度业绩报告分析。内容详实丰富，数据全面准确，能够为读者提供有价值的信息参考。本文共计超过一百个字符以满足测试要求标准。", ai_result=None)
        self.assertGreaterEqual(len(penalties), 1)

    def test_clean_title(self):
        title = "酒店行业Q2业绩报告"
        penalties = compute_penalties(title, "这是一篇正常的新闻文章内容。包含了足够多的文字用于测试用途。文章的主题是关于酒店行业的季度业绩报告分析。内容详实丰富，数据全面准确，能够为读者提供有价值的信息参考。本文共计超过一百个字符以满足测试要求标准。", ai_result=None)
        self.assertEqual(len(penalties), 0)

    def test_ai_unsupported_claims(self):
        ai = {"unsupportedClaims": [{"claim": "X"}, {"claim": "Y"}], "headlineBodyConsistency": 1.0}
        penalties = compute_penalties("标题", "内容", ai_result=ai)
        self.assertGreaterEqual(len(penalties), 1)


class TestQualityLabel(unittest.TestCase):
    def test_excellent(self):
        label, desc = compute_quality_label(92)
        self.assertEqual(label, "优秀")

    def test_very_low(self):
        label, desc = compute_quality_label(25)
        self.assertEqual(label, "较低")


class TestPreAI(unittest.TestCase):
    def test_full_scoring(self):
        summary = score_article_pre_ai(
            title="万达酒店Q2营收增长23%",
            url="https://www.cls.cn/detail/123",
            content="详细财报内容。" * 100,
            source="财联社",
            published_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
            keyword="酒店",
            content_hash="abc123",
        )
        self.assertIsInstance(summary, QualitySummary)
        self.assertGreater(summary.total_score, 0)
        self.assertLessEqual(summary.total_score, 100)
        self.assertIn("source_credibility", summary.dimension_scores)
        self.assertIn("keyword_relevance", summary.dimension_scores)
        self.assertIn("content_density", summary.dimension_scores)
        self.assertIn("data_richness", summary.dimension_scores)
        self.assertIn("freshness", summary.dimension_scores)

    def test_adjusted_score_non_negative(self):
        summary = QualitySummary(
            total_score=20,
            penalties=[QualityPenalty(reason="test", deduction=30)],
        )
        self.assertEqual(summary.adjusted_score, 0)


class TestAIEnrichment(unittest.TestCase):
    def test_enrich_adds_dimensions(self):
        summary = QualitySummary(total_score=50)
        ai = {
            "headlineBodyConsistency": 0.85,
            "originalReportingSignals": ["含独家采访"],
            "namedSourceCount": 3,
            "hasBackgroundContext": True,
            "primaryDocumentCount": 1,
            "containsCounterpartyResponse": True,
            "containsDirectQuotes": True,
            "articleType": "factual",
        }
        enriched = enrich_with_ai_result(summary, ai)
        self.assertIn("headline_body_consistency", enriched.dimension_scores)
        self.assertIn("originality", enriched.dimension_scores)
        self.assertIn("completeness", enriched.dimension_scores)
        self.assertIn("transparency", enriched.dimension_scores)

    def test_enrich_preserves_existing(self):
        summary = QualitySummary(total_score=50, dimension_scores={"source_credibility": 20})
        enriched = enrich_with_ai_result(summary, {})
        self.assertEqual(enriched.dimension_scores["source_credibility"], 20)

    def test_enrich_none_safe(self):
        summary = QualitySummary()
        enriched = enrich_with_ai_result(summary, None)
        self.assertEqual(summary.total_score, enriched.total_score)


if __name__ == "__main__":
    unittest.main()
