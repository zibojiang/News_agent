from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from database import load_last_search_state, save_last_search_state
from quality_scorer import NEWS_QUALITY_RULE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTestCase(unittest.TestCase):
    def test_discards_legacy_cards_in_a_new_streamlit_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "industry_news.db")
            with patch.dict(
                os.environ,
                {
                    "DATABASE_PATH": database_path,
                    "DEPLOYMENT_MODE": "cloud_demo",
                    "ADMIN_PASSWORD": "",
                },
            ):
                save_last_search_state(
                    {
                        "keyword": "复星AI",
                        "articles": [
                            {
                                "title": "刷新后仍然展示的新闻",
                                "url": "https://example.com/news",
                                "source": "Deloitte",
                                "published_at": "2026-07-29 10:00:00",
                                "language": "en",
                                "content": "",
                                "quality_pre": {
                                    "adjusted_score": 80,
                                    "total_score": 80,
                                    "dimension_scores": {"source_credibility": 24},
                                    "dimension_reasons": {},
                                    "penalties": [],
                                    "label": "良好",
                                },
                            }
                        ],
                        "results": {
                            "0": {
                                "analysis_status": "成功",
                                "storage_status": "已新增",
                                "score": 95,
                                "summary": "刷新后恢复的 AI 摘要。",
                                "quality_score": 80,
                                "quality_label": "良好",
                                "quality_details": {},
                            }
                        },
                        "run_summary": {},
                    }
                )

                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                self.assertEqual(list(app.exception), [])
                self.assertFalse(
                    any(
                        "刷新后仍然展示的新闻" in markdown.value
                        for markdown in app.markdown
                    )
                )
                self.assertEqual(load_last_search_state(), {})

    def test_restores_cards_scored_with_current_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "industry_news.db")
            dimensions = {
                "source_credibility": 20,
                "content_density": 8,
                "data_richness": 8,
                "freshness": 4,
            }
            body_dimensions = {
                "evidence_quality": 12,
                "completeness": 8,
                "transparency": 8,
                "headline_body_consistency": 5,
                "balance": 4,
                "clarity": 5,
            }
            with patch.dict(
                os.environ,
                {
                    "DATABASE_PATH": database_path,
                    "DEPLOYMENT_MODE": "cloud_demo",
                    "ADMIN_PASSWORD": "",
                },
            ):
                save_last_search_state(
                    {
                        "keyword": "复星AI",
                        "articles": [
                            {
                                "title": "当前规则评分新闻",
                                "search_relevance_score": 88,
                                "search_relevance_reason": "直接回答搜索问题",
                                "search_relevance_scored": True,
                                "quality_pre": {
                                    "adjusted_score": 40,
                                    "dimension_scores": dimensions,
                                    "dimension_reasons": {},
                                    "penalties": [],
                                    "label": "预筛",
                                    "rule_version": NEWS_QUALITY_RULE_VERSION,
                                    "source_score_method": "ai",
                                },
                            }
                        ],
                        "results": {
                            "0": {
                                "analysis_status": "成功",
                                "storage_status": "已更新",
                                "score": 88,
                                "summary": "当前规则生成的摘要。",
                                "quality_score": 82,
                                "quality_label": "质量良好",
                                "recommendation_score": 86,
                                "quality_details": {
                                    "adjusted_score": 82,
                                    "dimension_scores": {
                                        **dimensions,
                                        **body_dimensions,
                                    },
                                    "dimension_reasons": {},
                                    "penalties": [],
                                    "label": "质量良好",
                                    "rule_version": NEWS_QUALITY_RULE_VERSION,
                                    "source_score_method": "ai",
                                },
                            }
                        },
                        "run_summary": {},
                    }
                )

                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                self.assertEqual(list(app.exception), [])
                self.assertTrue(
                    any(
                        "当前规则评分新闻" in markdown.value
                        for markdown in app.markdown
                    )
                )
                self.assertTrue(
                    any(
                        "当前规则生成的摘要" in markdown.value
                        for markdown in app.markdown
                    )
                )

    def test_base_score_expander_keeps_ai_source_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "industry_news.db")
            with patch.dict(
                os.environ,
                {
                    "DATABASE_PATH": database_path,
                    "DEPLOYMENT_MODE": "cloud_demo",
                    "ADMIN_PASSWORD": "",
                },
            ):
                save_last_search_state(
                    {
                        "keyword": "复星AI",
                        "articles": [
                            {
                                "title": "基础评分细则测试新闻",
                                "search_relevance_score": 76,
                                "search_relevance_reason": "与搜索主题明显相关",
                                "search_relevance_scored": True,
                                "quality_pre": {
                                    "total_score": 41,
                                    "adjusted_score": 41,
                                    "dimension_scores": {
                                        "source_credibility": 21,
                                        "content_density": 8,
                                        "data_richness": 8,
                                        "freshness": 4,
                                    },
                                    "dimension_reasons": {
                                        "source_credibility": "AI 评估：主流媒体",
                                        "content_density": "正文结构完整",
                                        "data_richness": "含多项量化数据",
                                        "freshness": "2-4 个月内发布",
                                    },
                                    "penalties": [],
                                    "label": "预筛",
                                    "rule_version": NEWS_QUALITY_RULE_VERSION,
                                    "source_score_method": "ai",
                                },
                            }
                        ],
                        "results": {
                            "0": {
                                "analysis_status": "失败",
                                "storage_status": "未写入",
                                "score": 76,
                                "reason": "正文 AI 暂时失败",
                            }
                        },
                        "run_summary": {},
                    }
                )

                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                self.assertEqual(list(app.exception), [])
                self.assertTrue(
                    any(
                        "查看基础评分｜41/50" in expander.label
                        for expander in app.expander
                    )
                )
                rendered_markdown = [item.value for item in app.markdown]
                self.assertTrue(
                    any("来源权威度（AI 评分）：21/25 分" in value for value in rendered_markdown)
                )
                self.assertTrue(
                    any("信息密度：8/10 分" in value for value in rendered_markdown)
                )
                self.assertTrue(
                    any("数据含量：8/10 分" in value for value in rendered_markdown)
                )
                self.assertTrue(
                    any("时效性：4/5 分" in value for value in rendered_markdown)
                )

    def test_cloud_demo_is_open_without_admin_password(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = str(Path(temp_dir) / "industry_news.db")
            with patch.dict(
                os.environ,
                {
                    "DATABASE_PATH": database_path,
                    "DEPLOYMENT_MODE": "cloud_demo",
                    "ADMIN_PASSWORD": "",
                },
            ):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=20)
                self.assertEqual(list(app.exception), [])
                button_labels = [button.label for button in app.button]
                self.assertIn("立即搜索", button_labels)
                self.assertIn("⚙️", button_labels)
                self.assertNotIn("解锁管理操作", button_labels)
                self.assertTrue(
                    any("搜索新闻" in markdown.value for markdown in app.markdown)
                )
                self.assertTrue(
                    any("半导体产业研报" in markdown.value for markdown in app.markdown)
                )

                app.session_state["fetched_articles"] = [
                    {
                        "title": "复星 AI 测试新闻",
                        "url": "https://example.com/news",
                        "source": "测试媒体",
                        "published_at": "2026-07-29 10:00:00",
                        "search_relevance_score": 88,
                        "search_relevance_reason": "直接命中搜索主题",
                        "search_relevance_scored": True,
                        "quality_pre": {
                            "adjusted_score": 40,
                            "total_score": 40,
                            "dimension_scores": {
                                "source_credibility": 20,
                                "content_density": 8,
                                "data_richness": 8,
                                "freshness": 4,
                            },
                            "dimension_reasons": {},
                            "penalties": [],
                            "label": "预筛",
                            "rule_version": NEWS_QUALITY_RULE_VERSION,
                            "source_score_method": "ai",
                        },
                    }
                ]
                app.session_state["fetched_keyword"] = "复星AI"
                app.session_state["fetched_results"] = {
                    0: {
                        "analysis_status": "成功",
                        "storage_status": "已新增",
                        "score": 88,
                        "summary": "这是一条用于卡片展示的 AI 新闻摘要。",
                        "quality_score": 85,
                        "quality_label": "高质量",
                        "recommendation_score": 87,
                        "quality_details": {
                            "total_score": 85,
                            "adjusted_score": 85,
                            "dimension_scores": {
                                "source_credibility": 20,
                                "content_density": 8,
                                "data_richness": 8,
                                "freshness": 4,
                                "evidence_quality": 14,
                                "completeness": 9,
                                "transparency": 8,
                                "headline_body_consistency": 5,
                                "balance": 4,
                                "clarity": 5,
                            },
                            "dimension_reasons": {
                                "source_credibility": "主流媒体"
                            },
                            "penalties": [],
                            "label": "高质量",
                            "rule_version": NEWS_QUALITY_RULE_VERSION,
                            "source_score_method": "ai",
                            "score_cap": 100,
                        },
                    }
                }
                app.run(timeout=20)
                self.assertTrue(
                    any("已搜索到的文章（1篇）" in markdown.value for markdown in app.markdown)
                )
                self.assertTrue(
                    any("AI 评估：主流来源" in caption.value for caption in app.caption)
                )
                self.assertTrue(
                    any("AI 新闻摘要" in markdown.value for markdown in app.markdown)
                )
                self.assertTrue(
                    any("查看完整评分" in expander.label for expander in app.expander)
                )
                self.assertTrue(
                    any(
                        "基础分 40/50 + AI 正文质量 45/50 = 综合质量 85/100"
                        in markdown.value
                        for markdown in app.markdown
                    )
                )
                management_button = next(
                    button for button in app.button if button.label == "⚙️"
                )
                management_button.click().run(timeout=20)
                self.assertEqual(list(app.exception), [])
                self.assertTrue(
                    any("管理台" in markdown.value for markdown in app.markdown)
                )


if __name__ == "__main__":
    unittest.main()
