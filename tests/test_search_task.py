from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from search_task import get_background_search, start_background_search
from session_workspace import SessionWorkspace


class SearchTaskTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = SessionWorkspace()
        self.addCleanup(self.workspace.close)

    def _wait(self, task_id: str) -> dict:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = get_background_search(task_id, owner_id=self.workspace.owner_id)
            if snapshot and snapshot["status"] in {"completed", "failed"}:
                return snapshot
            time.sleep(0.01)
        self.fail("后台搜索测试超时")

    def test_background_search_completes_and_ranks_results(self) -> None:
        articles = [
            {
                "title": "English news article",
                "url": "https://example.com/news/1",
                "search_relevance_score": 88,
                "search_relevance_scored": True,
            }
        ]
        run_summary = {
            "details": [
                {
                    "title": "测试新闻",
                    "url": "https://example.com/news/1",
                    "score": 88,
                    "recommendation_score": 86,
                    "analysis_status": "成功",
                }
            ],
            "cases": [],
        }
        with (
            patch("search_task.agent.fetch_and_pre_score", return_value=articles) as fetch,
            patch("search_task.agent.score_sources_with_ai", return_value=[]),
            patch("search_task.agent.run_pipeline", return_value=run_summary),
        ):
            task_id = start_background_search(
                {
                    "primary_query": "中国品牌出海",
                    "confirmed_keyword": "中国品牌出海",
                    "max_articles": 8,
                    "min_score": 70,
                    "search_intent": {},
                    "news_language": "en",
                    "output_language": "en",
                },
                workspace=self.workspace,
            )
            snapshot = self._wait(task_id)

        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["progress"], 1.0)
        self.assertEqual(snapshot["articles"][0]["title"], "English news article")
        self.assertEqual(snapshot["results"][0]["recommendation_score"], 86)
        self.assertEqual(fetch.call_args.kwargs["news_language"], "en")
        self.assertEqual(snapshot["articles"][0]["output_language"], "en")
        self.assertEqual(snapshot["news_language"], "en")

    def test_background_search_keeps_failure_state(self) -> None:
        with patch(
            "search_task.agent.fetch_and_pre_score",
            side_effect=RuntimeError("测试抓取失败"),
        ):
            task_id = start_background_search(
                {
                    "primary_query": "测试",
                    "confirmed_keyword": "测试",
                    "max_articles": 8,
                    "min_score": 70,
                },
                workspace=self.workspace,
            )
            snapshot = self._wait(task_id)

        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("RuntimeError", snapshot["error"])

    def test_english_task_filters_wrong_language_before_ai_even_with_stale_preferences(self) -> None:
        chinese = {"title": "三星发布全新人工智能芯片", "content": "中文原文" * 100, "language": "en"}
        english = {"title": "New AI accelerator announced", "content": "English article body. " * 30, "url": "https://example.com/en"}
        def fetch(**kwargs):
            kwargs["article_callback"](chinese, 1, 2)
            kwargs["article_callback"](english, 2, 2)
            return [chinese, english]
        with (
            patch("search_task.agent.fetch_and_pre_score", side_effect=fetch) as fetch_news,
            patch("search_task.agent.score_sources_with_ai", return_value=[]) as score,
            patch("search_task.agent.run_pipeline", return_value={"details": [], "cases": []}),
        ):
            task_id = start_background_search({
                "primary_query": "AI hardware", "confirmed_keyword": "AI hardware",
                "max_articles": 2, "min_score": 70, "news_language": "both", "output_language": "en",
            }, workspace=self.workspace)
            snapshot = self._wait(task_id)
        self.assertEqual(fetch_news.call_args.kwargs["news_language"], "en")
        self.assertEqual([item["title"] for item in score.call_args.args[0]], [english["title"]])
        self.assertEqual([item["title"] for item in snapshot["articles"]], [english["title"]])

    def test_no_article_bodies_preserves_diagnostics_and_never_calls_scoring_ai(self) -> None:
        def fetch(**kwargs):
            kwargs["diagnostics"].update(candidates=8, extracted=0, unresolved=8, fallback_used=True)
            return []

        with (
            patch("search_task.agent.fetch_and_pre_score", side_effect=fetch),
            patch("search_task.agent.score_sources_with_ai") as source_ai,
            patch("search_task.agent.run_pipeline") as body_ai,
        ):
            task_id = start_background_search({
                "primary_query": "AI hardware accelerator", "confirmed_keyword": "AI hardware accelerator",
                "max_articles": 8, "min_score": 70,
            }, workspace=self.workspace)
            snapshot = self._wait(task_id)
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["search_diagnostics"]["unresolved"], 8)
        self.assertEqual(snapshot["run_summary"]["analyzed"], 0)
        source_ai.assert_not_called()
        body_ai.assert_not_called()


if __name__ == "__main__":
    unittest.main()
