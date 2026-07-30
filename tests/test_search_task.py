from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from search_task import get_background_search, start_background_search


class SearchTaskTestCase(unittest.TestCase):
    def _wait(self, task_id: str) -> dict:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            snapshot = get_background_search(task_id)
            if snapshot and snapshot["status"] in {"completed", "failed"}:
                return snapshot
            time.sleep(0.01)
        self.fail("后台搜索测试超时")

    def test_background_search_completes_and_ranks_results(self) -> None:
        articles = [
            {
                "title": "测试新闻",
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
            patch("search_task.agent.fetch_and_pre_score", return_value=articles),
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
                }
            )
            snapshot = self._wait(task_id)

        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["progress"], 1.0)
        self.assertEqual(snapshot["articles"][0]["title"], "测试新闻")
        self.assertEqual(snapshot["results"][0]["recommendation_score"], 86)

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
                }
            )
            snapshot = self._wait(task_id)

        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("RuntimeError", snapshot["error"])


if __name__ == "__main__":
    unittest.main()
