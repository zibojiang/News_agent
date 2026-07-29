from __future__ import annotations

import unittest
from concurrent.futures import Future
from unittest.mock import patch

from scraper import fetch_and_extract_batch


class ScraperBatchTestCase(unittest.TestCase):
    def test_resolves_each_candidate_once_inside_parallel_extraction(self) -> None:
        item = {
            "title": "测试新闻",
            "url": "https://news.google.com/articles/test",
            "source": "测试媒体",
            "published_at": "2026-07-29 10:00:00",
        }
        with (
            patch("scraper.fetch_latest_news", return_value=[item]) as fetch_news,
            patch(
                "scraper._resolve_final_url",
                return_value="https://example.com/news/1",
            ) as resolve_url,
            patch(
                "scraper.extract_article_text",
                return_value="有效正文" * 100,
            ) as extract_text,
        ):
            results = fetch_and_extract_batch("测试", max_articles=4, max_workers=1)

        fetch_news.assert_called_once_with("测试", max_articles=8)
        resolve_url.assert_called_once_with(item["url"])
        extract_text.assert_called_once_with(
            "https://example.com/news/1", resolve_url=False
        )
        self.assertEqual(results[0]["url"], "https://example.com/news/1")

    def test_reports_each_found_article_and_stops_waiting_at_limit(self) -> None:
        class RecordingExecutor:
            instance = None

            def __init__(self, max_workers: int) -> None:
                self.shutdown_args = None
                RecordingExecutor.instance = self

            def submit(self, function, item):
                future = Future()
                future.set_result(function(item))
                return future

            def shutdown(self, wait: bool, cancel_futures: bool) -> None:
                self.shutdown_args = (wait, cancel_futures)

        items = [
            {
                "title": f"测试新闻 {index}",
                "url": f"https://example.com/{index}",
                "source": "测试媒体",
                "published_at": "2026-07-29 10:00:00",
            }
            for index in range(2)
        ]
        updates = []
        with (
            patch("scraper.ThreadPoolExecutor", RecordingExecutor),
            patch("scraper.fetch_latest_news", return_value=items),
            patch("scraper._resolve_final_url", side_effect=lambda url: url),
            patch("scraper.extract_article_text", return_value="有效正文" * 100),
        ):
            results = fetch_and_extract_batch(
                "测试",
                max_articles=1,
                article_callback=lambda article, found, total: updates.append(
                    (article["title"], found, total)
                ),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(updates[0][1:], (1, 1))
        self.assertEqual(RecordingExecutor.instance.shutdown_args, (False, True))


if __name__ == "__main__":
    unittest.main()
