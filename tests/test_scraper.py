from __future__ import annotations

import unittest
from concurrent.futures import Future
from unittest.mock import patch
from urllib.parse import quote_plus

from scraper import _resolve_final_url, extract_article_text, fetch_and_extract_batch, fetch_latest_news


class ScraperBatchTestCase(unittest.TestCase):
    def test_single_language_search_only_requests_that_google_feed(self) -> None:
        for language, marker in (("en", "hl=en-US"), ("zh", "hl=zh-CN")):
            urls = []
            def fetch(url, limit, source_name):
                urls.append(url)
                return [{"title": source_name, "url": "https://example.com/news"}]
            with self.subTest(language=language), patch("scraper._fetch_rss_news", side_effect=fetch):
                articles = fetch_latest_news(
                    "人工智能硬件", english_query="AI hardware accelerator",
                    english_queries=["AI chips"], additional_queries=["算力芯片"],
                    news_language=language,
                )
            self.assertTrue(urls)
            self.assertTrue(all(marker in url for url in urls))
            self.assertTrue(all(article["language"] == language for article in articles))

    def test_bing_fallback_respects_english_choice_even_for_chinese_input(self) -> None:
        urls = []
        def fetch(url, limit, source_name):
            urls.append(url)
            return [] if source_name.startswith("Google") else [
                {"title": "AI chip report", "url": "https://example.com/news"},
            ]
        with patch("scraper._fetch_rss_news", side_effect=fetch):
            results = fetch_latest_news(
                "人工智能芯片", english_queries=["AI hardware accelerator"], news_language="en",
            )
        bing_urls = [url for url in urls if "bing.com" in url]
        self.assertEqual(len(bing_urls), 1)
        self.assertIn("q=AI+hardware+accelerator", bing_urls[0])
        self.assertIn("mkt=en-US", bing_urls[0])
        self.assertEqual(results[0]["language"], "en")

    def test_filters_actual_article_language_even_if_feed_labels_are_wrong(self) -> None:
        items = [
            {"title": "AI accelerator report", "url": "https://example.com/english", "language": "zh"},
            {"title": "人工智能硬件加速器市场报告", "url": "https://example.com/chinese", "language": "en"},
        ]
        def extract(url, **kwargs):
            return ("AI hardware revenue rose by 20 percent. " * 30 if url.endswith("english")
                    else "人工智能硬件加速器市场增长百分之二十。" * 30)
        for language, expected in (("en", ["en"]), ("zh", ["zh"]), ("both", ["en", "zh"])):
            stats = {}
            with (
                self.subTest(language=language),
                patch("scraper.fetch_latest_news", return_value=items),
                patch("scraper._resolve_final_url", side_effect=lambda url: url),
                patch("scraper.extract_article_text", side_effect=extract),
            ):
                results = fetch_and_extract_batch("AI hardware", max_articles=2, news_language=language, diagnostics=stats)
            self.assertEqual(sorted(x["language"] for x in results), expected)
            self.assertEqual(stats["language_mismatch"], 0 if language == "both" else 1)

    def test_accepts_current_and_legacy_google_decoder_success_fields(self) -> None:
        for flag in ("success", "status"):
            with (
                self.subTest(flag=flag),
                patch("scraper.gnewsdecoder", return_value={flag: True, "decoded_url": "https://example.com/story"}),
                patch("scraper.requests.head") as head,
            ):
                self.assertEqual(_resolve_final_url("https://news.google.com/rss/articles/id"), "https://example.com/story")
                head.assert_not_called()

    def test_rejects_unresolved_google_landing_page_without_fetching_it(self) -> None:
        with (
            patch("scraper.gnewsdecoder", return_value={"success": False, "message": "Unavailable"}),
            patch("scraper.requests.head") as head,
            patch("scraper._safe_get") as get,
        ):
            self.assertEqual(extract_article_text("https://news.google.com/rss/articles/id"), "")
        head.assert_not_called()
        get.assert_not_called()

    def test_bing_click_link_resolves_directly_to_publisher(self) -> None:
        publisher = "https://example.com/story?item=1&lang=en"
        with patch("scraper.requests.head") as head:
            result = _resolve_final_url("http://www.bing.com/news/apiclick.aspx?url=" + quote_plus(publisher))
        self.assertEqual(result, publisher)
        head.assert_not_called()

    def test_uses_bing_when_google_links_have_no_readable_bodies(self) -> None:
        google = {"title": "AI hardware", "url": "https://news.google.com/rss/articles/id"}
        bing = {"title": "AI hardware", "url": "https://example.com/news"}
        stats = {}
        with (
            patch("scraper.fetch_latest_news", return_value=[google]),
            patch("scraper.gnewsdecoder", return_value={"success": False}),
            patch("scraper._fetch_bing_news", return_value=[bing]) as fallback,
            patch("scraper.requests.head", side_effect=RuntimeError("publisher error")),
        ):
            # A broken article is counted, not allowed to abort the entire batch.
            self.assertEqual(fetch_and_extract_batch("AI hardware", diagnostics=stats), [])
        fallback.assert_called_once()
        self.assertEqual(stats["candidates"], 2)
        self.assertEqual(stats["unresolved"], 1)
        self.assertEqual(stats["errors"], 1)
        self.assertTrue(stats["fallback_used"])

    def test_bing_can_recover_the_same_headline_using_a_direct_link(self) -> None:
        items = [{"title": "AI hardware", "url": "https://news.google.com/rss/articles/id"}]
        fallback_items = [{"title": "AI hardware", "url": "https://example.com/news"}]
        stats = {}
        with (
            patch("scraper.fetch_latest_news", return_value=items),
            patch("scraper._resolve_final_url", side_effect=lambda url: url),
            patch("scraper._fetch_bing_news", return_value=fallback_items),
            patch("scraper.extract_article_text", return_value="Article text. " * 30),
        ):
            results = fetch_and_extract_batch("AI hardware", max_articles=1, diagnostics=stats)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], fallback_items[0]["url"])
        self.assertEqual(stats["unresolved"], 1)
        self.assertEqual(stats["extracted"], 1)

    def test_does_not_repeat_bing_fallback_and_reports_empty_or_short_text(self) -> None:
        items = [
            {"title": "Unavailable", "url": "https://example.com/empty", "feed_provider": "bing"},
            {"title": "Short", "url": "https://example.com/short", "feed_provider": "bing"},
        ]
        stats = {}
        with (
            patch("scraper.fetch_latest_news", return_value=items),
            patch("scraper._resolve_final_url", side_effect=lambda url: url),
            patch("scraper._fetch_bing_news") as fallback,
            patch("scraper.extract_article_text", side_effect=lambda url, **kw: "Short text" if url.endswith("short") else ""),
        ):
            results = fetch_and_extract_batch("AI hardware", diagnostics=stats)
        self.assertEqual(results, [])
        fallback.assert_not_called()
        self.assertEqual(stats["unavailable"], 1)
        self.assertEqual(stats["too_short"], 1)

    def test_one_broken_publisher_does_not_discard_another_article(self) -> None:
        items = [
            {"title": "Bad", "url": "https://example.com/bad"},
            {"title": "Good", "url": "https://example.com/good"},
        ]
        def extract(url, **kwargs):
            if url.endswith("bad"):
                raise ValueError("Bad page")
            return "Valid article. " * 30
        with (
            patch("scraper.fetch_latest_news", return_value=items),
            patch("scraper._resolve_final_url", side_effect=lambda url: url),
            patch("scraper.extract_article_text", side_effect=extract),
            patch("scraper._fetch_bing_news") as fallback,
        ):
            results = fetch_and_extract_batch("AI hardware", max_articles=2)
        self.assertEqual([x["title"] for x in results], ["Good"])
        fallback.assert_not_called()

    def test_does_not_send_untranslated_chinese_query_to_english_feed(self) -> None:
        requested_sources = []

        def fetch_source(_url: str, _limit: int, source_name: str):
            requested_sources.append(source_name)
            return [
                {
                    "title": "海外产品与中国供应链研发",
                    "url": "https://example.cn/supply-chain",
                    "source": "测试媒体",
                }
            ]

        with patch("scraper._fetch_rss_news", side_effect=fetch_source):
            results = fetch_latest_news(
                "海外产品 中国供应链研发",
                max_articles=2,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(requested_sources, ["Google News 中文"])

    def test_merges_english_news_and_prioritizes_authoritative_sources(self) -> None:
        requested_urls = []

        def fetch_source(_url: str, _limit: int, source_name: str):
            requested_urls.append(_url)
            if source_name.startswith("Google News 中文"):
                return [
                    {"title": "中文行业新闻", "url": "https://example.cn/1", "source": "普通媒体"},
                    {"title": "中文权威新闻", "url": "https://example.cn/2", "source": "新华社"},
                ]
            return [
                {"title": "English industry report", "url": "https://reuters.com/1", "source": "Reuters"},
            ]

        with patch("scraper._fetch_rss_news", side_effect=fetch_source):
            results = fetch_latest_news(
                "复星AI",
                max_articles=3,
                english_query="Fosun AI",
                additional_queries=["复星 人工智能 产业布局"],
                english_queries=["Fosun artificial intelligence strategy"],
            )

        self.assertEqual(results[0]["source"], "Reuters")
        self.assertTrue(any(item.get("language") == "en" for item in results))
        self.assertTrue(any("q=Fosun+AI" in url for url in requested_urls))
        self.assertTrue(
            any(
                "q=%E5%A4%8D%E6%98%9F+%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD+%E4%BA%A7%E4%B8%9A%E5%B8%83%E5%B1%80"
                in url
                for url in requested_urls
            )
        )
        self.assertTrue(
            any(
                "q=Fosun+artificial+intelligence+strategy" in url
                for url in requested_urls
            )
        )

    def test_searches_up_to_four_selected_intent_queries_per_language(self) -> None:
        requested_urls = []

        def fetch_source(url: str, _limit: int, source_name: str):
            requested_urls.append(url)
            return [
                {
                    "title": f"{source_name} 新闻",
                    "url": f"https://example.com/{len(requested_urls)}",
                    "source": source_name,
                }
            ]

        with patch("scraper._fetch_rss_news", side_effect=fetch_source):
            fetch_latest_news(
                "中文方向一",
                max_articles=9,
                english_query="English direction one",
                additional_queries=["中文方向二", "中文方向三", "中文方向四"],
                english_queries=[
                    "English direction two",
                    "English direction three",
                    "English direction four",
                    "English manual supplement",
                ],
            )

        self.assertEqual(len(requested_urls), 9)
        for query in (
            "中文方向一",
            "中文方向二",
            "中文方向三",
            "中文方向四",
            "English direction one",
            "English direction two",
            "English direction three",
            "English direction four",
            "English manual supplement",
        ):
            encoded = quote_plus(query)
            self.assertTrue(any(f"q={encoded}" in url for url in requested_urls))

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
