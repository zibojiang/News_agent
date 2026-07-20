from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from database import (
    append_case,
    append_cases_batch_with_summary,
    get_last_run_time,
    initialize_database,
    load_cases,
    load_task_runs,
    load_topics,
    record_task_run,
    update_case_review_status,
)


class DatabaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_database_path = os.environ.get("DATABASE_PATH")
        self.previous_topics_path = os.environ.get("TOPICS_XLSX_PATH")
        os.environ["DATABASE_PATH"] = str(Path(self.temp_dir.name) / "test.db")
        os.environ["TOPICS_XLSX_PATH"] = str(
            Path(__file__).resolve().parents[1]
            / "复星旅文DeepResearch研究主题目录_更新版_V4.xlsx"
        )
        initialize_database()

    def tearDown(self) -> None:
        if self.previous_database_path is None:
            os.environ.pop("DATABASE_PATH", None)
        else:
            os.environ["DATABASE_PATH"] = self.previous_database_path
        if self.previous_topics_path is None:
            os.environ.pop("TOPICS_XLSX_PATH", None)
        else:
            os.environ["TOPICS_XLSX_PATH"] = self.previous_topics_path
        self.temp_dir.cleanup()

    def _case(self, **overrides: object) -> dict[str, object]:
        case: dict[str, object] = {
            "title": "测试新闻",
            "url": "https://example.com/news/1",
            "source": "测试媒体",
            "content_hash": "hash-one",
            "topic_id": "S1.1",
            "topic_name": "全球重点市场动态",
            "dimension": "事",
            "category": "供给端",
            "industry_keyword": "文旅 全球市场",
            "summary": "测试摘要",
            "bullet_points": ["测试企业营收同比增长 20%"],
            "evidence_quotes": ["营收同比增长 20%"],
            "involved_companies": ["测试企业"],
            "regions": ["中国"],
            "metric_tags": ["营收"],
            "relevance_score": 85,
        }
        case.update(overrides)
        return case

    def test_imports_28_excel_topics_and_custom_topic(self) -> None:
        topics = load_topics(include_custom=False)
        all_topics = load_topics(include_custom=True)
        self.assertEqual(len(topics), 28)
        self.assertEqual(len(all_topics), 29)
        self.assertIn("S1.1", topics["topic_id"].tolist())

    def test_news_pool_qualification_and_topic_scoped_deduplication(self) -> None:
        self.assertTrue(append_case(self._case(), min_score=70))
        self.assertFalse(append_case(self._case(), min_score=70))

        low_score = self._case(
            title="低分新闻",
            url="https://example.com/news/2",
            content_hash="hash-two",
            relevance_score=45,
            bullet_points=[],
        )
        self.assertTrue(append_case(low_score, min_score=70))

        no_evidence = self._case(
            title="无证据高分新闻",
            url="https://example.com/news/3",
            content_hash="hash-three",
            evidence_quotes=[],
        )
        self.assertTrue(append_case(no_evidence, min_score=70))

        # 同一 URL 可同时归入另一研究主题。
        second_topic = self._case(topic_id="S1.2", topic_name="对标品牌追踪")
        self.assertTrue(append_case(second_topic, min_score=70))

        self.assertEqual(len(load_cases()), 4)
        self.assertEqual(len(load_cases(qualified_only=True)), 2)
        self.assertEqual(len(load_cases(qualified_only=False)), 2)

    def test_batch_summary_distinguishes_news_cases_and_duplicates(self) -> None:
        low_score = self._case(
            title="低分新闻",
            url="https://example.com/news/2",
            content_hash="hash-two",
            relevance_score=45,
            bullet_points=[],
            evidence_quotes=[],
        )
        result = append_cases_batch_with_summary(
            [self._case(), low_score], min_score=70
        )

        self.assertEqual(result["news_inserted"], 2)
        self.assertEqual(result["qualified_inserted"], 1)
        self.assertEqual(result["duplicates"], 0)
        self.assertEqual(result["write_failed"], 0)
        self.assertEqual(len(load_cases()), 2)

        duplicate_result = append_cases_batch_with_summary(
            [self._case()], min_score=70
        )
        self.assertEqual(duplicate_result["news_inserted"], 0)
        self.assertEqual(duplicate_result["duplicates"], 1)

    def test_review_status_and_task_log(self) -> None:
        self.assertTrue(append_case(self._case(), min_score=70))
        case_id = int(load_cases(qualified_only=True).iloc[0]["id"])
        self.assertTrue(update_case_review_status(case_id, "已确认"))
        self.assertEqual(load_cases().iloc[0]["review_status"], "已确认")

        record_task_run(
            {
                "topic_id": "S1.1",
                "keyword": "文旅 全球市场",
                "trigger_type": "manual",
                "started_at": "2026-07-13 10:00:00",
                "finished_at": "2026-07-13 10:01:00",
                "status": "success",
                "processed": 1,
                "saved": 1,
                "skipped": 0,
                "errors": [],
            }
        )
        runs = load_task_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs.iloc[0]["status"], "success")
        self.assertIsNotNone(get_last_run_time("S1.1"))


if __name__ == "__main__":
    unittest.main()
