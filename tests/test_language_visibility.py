from __future__ import annotations

import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent import SEARCH_RELEVANCE_RULE_VERSION
from database import append_case, load_cases, save_last_search_state
from quality_scorer import NEWS_QUALITY_RULE_VERSION
from session_workspace import SessionWorkspace


ROOT = Path(__file__).resolve().parents[1]


class LanguageVisibilityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = SessionWorkspace()
        self.addCleanup(self.workspace.close)
        environment = patch.dict(os.environ, {"DEPLOYMENT_MODE": "cloud_demo"})
        environment.start()
        self.addCleanup(environment.stop)
        self.articles = []
        self.results = {}
        for index, (title, content, score, dimensions) in enumerate((
            ("中文芯片市场报告", "中文芯片市场规模增长百分之二十。", 98, (40, 30, 20, 8)),
            ("English AI hardware report", "Hardware revenue grew 20 percent.", 81, (36, 25, 15, 5)),
        )):
            self.articles.append({
                "title": title, "url": f"https://example.com/{index}", "content": content * 20,
                "language": "en",  # Deliberately wrong for the first result.
                "search_relevance_score": score, "search_relevance_scored": True,
                "search_relevance_rule_version": SEARCH_RELEVANCE_RULE_VERSION,
                "search_relevance_dimensions": dict(zip((
                    "core_topic_match", "information_need_match", "semantic_coverage", "directness",
                ), dimensions)),
                "quality_pre": {
                    "adjusted_score": 40, "label": "预筛", "source_score_method": "ai",
                    "rule_version": NEWS_QUALITY_RULE_VERSION,
                    "dimension_scores": {"source_credibility": 20, "content_density": 8, "data_richness": 8, "freshness": 4},
                },
            })
            self.results[index] = {
                "title": title, "url": f"https://example.com/{index}", "score": score,
                "summary": content, "analysis_status": "未完成", "storage_status": "未写入",
            }

    def _app(self) -> AppTest:
        app = AppTest.from_file(ROOT / "app.py")
        app.session_state["private_workspace"] = self.workspace
        return app

    def _assert_cards(self, app: AppTest, chinese: bool) -> None:
        self.assertEqual(list(app.exception), [])
        text = "\n".join(item.value for item in app.markdown)
        self.assertIn(self.articles[1]["title"], text)
        self.assertEqual(self.articles[0]["title"] in text, chinese)
        relevance = [item.value for item in app.metric if item.label in {"Search relevance", "搜索相关性"}]
        self.assertEqual(relevance, ["98/100", "81/100"] if chinese else ["81/100"])

    def test_running_mixed_task_filters_cards_without_mispairing_scores_or_restarting(self) -> None:
        snapshot = {
            "status": "running", "stage": "source", "progress": .3,
            "message": "新闻已展示，正在计算相关性和来源权威度…",
            "keyword": "AI", "articles": self.articles, "results": self.results,
        }
        original = copy.deepcopy(snapshot)
        app = self._app()
        app.session_state["active_search_task_id"] = "existing-mixed-task"
        with (
            patch("search_task.get_background_search", return_value=snapshot),
            patch("agent.fetch_and_pre_score") as fetch,
            patch("agent.score_sources_with_ai") as score,
        ):
            app.run(timeout=20)
            self._assert_cards(app, chinese=False)
            for chinese in (True, False):
                app.button(key="language_toggle").click().run(timeout=20)
                self._assert_cards(app, chinese=chinese)
            fetch.assert_not_called()
            score.assert_not_called()
        self.assertEqual(snapshot, original)

    def test_restored_cards_tables_and_exports_filter_without_deleting_history(self) -> None:
        cases = []
        for index, article in enumerate(self.articles):
            case = {
                **article, "summary": self.results[index]["summary"],
                "relevance_score": article["search_relevance_score"], "topic_name": "AI hardware",
                "bullet_points": [self.results[index]["summary"]],
                "evidence_quotes": [self.results[index]["summary"]],
                "quality_score": 85, "quality_details": {"adjusted_score": 85, "label": "质量良好"},
                "is_qualified": True,
            }
            cases.append(case)
            append_case(case, db_path=self.workspace.db_path)
        save_last_search_state({
            "keyword": "AI", "articles": self.articles, "results": self.results,
            "run_summary": {"started_at": "language-test", "cases": cases, "details": list(self.results.values())},
        }, db_path=self.workspace.db_path)
        app = self._app().run(timeout=20)
        self._assert_cards(app, chinese=False)
        original_articles = copy.deepcopy(app.session_state["fetched_articles"])
        for chinese in (False, True, False):
            if chinese != (app.session_state["ui_language"] == "zh"):
                app.button(key="language_toggle").click().run(timeout=20)
            self._assert_cards(app, chinese=chinese)
            tables = "\n".join(item.value.to_string() for item in app.dataframe)
            self.assertIn(self.articles[1]["title"], tables)
            self.assertEqual(self.articles[0]["title"] in tables, chinese)
            export_frame = next(item.value for item in app.dataframe if "导出" in item.value.columns)
            self.assertEqual(len(export_frame), 2 if chinese else 1)
            self.assertEqual(app.session_state["fetched_articles"], original_articles)
        self.assertEqual(len(load_cases(db_path=self.workspace.db_path)), 2)
        app.button(key="open_history").click().run(timeout=20)
        self.assertEqual(list(app.exception), [])
        history_tables = "\n".join(item.value.to_string() for item in app.dataframe)
        self.assertIn(self.articles[1]["title"], history_tables)
        self.assertNotIn(self.articles[0]["title"], history_tables)


if __name__ == "__main__":
    unittest.main()
