from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent import BodyQualitySchema, NewsCaseSchema, run_pipeline
from database import (
    append_case, load_cases, load_task_runs, save_last_search_state,
    update_case_review_status,
)
from search_task import get_background_search, start_background_search
from session_workspace import SessionWorkspace


ROOT = Path(__file__).resolve().parents[1]


class SessionIsolationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.shared_db = str(Path(directory.name) / "legacy.db")
        environment = patch.dict(os.environ, {
            "DATABASE_PATH": self.shared_db, "DEPLOYMENT_MODE": "cloud_demo",
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.alice = SessionWorkspace()
        self.bob = SessionWorkspace()
        self.addCleanup(self.alice.close)
        self.addCleanup(self.bob.close)

    def _case(self, title: str) -> dict:
        return {
            "title": title, "url": "https://example.com/same-story",
            "content_hash": "same-story", "topic_id": "CUSTOM",
            "topic_name": "Private topic", "industry_keyword": title,
            "summary": title + " private summary", "bullet_points": ["Revenue grew 20%"],
            "evidence_quotes": ["Revenue grew 20%"], "relevance_score": 90,
        }

    def _app(self, workspace: SessionWorkspace | None = None) -> AppTest:
        app = AppTest.from_file(ROOT / "app.py")
        if workspace:
            app.session_state["private_workspace"] = workspace
        app.run(timeout=20)
        self.assertEqual(list(app.exception), [])
        return app

    def _visible_tables(self, app: AppTest) -> str:
        return "\n".join(item.value.to_string() for item in app.dataframe)

    def _wait(self, task_id: str, owner: SessionWorkspace) -> dict:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            task = get_background_search(task_id, owner_id=owner.owner_id)
            if task and task["status"] != "running":
                return task
            time.sleep(.01)
        self.fail("Background task did not finish")

    def test_news_cases_reviews_and_new_sessions_are_isolated(self) -> None:
        append_case(self._case("LEGACY PRIVATE"), db_path=self.shared_db)
        save_last_search_state({"keyword": "LEGACY PRIVATE", "articles": []}, db_path=self.shared_db)
        self.assertTrue(append_case(self._case("ALICE PRIVATE"), db_path=self.alice.db_path))
        self.assertTrue(append_case(self._case("BOB PRIVATE"), db_path=self.bob.db_path))
        alice_app, bob_app = self._app(self.alice), self._app(self.bob)
        self.assertIn("ALICE PRIVATE", self._visible_tables(alice_app))
        self.assertNotIn("BOB PRIVATE", self._visible_tables(alice_app))
        self.assertIn("BOB PRIVATE", self._visible_tables(bob_app))
        self.assertNotIn("ALICE PRIVATE", self._visible_tables(bob_app))
        self.assertNotIn("LEGACY PRIVATE", self._visible_tables(alice_app))
        alice_app.button(key="open_history").click().run(timeout=20)
        # Review IDs may collide between private databases; updates stay private.
        alice_app.button(key="save_review").click().run(timeout=20)
        self.assertEqual(list(alice_app.exception), [])
        self.assertEqual(load_cases(db_path=self.alice.db_path).iloc[0]["review_status"], "已确认")
        self.assertEqual(load_cases(db_path=self.bob.db_path).iloc[0]["review_status"], "待审核")
        new_app = self._app()
        self.assertEqual(len(new_app.dataframe), 0)
        self.assertNotIn("last_search_keyword", new_app.session_state)
        self.assertNotEqual(new_app.session_state["private_workspace"].owner_id, self.alice.owner_id)
        self.assertEqual(load_cases(db_path=self.shared_db).iloc[0]["title"], "LEGACY PRIVATE")

    def test_task_ids_and_urls_cannot_expose_another_sessions_history(self) -> None:
        with patch("search_task.agent.fetch_and_pre_score", return_value=[]):
            task_id = start_background_search({
                "primary_query": "ALICE SECRET QUERY", "confirmed_keyword": "ALICE SECRET QUERY",
                "max_articles": 1, "min_score": 70,
            }, workspace=self.alice)
            self._wait(task_id, self.alice)
        self.assertIsNone(get_background_search(task_id, owner_id=self.bob.owner_id))
        self.assertIsNone(get_background_search(task_id, owner_id=""))
        alice_app = self._app(self.alice)
        alice_app.session_state["search_history"] = [task_id]
        alice_app.button(key="open_history").click().run(timeout=20)
        self.assertTrue(any("ALICE SECRET QUERY" in item.value for item in alice_app.markdown))
        bob_app = AppTest.from_file(ROOT / "app.py")
        bob_app.query_params["search_task"] = task_id
        bob_app.session_state["private_workspace"] = self.bob
        bob_app.session_state["active_search_task_id"] = task_id
        bob_app.session_state["search_history"] = [task_id]
        bob_app.run(timeout=20)
        self.assertEqual(list(bob_app.exception), [])
        self.assertNotIn("search_task", bob_app.query_params)
        bob_app.button(key="open_history").click().run(timeout=20)
        self.assertFalse(any("ALICE SECRET QUERY" in item.value for item in bob_app.markdown))
        self.alice.close()
        self.assertIsNone(get_background_search(task_id, owner_id=self.alice.owner_id))
        self.assertFalse(Path(self.alice.db_path).parent.exists())

    def test_pipeline_writes_articles_and_logs_only_to_the_selected_workspace(self) -> None:
        case = self._case("ALICE PIPELINE")
        article = {
            "title": case["title"], "url": case["url"], "content": "Revenue grew 20%. " * 40,
            "content_hash": "pipeline-hash", "source": "Example News", "published_at": "",
        }
        quality = BodyQualitySchema(
            evidence_score=12, evidence_reason="Supported", completeness_score=8,
            completeness_reason="Complete", transparency_score=8, transparency_reason="Named source",
            headline_body_consistency_score=5, headline_body_consistency_reason="Consistent",
            balance_score=4, balance_reason="Balanced", clarity_score=5, clarity_reason="Clear",
            has_serious_unsupported_claims=False, unsupported_claims_reason="",
        )
        analysis = NewsCaseSchema(
            title=case["title"], url=case["url"], summary=case["summary"],
            bullet_points=case["bullet_points"], evidence_quotes=case["evidence_quotes"],
            involved_companies=[], regions=[], metric_tags=[], relevance_score=90,
            source_credibility_score=20, source_credibility_reason="Verified", body_quality=quality,
        )
        with patch("agent.analyze_article", return_value=analysis):
            result = run_pipeline(
                "ALICE PIPELINE", pre_fetched_articles=[article], pre_screen_completed=True,
                db_path=self.alice.db_path,
            )
        self.assertEqual(result["news_saved"], 1)
        self.assertEqual(len(load_cases(db_path=self.alice.db_path)), 1)
        self.assertEqual(len(load_task_runs(db_path=self.alice.db_path)), 1)
        self.assertTrue(load_cases(db_path=self.bob.db_path).empty)
        self.assertTrue(load_task_runs(db_path=self.bob.db_path).empty)
        self.assertTrue(load_cases(db_path=self.shared_db).empty)
        self.assertTrue(load_task_runs(db_path=self.shared_db).empty)

    def test_background_worker_uses_its_workspace_not_request_or_global_path(self) -> None:
        articles = [{"title": "Private", "url": "https://example.com", "search_relevance_scored": True, "search_relevance_score": 90}]
        with (
            patch("search_task.agent.fetch_and_pre_score", return_value=articles),
            patch("search_task.agent.score_sources_with_ai", return_value=[]),
            patch("search_task.agent.run_pipeline", return_value={"details": [], "cases": []}) as pipeline,
        ):
            task_id = start_background_search({
                "primary_query": "Private", "confirmed_keyword": "Private", "max_articles": 1,
                "min_score": 70, "db_path": self.shared_db,
            }, workspace=self.alice)
            self._wait(task_id, self.alice)
        self.assertEqual(pipeline.call_args.kwargs["db_path"], self.alice.db_path)


if __name__ == "__main__":
    unittest.main()
