from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTestCase(unittest.TestCase):
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
                    }
                ]
                app.session_state["fetched_keyword"] = "复星AI"
                app.run(timeout=20)
                self.assertTrue(
                    any("已搜索到的文章（1篇）" in markdown.value for markdown in app.markdown)
                )
                self.assertTrue(
                    any("✓ 已搜索到" in markdown.value for markdown in app.markdown)
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
