from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from i18n import label_formatter, localize, localize_dataframe


class DisplayTranslationTestCase(unittest.TestCase):
    def test_option_labels_retain_the_rendering_sessions_language(self) -> None:
        with patch("i18n.get_language", return_value="zh"):
            chinese_formatter = label_formatter()
        with patch("i18n.get_language", return_value="en"):
            english_formatter = label_formatter()
            self.assertEqual(chinese_formatter("全部"), "全部")
        with patch("i18n.get_language", return_value="zh"):
            self.assertEqual(english_formatter("全部"), "All")
            self.assertEqual(english_formatter("🟢 高质量"), "🟢 High quality")

    def test_localizing_table_preserves_news_text_ids_and_scores(self) -> None:
        # Content may happen to equal a translated system label. It is still news.
        source = pd.DataFrame(
            {
                "新闻标题": ["成功", "高质量"],
                "新闻摘要": ["运行中", "来源"],
                "来源": ["测试媒体", "Example News"],
                "审核状态": ["待审核", "已确认"],
                "质量评级": ["高质量", "质量良好"],
                "新闻质量": [91, 82],
                "topic_id": [10, 11],
                "导出": [True, False],
            },
            index=[42, 43],
        )
        original = source.copy(deep=True)

        with patch("i18n.get_language", return_value="en"):
            english = localize_dataframe(source)
        self.assertEqual(english["Headline"].tolist(), ["成功", "高质量"])
        self.assertEqual(english["Summary"].tolist(), ["运行中", "来源"])
        self.assertEqual(english["Source"].tolist(), ["测试媒体", "Example News"])
        self.assertEqual(english["Review status"].tolist(), ["Pending review", "Confirmed"])
        self.assertEqual(english["Quality rating"].tolist(), ["High quality", "Good quality"])
        self.assertEqual(english["News quality"].tolist(), [91, 82])
        self.assertEqual(english["topic_id"].tolist(), [10, 11])
        self.assertEqual(english["Export"].tolist(), [True, False])
        self.assertEqual(english.index.tolist(), [42, 43])
        pd.testing.assert_frame_equal(source, original)

        with patch("i18n.get_language", return_value="zh"):
            chinese = localize_dataframe(source)
        pd.testing.assert_frame_equal(chinese, original)
        self.assertIsNot(chinese, source)

    def test_only_known_system_messages_are_translated(self) -> None:
        user_text = "报告指出：正文 480 字，介绍行业趋势。"
        with patch("i18n.get_language", return_value="en"):
            self.assertEqual(localize("正文 480 字"), "Article length: 480 characters")
            self.assertEqual(localize("🟢 高质量"), "🟢 High quality")
            self.assertEqual(localize(user_text), user_text)
            self.assertEqual(localize("营收增长 20%，已核验"), "营收增长 20%，已核验")
            self.assertIsNone(localize(None))
            self.assertEqual(localize(88), 88)


if __name__ == "__main__":
    unittest.main()
