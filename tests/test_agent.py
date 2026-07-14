from __future__ import annotations

import unittest

from agent import _keep_verifiable_evidence


class AgentEvidenceTestCase(unittest.TestCase):
    def test_keeps_only_quotes_present_in_article(self) -> None:
        article = "万豪 2025 年营收达 255 亿美元，同比增长 8%。"
        quotes = [
            "营收达 255 亿美元",
            "净利润同比增长 30%",
            "  ",
        ]
        self.assertEqual(
            _keep_verifiable_evidence(quotes, article),
            ["营收达 255 亿美元"],
        )


if __name__ == "__main__":
    unittest.main()
