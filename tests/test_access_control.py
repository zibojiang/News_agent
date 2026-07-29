from __future__ import annotations

import os
import unittest

from access_control import deployment_mode, is_cloud_demo


class AccessControlTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_mode = os.environ.get("DEPLOYMENT_MODE")

    def tearDown(self) -> None:
        if self.previous_mode is None:
            os.environ.pop("DEPLOYMENT_MODE", None)
        else:
            os.environ["DEPLOYMENT_MODE"] = self.previous_mode

    def test_local_mode_is_default(self) -> None:
        os.environ.pop("DEPLOYMENT_MODE", None)
        self.assertEqual(deployment_mode(), "local")
        self.assertFalse(is_cloud_demo())

    def test_local_mode_is_not_cloud_demo(self) -> None:
        os.environ["DEPLOYMENT_MODE"] = "local"
        self.assertFalse(is_cloud_demo())

    def test_cloud_aliases_are_recognized(self) -> None:
        os.environ["DEPLOYMENT_MODE"] = "cloud_demo"
        self.assertTrue(is_cloud_demo())
        os.environ["DEPLOYMENT_MODE"] = " Streamlit_Cloud "
        self.assertTrue(is_cloud_demo())


if __name__ == "__main__":
    unittest.main()
