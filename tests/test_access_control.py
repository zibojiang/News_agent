from __future__ import annotations

import os
import unittest

from access_control import (
    admin_password_configured,
    is_cloud_demo,
    local_admin_without_password,
    verify_admin_password,
)


class AccessControlTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_mode = os.environ.get("DEPLOYMENT_MODE")
        self.previous_password = os.environ.get("ADMIN_PASSWORD")

    def tearDown(self) -> None:
        if self.previous_mode is None:
            os.environ.pop("DEPLOYMENT_MODE", None)
        else:
            os.environ["DEPLOYMENT_MODE"] = self.previous_mode
        if self.previous_password is None:
            os.environ.pop("ADMIN_PASSWORD", None)
        else:
            os.environ["ADMIN_PASSWORD"] = self.previous_password

    def test_local_mode_keeps_admin_access_without_password(self) -> None:
        os.environ["DEPLOYMENT_MODE"] = "local"
        os.environ.pop("ADMIN_PASSWORD", None)
        self.assertFalse(is_cloud_demo())
        self.assertFalse(admin_password_configured())
        self.assertTrue(local_admin_without_password())

    def test_cloud_mode_fails_closed_without_password(self) -> None:
        os.environ["DEPLOYMENT_MODE"] = "cloud_demo"
        os.environ.pop("ADMIN_PASSWORD", None)
        self.assertTrue(is_cloud_demo())
        self.assertFalse(local_admin_without_password())
        self.assertFalse(verify_admin_password("anything"))

    def test_cloud_mode_accepts_only_matching_password(self) -> None:
        os.environ["DEPLOYMENT_MODE"] = "cloud_demo"
        os.environ["ADMIN_PASSWORD"] = "a-long-demo-password"
        self.assertTrue(admin_password_configured())
        self.assertTrue(verify_admin_password("a-long-demo-password"))
        self.assertFalse(verify_admin_password("wrong-password"))


if __name__ == "__main__":
    unittest.main()
