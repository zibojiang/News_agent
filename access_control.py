"""
access_control.py — 本地/Community Cloud 访问模式判断

Cloud 展示模式下，如果未配置管理员口令，管理操作默认锁定。
"""

from __future__ import annotations

import hmac
import os

CLOUD_MODES = {"cloud", "cloud_demo", "streamlit_cloud"}


def deployment_mode() -> str:
    """返回当前部署模式。"""
    return os.getenv("DEPLOYMENT_MODE", "local").strip().lower()


def is_cloud_demo() -> bool:
    """是否运行在 Community Cloud 展示模式。"""
    return deployment_mode() in CLOUD_MODES


def admin_password_configured() -> bool:
    """是否已配置有效的管理员口令。"""
    password = os.getenv("ADMIN_PASSWORD", "").strip()
    return bool(password and password != "replace_with_a_strong_password")


def verify_admin_password(candidate: str) -> bool:
    """使用常量时间比较验证管理员口令。"""
    expected = os.getenv("ADMIN_PASSWORD", "").strip()
    if not admin_password_configured():
        return False
    return hmac.compare_digest(candidate, expected)


def local_admin_without_password() -> bool:
    """本地模式未配口令时保留原有的管理员体验。"""
    return not is_cloud_demo() and not admin_password_configured()
