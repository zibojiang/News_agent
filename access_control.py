"""access_control.py — 本地/Community Cloud 运行模式判断。"""

from __future__ import annotations

import os

CLOUD_MODES = {"cloud", "cloud_demo", "streamlit_cloud"}


def deployment_mode() -> str:
    """返回当前部署模式。"""
    return os.getenv("DEPLOYMENT_MODE", "local").strip().lower()


def is_cloud_demo() -> bool:
    """是否运行在 Community Cloud 展示模式。"""
    return deployment_mode() in CLOUD_MODES
