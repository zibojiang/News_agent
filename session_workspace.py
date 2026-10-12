"""Private, temporary storage owned by one anonymous browser session."""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any


class SessionWorkspace:
    """Keep storage alive while a session or one of its background jobs uses it."""

    def __init__(self) -> None:
        self.owner_id = uuid.uuid4().hex
        self._directory = tempfile.TemporaryDirectory(
            prefix="news-session-", ignore_cleanup_errors=True
        )
        self.db_path = str(Path(self._directory.name) / "news.db")
        self._cleanup: list[Callable[[], None]] = []

    def on_close(self, callback: Callable[[], None]) -> None:
        self._cleanup.append(callback)

    def close(self) -> None:
        callbacks, self._cleanup = self._cleanup, []
        for callback in callbacks:
            callback()
        self._directory.cleanup()

    def __del__(self) -> None:
        # Streamlit releases its session state after a disconnected session
        # expires. Running jobs retain this object until they finish.
        try:
            self.close()
        except Exception:
            pass


def session_workspace(state: MutableMapping[str, Any]) -> SessionWorkspace:
    if "private_workspace" not in state:
        state["private_workspace"] = SessionWorkspace()
    return state["private_workspace"]
