"""独立定时采集进程入口。"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from scheduler import start_worker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


if __name__ == "__main__":
    start_worker(
        poll_minutes=int(os.getenv("SCHEDULER_POLL_MINUTES", "30")),
        run_immediately=False,
    )
