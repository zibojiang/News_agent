"""
scheduler.py — 独立主题采集调度器

worker 定期扫描已启用的研究主题，只对已到采集时间的主题
执行「抓取 → Agent 分析 → SQLite 入库」流水线。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from agent import run_pipeline
from database import get_last_run_time, get_scheduler_health, load_topics, set_system_state

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_POLL_MINUTES = 30


def _heartbeat() -> None:
    """写入 worker 心跳，供 Streamlit 任务中心显示状态。"""
    set_system_state("worker_heartbeat", "独立定时 worker 运行中")


def _topic_is_due(topic: dict[str, Any], now: datetime | None = None) -> bool:
    """判断主题是否已到下一次采集时间。"""
    now = now or datetime.now()
    last_run = get_last_run_time(str(topic["topic_id"]))
    if last_run is None:
        return True
    interval_hours = max(1, int(topic.get("collection_interval_hours", 24)))
    return now >= last_run + timedelta(hours=interval_hours)


def _run_topic(topic: dict[str, Any]) -> dict[str, Any]:
    """执行单个研究主题的定时采集。"""
    keyword = str(topic.get("search_keywords") or topic.get("topic_name") or "").strip()
    logger.info("定时采集主题: %s %s", topic.get("topic_id"), keyword)
    return run_pipeline(
        industry_keyword=keyword,
        min_score=int(topic.get("min_score", 70)),
        max_articles=int(topic.get("max_articles", 8)),
        topic=topic,
        trigger_type="scheduled",
    )


def run_due_topics(
    force: bool = False, max_topics: int | None = None
) -> list[dict[str, Any]]:
    """扫描并执行已到期主题，每轮默认最多处理 2 个主题。"""
    _heartbeat()
    topics_df = load_topics(enabled_only=True, include_custom=False)
    summaries: list[dict[str, Any]] = []
    cycle_limit = max_topics or int(os.getenv("MAX_TOPICS_PER_CYCLE", "2"))
    cycle_limit = max(1, cycle_limit)

    for topic in topics_df.to_dict(orient="records"):
        if len(summaries) >= cycle_limit:
            break
        if not force and not _topic_is_due(topic):
            continue
        try:
            summaries.append(_run_topic(topic))
        except Exception as exc:
            logger.error(
                "定时主题执行异常 [%s]: %s",
                topic.get("topic_id"),
                exc,
                exc_info=True,
            )

    logger.info("本轮定时扫描完成，执行主题数: %d", len(summaries))
    _heartbeat()
    return summaries


def start_worker(
    poll_minutes: int = DEFAULT_POLL_MINUTES,
    run_immediately: bool = True,
) -> None:
    """启动阻塞式独立 worker，通常由 ``python3 worker.py`` 调用。"""
    poll_minutes = max(5, int(poll_minutes))
    scheduler = BlockingScheduler(
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": 3600,
        },
        timezone=os.getenv("TZ", "Asia/Shanghai"),
    )
    scheduler.add_job(
        _heartbeat,
        trigger=IntervalTrigger(minutes=1),
        id="worker_heartbeat",
        replace_existing=True,
    )
    scheduler.add_job(
        run_due_topics,
        trigger=IntervalTrigger(minutes=poll_minutes),
        id="industry_news_due_topics",
        replace_existing=True,
    )

    _heartbeat()
    logger.info("Industry News worker 已启动，每 %d 分钟扫描主题", poll_minutes)
    if run_immediately:
        run_due_topics()
    try:
        scheduler.start()
    finally:
        set_system_state("worker_heartbeat", "worker 已停止")


def is_scheduler_running() -> bool:
    """兼容旧入口：根据 worker 心跳返回状态。"""
    return bool(get_scheduler_health().get("running"))


def get_next_run_time() -> str | None:
    """独立 worker 按主题计算到期时间，不再提供单一全局时间。"""
    return None
