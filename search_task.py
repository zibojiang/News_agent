"""用户手动搜索的后台任务管理器，避免 Streamlit 页面切换中断分析。"""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

import agent


logger = logging.getLogger(__name__)

_TASK_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="manual-search")
_TASK_LOCK = threading.Lock()
_TASKS: dict[str, dict[str, Any]] = {}


def _update_task(task_id: str, **values: Any) -> None:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is not None:
            task.update(values)


def get_background_search(task_id: str) -> dict[str, Any] | None:
    """返回任务快照，调用方无法修改后台线程持有的数据。"""
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        return copy.deepcopy(task) if task is not None else None


def _ranking_score(detail: dict[str, Any]) -> float:
    recommendation = detail.get("recommendation_score")
    if isinstance(recommendation, (int, float)) and not isinstance(
        recommendation, bool
    ):
        return float(recommendation)
    relevance = detail.get("score", 0)
    if isinstance(relevance, (int, float)) and not isinstance(relevance, bool):
        return float(agent.calculate_recommendation_score(relevance, 0))
    return -1.0


def _run_background_search(task_id: str, request: dict[str, Any]) -> None:
    articles: list[dict[str, Any]] = []
    search_started = perf_counter()
    try:
        _update_task(
            task_id,
            stage="search",
            message="正在搜索中英文新闻并提取正文…",
            progress=0.03,
        )

        def article_callback(
            article: dict[str, Any], found: int, total: int
        ) -> None:
            articles.append(article)
            _update_task(
                task_id,
                articles=copy.deepcopy(articles),
                message=f"已搜索到 {found}/{total} 篇有效新闻…",
                progress=min(0.3, 0.03 + 0.27 * found / max(1, total)),
            )

        fetched_articles = agent.fetch_and_pre_score(
            industry_keyword=str(request["primary_query"]),
            max_articles=int(request["max_articles"]),
            article_callback=article_callback,
            english_keyword=str(request.get("english_keyword") or "") or None,
            additional_queries=list(request.get("chinese_queries") or []),
            english_queries=list(request.get("english_queries") or []),
        )
        articles = fetched_articles
        search_seconds = perf_counter() - search_started
        if not articles:
            _update_task(
                task_id,
                status="completed",
                stage="completed",
                message="没有找到可分析的文章",
                progress=1.0,
                articles=[],
                results={},
                run_summary={},
            )
            return

        _update_task(
            task_id,
            stage="source",
            articles=copy.deepcopy(articles),
            message="新闻已展示，正在计算相关性和来源权威度…",
            progress=0.32,
        )
        source_errors = agent.score_sources_with_ai(
            articles,
            original_query=str(request["confirmed_keyword"]),
            search_intent=request.get("search_intent"),
            progress_callback=lambda message, value: _update_task(
                task_id,
                message=message,
                progress=0.32 + 0.23 * value,
                articles=copy.deepcopy(articles),
            ),
        )
        visible_articles = [
            article
            for article in articles
            if article.get("search_relevance_scored") is not True
            or int(article.get("search_relevance_score", 0) or 0)
            >= agent.SEARCH_RELEVANCE_DISPLAY_THRESHOLD
        ]
        relevance_filtered = len(articles) - len(visible_articles)
        articles = visible_articles
        if not articles:
            _update_task(
                task_id,
                status="completed",
                stage="completed",
                message="候选新闻均未达到搜索相关性展示门槛",
                progress=1.0,
                articles=[],
                results={},
                run_summary={
                    "processed": 0,
                    "analyzed": 0,
                    "relevance_filtered": relevance_filtered,
                    "errors": source_errors,
                    "details": [],
                    "cases": [],
                },
            )
            return

        analysis_started = perf_counter()
        _update_task(
            task_id,
            stage="body",
            articles=copy.deepcopy(articles),
            message="相关性初筛完成，正在分析新闻正文质量…",
            progress=0.58,
        )
        run_summary = agent.run_pipeline(
            industry_keyword=str(request["confirmed_keyword"]),
            min_score=int(request["min_score"]),
            max_articles=int(request["max_articles"]),
            topic=None,
            trigger_type="manual",
            progress_callback=lambda message, value: _update_task(
                task_id,
                message=message,
                progress=0.58 + 0.4 * value,
            ),
            pre_fetched_articles=articles,
            pre_screen_completed=True,
        )
        run_summary["search_seconds"] = search_seconds
        run_summary["analysis_seconds"] = perf_counter() - analysis_started
        run_summary["relevance_filtered"] = relevance_filtered
        run_summary["source_errors"] = source_errors

        details = list(run_summary.get("details", []))
        ranked_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for index, article in enumerate(articles):
            detail = (
                details[index]
                if index < len(details)
                else {
                    "title": str(article.get("title", "")),
                    "url": str(article.get("url", "")),
                    "score": article.get("search_relevance_score"),
                    "analysis_status": "未完成",
                    "qualification_status": "-",
                    "storage_status": "未写入",
                    "reason": "分析任务提前结束",
                }
            )
            ranked_pairs.append((article, detail))
        ranked_pairs.sort(key=lambda pair: _ranking_score(pair[1]), reverse=True)
        articles = [article for article, _ in ranked_pairs]
        run_summary["details"] = [detail for _, detail in ranked_pairs]
        results = {
            index: detail for index, detail in enumerate(run_summary["details"])
        }
        _update_task(
            task_id,
            status="completed",
            stage="completed",
            message="搜索和 AI 分析已完成",
            progress=1.0,
            articles=copy.deepcopy(articles),
            results=copy.deepcopy(results),
            run_summary=copy.deepcopy(run_summary),
        )
    except Exception as exc:
        logger.error("后台手动搜索失败: %s", type(exc).__name__, exc_info=True)
        _update_task(
            task_id,
            status="failed",
            stage="failed",
            message="搜索任务失败",
            error=f"{type(exc).__name__}: {exc}",
        )


def start_background_search(request: dict[str, Any]) -> str:
    """启动后台搜索并立即返回任务 ID。"""
    task_id = uuid.uuid4().hex
    with _TASK_LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "status": "running",
            "stage": "queued",
            "message": "搜索任务已提交…",
            "progress": 0.0,
            "keyword": str(request.get("confirmed_keyword") or ""),
            "search_intent": copy.deepcopy(request.get("search_intent") or {}),
            "articles": [],
            "results": {},
            "run_summary": {},
            "error": "",
        }
    _TASK_EXECUTOR.submit(_run_background_search, task_id, copy.deepcopy(request))
    return task_id
