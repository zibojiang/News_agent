"""
database.py — 本地 SQLite 数据层

管理研究主题、新闻案例、审核状态和任务日志。
首次运行时会从项目目录下的研究主题 Excel 导入初始主题。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = "data/industry_news.db"
DEFAULT_TOPICS_XLSX = PROJECT_ROOT / "复星旅文DeepResearch研究主题目录_更新版_V4.xlsx"

JSON_COLUMNS = [
    "bullet_points",
    "evidence_quotes",
    "involved_companies",
    "regions",
    "metric_tags",
]

CASE_COLUMNS = [
    "id",
    "discovered_at",
    "published_at",
    "title",
    "url",
    "source",
    "topic_id",
    "dimension",
    "category",
    "topic_name",
    "industry_keyword",
    "summary",
    "bullet_points",
    "evidence_quotes",
    "involved_companies",
    "regions",
    "metric_tags",
    "relevance_score",
    "is_qualified",
    "review_status",
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _database_path(db_path: str | None = None) -> str:
    return db_path or os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH)


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = _database_path(db_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _collection_interval_hours(report_frequency: str) -> int:
    """报告频率不等于采集频率：实时/事件主题 4 小时采集，其他主题每日采集。"""
    return 4 if report_frequency in {"实时", "事件驱动"} else 24


def _seed_topics(connection: sqlite3.Connection) -> None:
    existing_count = connection.execute("SELECT COUNT(*) FROM research_topics").fetchone()[0]
    if existing_count:
        return

    xlsx_path = Path(os.getenv("TOPICS_XLSX_PATH", str(DEFAULT_TOPICS_XLSX)))
    if xlsx_path.exists():
        try:
            topics_df = pd.read_excel(xlsx_path, engine="openpyxl")
            for _, row in topics_df.iterrows():
                topic_id = str(row.get("主题ID", "")).strip()
                topic_name = str(row.get("研究主题", "")).strip()
                if not topic_id or not topic_name or topic_id.lower() == "nan":
                    continue
                frequency = str(row.get("更新频率", "日报")).strip()
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_topics (
                        topic_id, sequence_no, dimension, category, topic_name,
                        report_frequency, search_keywords, collection_interval_hours,
                        typical_case_2024_2025, typical_case_2025_2026, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic_id,
                        int(row.get("序号", 0)),
                        str(row.get("维度", "")).strip(),
                        str(row.get("分类", "")).strip(),
                        topic_name,
                        frequency,
                        f"文旅 {topic_name}",
                        _collection_interval_hours(frequency),
                        str(row.get("典型案例（2024-2025）", "") or "").strip(),
                        str(row.get("典型案例（2025-2026）", "") or "").strip(),
                        _now(),
                    ),
                )
            logger.info("已从 Excel 导入研究主题: %s", xlsx_path)
        except Exception as exc:
            logger.error("导入研究主题 Excel 失败: %s", exc, exc_info=True)
    else:
        logger.warning("研究主题 Excel 不存在: %s", xlsx_path)

    # 保留一个自定义入口，用于临时行业/关键词研究。
    connection.execute(
        """
        INSERT OR IGNORE INTO research_topics (
            topic_id, sequence_no, dimension, category, topic_name,
            report_frequency, search_keywords, collection_interval_hours, updated_at
        ) VALUES ('CUSTOM', 999, '自定义', '自定义', '自定义行业研究', '手动', '文旅', 24, ?)
        """,
        (_now(),),
    )


def initialize_database(db_path: str | None = None) -> str:
    """初始化数据表和索引，重复调用安全。"""
    path = _database_path(db_path)
    with _connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_topics (
                topic_id TEXT PRIMARY KEY,
                sequence_no INTEGER NOT NULL DEFAULT 0,
                dimension TEXT NOT NULL,
                category TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                report_frequency TEXT NOT NULL,
                search_keywords TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                min_score INTEGER NOT NULL DEFAULT 70,
                max_articles INTEGER NOT NULL DEFAULT 8,
                collection_interval_hours INTEGER NOT NULL DEFAULT 24,
                typical_case_2024_2025 TEXT NOT NULL DEFAULT '',
                typical_case_2025_2026 TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS news_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discovered_at TEXT NOT NULL,
                published_at TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                topic_id TEXT NOT NULL DEFAULT 'CUSTOM',
                dimension TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT '',
                topic_name TEXT NOT NULL DEFAULT '',
                industry_keyword TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                bullet_points TEXT NOT NULL DEFAULT '[]',
                evidence_quotes TEXT NOT NULL DEFAULT '[]',
                involved_companies TEXT NOT NULL DEFAULT '[]',
                regions TEXT NOT NULL DEFAULT '[]',
                metric_tags TEXT NOT NULL DEFAULT '[]',
                relevance_score INTEGER NOT NULL DEFAULT 0,
                is_qualified INTEGER NOT NULL DEFAULT 0,
                review_status TEXT NOT NULL DEFAULT '待审核',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(topic_id, url)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_news_cases_topic_hash
                ON news_cases(topic_id, content_hash)
                WHERE content_hash <> '';
            CREATE INDEX IF NOT EXISTS idx_news_cases_discovered_at
                ON news_cases(discovered_at DESC);
            CREATE INDEX IF NOT EXISTS idx_news_cases_topic_id
                ON news_cases(topic_id);

            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id TEXT NOT NULL DEFAULT 'CUSTOM',
                keyword TEXT NOT NULL DEFAULT '',
                trigger_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                status TEXT NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                saved INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                errors TEXT NOT NULL DEFAULT '[]'
            );

            CREATE INDEX IF NOT EXISTS idx_task_runs_started_at
                ON task_runs(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_task_runs_topic_id
                ON task_runs(topic_id);

            CREATE TABLE IF NOT EXISTS system_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _seed_topics(connection)
        connection.commit()
    return path


def load_topics(
    db_path: str | None = None,
    enabled_only: bool = False,
    include_custom: bool = True,
) -> pd.DataFrame:
    """加载研究主题配置。"""
    path = initialize_database(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if enabled_only:
        clauses.append("enabled = 1")
    if not include_custom:
        clauses.append("topic_id <> 'CUSTOM'")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(path) as connection:
        return pd.read_sql_query(
            f"SELECT * FROM research_topics {where_sql} ORDER BY sequence_no, topic_id",
            connection,
            params=params,
        )


def get_topic(topic_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    """按主题 ID 获取单条配置。"""
    path = initialize_database(db_path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM research_topics WHERE topic_id = ?", (topic_id,)
        ).fetchone()
    return dict(row) if row else None


def update_topic(topic_id: str, values: dict[str, Any], db_path: str | None = None) -> bool:
    """更新可编辑的主题配置字段。"""
    allowed_fields = {
        "search_keywords",
        "enabled",
        "min_score",
        "max_articles",
        "collection_interval_hours",
    }
    clean_values = {key: value for key, value in values.items() if key in allowed_fields}
    if not clean_values:
        return False

    if "enabled" in clean_values:
        clean_values["enabled"] = int(bool(clean_values["enabled"]))
    for key in ("min_score", "max_articles", "collection_interval_hours"):
        if key in clean_values:
            clean_values[key] = int(clean_values[key])
    clean_values["updated_at"] = _now()

    assignments = ", ".join(f"{key} = ?" for key in clean_values)
    params = list(clean_values.values()) + [topic_id]
    path = initialize_database(db_path)
    with _connect(path) as connection:
        cursor = connection.execute(
            f"UPDATE research_topics SET {assignments} WHERE topic_id = ?", params
        )
        connection.commit()
    return cursor.rowcount > 0


def load_cases(
    db_path: str | None = None,
    qualified_only: bool | None = None,
    topic_id: str | None = None,
    review_status: str | None = None,
) -> pd.DataFrame:
    """加载新闻池或符合入库标准的案例。"""
    path = initialize_database(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if qualified_only is not None:
        clauses.append("is_qualified = ?")
        params.append(int(qualified_only))
    if topic_id:
        clauses.append("topic_id = ?")
        params.append(topic_id)
    if review_status:
        clauses.append("review_status = ?")
        params.append(review_status)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with _connect(path) as connection:
        df = pd.read_sql_query(
            f"SELECT {', '.join(CASE_COLUMNS)} FROM news_cases "
            f"{where_sql} ORDER BY discovered_at DESC, id DESC",
            connection,
            params=params,
        )
    return df


def url_exists(
    url: str,
    db_path: str | None = None,
    topic_id: str | None = None,
    content_hash: str = "",
) -> bool:
    """按主题 + URL/正文指纹检查重复。"""
    path = initialize_database(db_path)
    clauses = ["url = ?"]
    params: list[Any] = [url]
    if content_hash:
        clauses.append("content_hash = ?")
        params.append(content_hash)
    topic_clause = "topic_id = ? AND " if topic_id else ""
    if topic_id:
        params.insert(0, topic_id)
    with _connect(path) as connection:
        row = connection.execute(
            f"SELECT 1 FROM news_cases WHERE {topic_clause}({' OR '.join(clauses)}) LIMIT 1",
            params,
        ).fetchone()
    return row is not None


def _json_string(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if value in (None, ""):
        return "[]"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps([value], ensure_ascii=False)
    return json.dumps([str(value)], ensure_ascii=False)


def append_case(
    case: dict[str, Any],
    db_path: str | None = None,
    min_score: int = 0,
) -> bool:
    """
    写入一条 Agent 分析结果。

    低分新闻也保留在新闻池，但 ``is_qualified`` 为 0；
    同一主题下的相同 URL 或正文指纹不会重复写入。
    """
    path = initialize_database(db_path)
    url = str(case.get("url", "")).strip()
    if not url:
        logger.warning("案例缺少 URL，跳过写入")
        return False

    score = max(0, min(100, int(case.get("relevance_score", 0))))
    topic_id = str(case.get("topic_id") or "CUSTOM")
    content_hash = str(case.get("content_hash", "")).strip()
    is_qualified = int(
        score >= min_score
        and bool(case.get("bullet_points"))
        and bool(case.get("evidence_quotes"))
    )
    review_status = str(
        case.get("review_status") or ("待审核" if is_qualified else "低相关")
    )
    now = _now()

    values = (
        str(case.get("discovered_at") or now),
        str(case.get("published_at") or ""),
        str(case.get("title") or ""),
        url,
        str(case.get("source") or ""),
        content_hash,
        topic_id,
        str(case.get("dimension") or ""),
        str(case.get("category") or ""),
        str(case.get("topic_name") or ""),
        str(case.get("industry_keyword") or ""),
        str(case.get("summary") or ""),
        _json_string(case.get("bullet_points")),
        _json_string(case.get("evidence_quotes")),
        _json_string(case.get("involved_companies")),
        _json_string(case.get("regions")),
        _json_string(case.get("metric_tags")),
        score,
        is_qualified,
        review_status,
        now,
        now,
    )

    try:
        with _connect(path) as connection:
            connection.execute(
                """
                INSERT INTO news_cases (
                    discovered_at, published_at, title, url, source, content_hash,
                    topic_id, dimension, category, topic_name, industry_keyword,
                    summary, bullet_points, evidence_quotes, involved_companies,
                    regions, metric_tags, relevance_score, is_qualified,
                    review_status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            connection.commit()
        logger.info("已写入新闻: %s (score=%d)", case.get("title"), score)
        return True
    except sqlite3.IntegrityError:
        logger.info("新闻已存在，跳过: %s", url)
        return False
    except Exception as exc:
        logger.error("写入 SQLite 失败: %s", exc, exc_info=True)
        return False


def append_cases_batch(
    cases: list[dict[str, Any]],
    db_path: str | None = None,
    min_score: int = 70,
) -> int:
    """批量写入分析结果，返回新增的合格案例数。"""
    qualified_saved = 0
    for case in cases:
        inserted = append_case(case, db_path=db_path, min_score=min_score)
        if (
            inserted
            and int(case.get("relevance_score", 0)) >= min_score
            and bool(case.get("bullet_points"))
            and bool(case.get("evidence_quotes"))
        ):
            qualified_saved += 1
    return qualified_saved


def update_case_review_status(
    case_id: int, review_status: str, db_path: str | None = None
) -> bool:
    """更新人工审核状态。"""
    allowed = {"待审核", "已确认", "已忽略", "低相关"}
    if review_status not in allowed:
        raise ValueError(f"不支持的审核状态: {review_status}")
    path = initialize_database(db_path)
    with _connect(path) as connection:
        cursor = connection.execute(
            "UPDATE news_cases SET review_status = ?, updated_at = ? WHERE id = ?",
            (review_status, _now(), int(case_id)),
        )
        connection.commit()
    return cursor.rowcount > 0


def record_task_run(summary: dict[str, Any], db_path: str | None = None) -> int:
    """记录一次手动或定时任务的运行摘要。"""
    path = initialize_database(db_path)
    errors = summary.get("errors", [])
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO task_runs (
                topic_id, keyword, trigger_type, started_at, finished_at, status,
                processed, saved, skipped, error_count, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(summary.get("topic_id") or "CUSTOM"),
                str(summary.get("keyword") or ""),
                str(summary.get("trigger_type") or "manual"),
                str(summary.get("started_at") or _now()),
                str(summary.get("finished_at") or _now()),
                str(summary.get("status") or "unknown"),
                int(summary.get("processed", 0)),
                int(summary.get("saved", 0)),
                int(summary.get("skipped", 0)),
                len(errors),
                json.dumps(errors, ensure_ascii=False),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def load_task_runs(limit: int = 100, db_path: str | None = None) -> pd.DataFrame:
    """加载最近任务日志。"""
    path = initialize_database(db_path)
    with _connect(path) as connection:
        return pd.read_sql_query(
            """
            SELECT id, topic_id, keyword, trigger_type, started_at, finished_at,
                   status, processed, saved, skipped, error_count, errors
            FROM task_runs ORDER BY id DESC LIMIT ?
            """,
            connection,
            params=(int(limit),),
        )


def get_last_run_time(topic_id: str, db_path: str | None = None) -> datetime | None:
    """获取主题最后一次任务开始时间。"""
    path = initialize_database(db_path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT MAX(started_at) AS last_run FROM task_runs WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
    if not row or not row["last_run"]:
        return None
    try:
        return datetime.strptime(row["last_run"], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def set_system_state(key: str, value: str, db_path: str | None = None) -> None:
    """写入 worker 心跳等轻量系统状态。"""
    path = initialize_database(db_path)
    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO system_state (state_key, state_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(state_key) DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = excluded.updated_at
            """,
            (key, value, _now()),
        )
        connection.commit()


def get_scheduler_health(db_path: str | None = None, stale_seconds: int = 180) -> dict[str, Any]:
    """根据独立 worker 心跳判断定时服务是否活跃。"""
    path = initialize_database(db_path)
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT state_value, updated_at FROM system_state WHERE state_key = 'worker_heartbeat'"
        ).fetchone()
    if not row:
        return {"running": False, "last_heartbeat": None, "message": "未检测到 worker"}
    try:
        updated_at = datetime.strptime(row["updated_at"], "%Y-%m-%d %H:%M:%S")
        age = (datetime.now() - updated_at).total_seconds()
        running = age <= stale_seconds
    except ValueError:
        running = False
    return {
        "running": running,
        "last_heartbeat": row["updated_at"],
        "message": row["state_value"],
    }


def format_json_list_for_display(raw_value: Any, numbered: bool = False) -> str:
    """将 SQLite 中的 JSON 数组转为前端/导出可读文本。"""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return ""
    if isinstance(raw_value, str):
        try:
            items = json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
    elif isinstance(raw_value, list):
        items = raw_value
    else:
        return str(raw_value)
    if not isinstance(items, list):
        return str(items)
    if numbered:
        return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))
    return "、".join(str(item) for item in items)


def format_bullet_points_for_display(bullet_points_raw: Any) -> str:
    """将量化案例数组转为编号多行文本。"""
    return format_json_list_for_display(bullet_points_raw, numbered=True)


def export_to_excel(df: pd.DataFrame, output_path: str) -> str:
    """将新闻案例导出为 Excel。"""
    export_df = df.copy()
    for column in JSON_COLUMNS:
        if column in export_df.columns:
            export_df[column] = export_df[column].apply(
                lambda value: format_json_list_for_display(
                    value, numbered=(column in {"bullet_points", "evidence_quotes"})
                )
            )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_df.to_excel(output_path, index=False, engine="openpyxl")
    logger.info("已导出 Excel: %s", output_path)
    return output_path
