import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List

from src.config import get_settings

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS query_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   REAL    NOT NULL,
        query       TEXT    NOT NULL,
        answer      TEXT    NOT NULL,
        num_docs    INTEGER NOT NULL,
        latency_ms  INTEGER NOT NULL
    )
"""


def _ensure_schema(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_TABLE_SQL)


@contextmanager
def _connection() -> Generator[sqlite3.Connection, None, None]:
    db_path = get_settings().monitor_db_path
    _ensure_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_query(query: str, answer: str, num_docs: int, latency_ms: int) -> None:
    with _connection() as conn:
        conn.execute(
            "INSERT INTO query_log (timestamp, query, answer, num_docs, latency_ms) VALUES (?,?,?,?,?)",
            (time.time(), query, answer, num_docs, latency_ms),
        )


def get_stats() -> dict:
    with _connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*), AVG(latency_ms), AVG(num_docs), MIN(latency_ms), MAX(latency_ms)
               FROM query_log"""
        ).fetchone()

    if not row or row[0] == 0:
        return {"total_queries": 0}

    return {
        "total_queries": row[0],
        "avg_latency_ms": round(row[1], 1),
        "avg_docs_retrieved": round(row[2], 2),
        "min_latency_ms": row[3],
        "max_latency_ms": row[4],
    }


def get_recent_queries(limit: int = 20) -> List[dict]:
    with _connection() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, query, answer, num_docs, latency_ms FROM query_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "query": r[2],
            "answer": r[3],
            "num_docs": r[4],
            "latency_ms": r[5],
        }
        for r in rows
    ]
