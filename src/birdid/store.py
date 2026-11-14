"""识别结果存储。默认使用 SQLite，可选内存后端。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ResultStore(ABC):
    """结果存储接口。"""

    @abstractmethod
    def put(self, result_id: str, payload: Any) -> None:
        """写入识别结果。"""

    @abstractmethod
    def get(self, result_id: str) -> Any | None:
        """读取识别结果，过期或不存在时返回 None。"""

    @abstractmethod
    def mark_pending(self, result_id: str) -> None:
        """标记任务为处理中。"""

    @abstractmethod
    def is_pending(self, result_id: str) -> bool:
        """判断任务是否仍在处理中。"""


class MemoryResultStore(ResultStore):
    """进程内存储，仅适合单进程调试。"""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, Any]] = {}

    def put(self, result_id: str, payload: Any) -> None:
        with self._lock:
            self._data[result_id] = (time.time() + self._ttl, payload)

    def get(self, result_id: str) -> Any | None:
        with self._lock:
            entry = self._data.get(result_id)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at < time.time():
                self._data.pop(result_id, None)
                return None
            return payload

    def mark_pending(self, result_id: str) -> None:
        with self._lock:
            self._data[result_id] = (time.time() + self._ttl, None)

    def is_pending(self, result_id: str) -> bool:
        with self._lock:
            entry = self._data.get(result_id)
            return entry is not None and entry[1] is None


class SqliteResultStore(ResultStore):
    """SQLite 存储，支持多 worker 共享。"""

    def __init__(self, path: Path, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS results ("
                " result_id TEXT PRIMARY KEY,"
                " payload TEXT,"
                " pending INTEGER NOT NULL DEFAULT 0,"
                " expires_at REAL NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON results(expires_at)")

    def put(self, result_id: str, payload: Any) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO results(result_id, payload, pending, expires_at) VALUES(?,?,0,?)",
                (result_id, json.dumps(payload, ensure_ascii=False), time.time() + self._ttl),
            )
            self._purge(conn)

    def mark_pending(self, result_id: str) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO results(result_id, payload, pending, expires_at) VALUES(?,?,1,?)",
                (result_id, None, time.time() + self._ttl),
            )

    def get(self, result_id: str) -> Any | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT payload, pending, expires_at FROM results WHERE result_id=?", (result_id,)
        ).fetchone()
        if row is None:
            return None
        payload, pending, expires_at = row
        if expires_at < time.time():
            with conn:
                conn.execute("DELETE FROM results WHERE result_id=?", (result_id,))
            return None
        if pending:
            return None
        return json.loads(payload) if payload is not None else None

    def is_pending(self, result_id: str) -> bool:
        conn = self._conn()
        row = conn.execute(
            "SELECT pending, expires_at FROM results WHERE result_id=?", (result_id,)
        ).fetchone()
        if row is None:
            return False
        pending, expires_at = row
        return bool(pending) and expires_at >= time.time()

    def _purge(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM results WHERE expires_at < ?", (time.time(),))


def build_store(backend: str, path: Path, ttl_seconds: int) -> ResultStore:
    """按配置构造存储后端。"""
    if backend == "memory":
        return MemoryResultStore(ttl_seconds)
    return SqliteResultStore(path, ttl_seconds)
