"""异步任务执行：受理请求、后台推理、结果落库，供轮询接口读取。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .protocol import MSG_NO_ANIMALS, no_animals, ok
from .store import ResultStore

LOGGER = logging.getLogger(__name__)


@dataclass
class TaskRecord:
    """一次识别任务的内部状态。"""

    result_id: str
    kind: str
    class_code: str
    created_at: float
    payload: Any = None
    state: str = "pending"
    extra: dict[str, Any] = field(default_factory=dict)


def build_result_id(class_code: str) -> str:
    """生成懂鸟风格的 resultid。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    return f"HH-{class_code}_{stamp}_{uuid.uuid4().hex[:4]}"


class TaskExecutor:
    """后台线程池执行推理，结果写入 ResultStore。"""

    def __init__(self, store: ResultStore, max_workers: int = 2, ttl_seconds: int = 1800) -> None:
        self._store = store
        self._ttl = ttl_seconds
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="birdid")
        self._lock = threading.Lock()

    def submit(
        self,
        kind: str,
        class_code: str,
        work: Callable[[], Any],
        accepted_payload: Any = None,
    ) -> str:
        """受理任务，返回 resultid；work 在后台线程执行。"""
        result_id = build_result_id(class_code)
        self._store.put(result_id, {"state": "processing", "kind": kind}, ttl_seconds=self._ttl)
        with self._lock:
            self._pool.submit(self._run, result_id, kind, work, accepted_payload)
        return result_id

    def _run(self, result_id: str, kind: str, work: Callable[[], Any], accepted_payload: Any) -> None:
        started = time.perf_counter()
        try:
            outcome = work()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if outcome is None:
                payload = no_animals()
            else:
                payload = ok(outcome)
            record = {
                "state": "done",
                "kind": kind,
                "payload": payload,
                "elapsed_ms": round(elapsed_ms, 2),
            }
            LOGGER.info("任务完成 resultid=%s kind=%s %.1fms", result_id, kind, elapsed_ms)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("任务失败 resultid=%s", result_id)
            record = {"state": "failed", "kind": kind, "error": str(exc), "payload": no_animals()}
        self._store.put(result_id, record, ttl_seconds=self._ttl)

    def get(self, result_id: str) -> dict[str, Any] | None:
        """读取任务结果。"""
        return self._store.get(result_id)

    def shutdown(self) -> None:
        """停止线程池。"""
        self._pool.shutdown(wait=False, cancel_futures=True)