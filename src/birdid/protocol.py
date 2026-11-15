"""懂鸟协议的消息码与响应体构造。"""

from __future__ import annotations

from typing import Any, Final

CODE_OK: Final[int] = 1000
CODE_PROCESSING: Final[int] = 1001
CODE_SOUND_ERROR: Final[int] = 1002
CODE_QUOTA: Final[int] = 1007
CODE_NO_ANIMALS: Final[int] = 1010

MSG_NO_ANIMALS: Final[str] = "No animals"
MSG_URL_FAILED: Final[str] = "URL download failed"
MSG_SOUND_URL_ERROR: Final[str] = "Sound url error"
MSG_PROCESSING: Final[str] = "processing"
MSG_BUSY: Final[str] = "server busy"

VALID_CLASSES: Final[frozenset[str]] = frozenset({"B", "M", "A", "R", "F"})


def ok(payload: Any) -> list[Any]:
    """业务成功响应。"""
    return [CODE_OK, payload]


def processing(result_id: str) -> list[Any]:
    """任务已受理或仍在处理。"""
    return [CODE_PROCESSING, result_id]


def no_animals() -> list[Any]:
    """未识别到任何动物。"""
    return [CODE_NO_ANIMALS, MSG_NO_ANIMALS]


def error(code: int, message: str) -> list[Any]:
    """业务失败响应。"""
    return [code, message]


def normalize_class(value: str | None) -> str:
    """校验 class 字段，非法值回退为 B。"""
    if not value:
        return "B"
    cleaned = "".join(ch for ch in value.upper() if ch in VALID_CLASSES)
    return cleaned or "B"


def result_entry(
    box: tuple[float, float, float, float],
    items: list[list[Any]],
) -> dict[str, Any]:
    """构造结果数组中的单个条目。"""
    x1, y1, x2, y2 = box
    return {
        "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        "list": items,
    }


def species_label(common: str, english: str, scientific: str) -> str:
    """构造懂鸟使用的 "中文名|英文名|学名" 字符串。"""
    return f"{common}|{english}|{scientific}"
