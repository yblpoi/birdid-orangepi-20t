"""BirdTrace-FastViT-MA36-10K 分类器封装。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SpeciesPrediction:
    """单个物种预测。"""

    index: int
    species_id: str
    species_name: str
    confidence: float


class SpeciesRegistry:
    """把模型输出下标映射到物种信息。"""

    def __init__(self, classes_path: Path) -> None:
        with classes_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        entries = raw.get("classes") if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError(f"无法解析类别文件: {classes_path}")
        self._by_index: dict[int, tuple[str, str]] = {}
        for position, item in enumerate(entries):
            if isinstance(item, dict):
                index = int(item.get("index", position))
                species_id = str(item.get("species_id", index))
                species_name = str(item.get("species_name", species_id))
            else:
                index, species_id, species_name = position, str(item), str(item)
            self._by_index[index] = (species_id, species_name)

    def __len__(self) -> int:
        return len(self._by_index)

    def lookup(self, index: int) -> tuple[str, str]:
        """返回 (species_id, species_name)；未知下标回退为占位值。"""
        return self._by_index.get(index, (str(index), f"species_{index}"))

    def top_k(self, logits: np.ndarray, k: int) -> list[SpeciesPrediction]:
        """对 logits 做 softmax 后取 top-k。"""
        vector = np.asarray(logits, dtype=np.float32).reshape(-1)
        if len(self._by_index):
            vector = vector[: len(self._by_index)]
        shifted = vector - float(np.max(vector))
        exp = np.exp(shifted)
        probs = exp / exp.sum()
        top = np.argsort(-probs)[:k]
        results: list[SpeciesPrediction] = []
        for index in top:
            species_id, species_name = self.lookup(int(index))
            results.append(
                SpeciesPrediction(
                    index=int(index),
                    species_id=species_id,
                    species_name=species_name,
                    confidence=float(probs[index]) * 100.0,
                )
            )
        return results


def scientific_name(raw: str) -> str:
    """把 species_name 的下划线形式转为学名形式，例如 Turdus_merula -> Turdus merula。"""
    return raw.replace("_", " ").strip()