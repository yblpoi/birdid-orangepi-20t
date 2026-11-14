"""鸟鸣识别链路：BirdNET 物种识别 + GeoModel 地理/季节后过滤。

BirdNET 负责逐时间窗给出物种概率，GeoModel 依据 (纬度, 经度, 周) 给出
该物种在当地当时的出现概率，两者相乘后排序。

注意：BirdNET 与 GeoModel 是两套独立的分类体系（11,560 类 vs 14,082 类），
行序号互不相干，二者只能按学名关联。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# BirdNET 的 week 取 1..48，每周按 7.6 天切分（官方约定）。
WEEKS_IN_YEAR = 48


@dataclass(frozen=True)
class SoundDetection:
    """单条鸟鸣识别结果。"""

    species_code: str
    scientific_name: str
    common_name: str
    confidence: float
    geo_score: float


@dataclass(frozen=True)
class SoundLabel:
    """标签表的一行。"""

    index: int
    species_code: str
    scientific_name: str
    common_name: str


class LabelTable:
    """加载 BirdNET / GeoModel 的标签表。

    支持三种格式：
      - BirdNET v3 官方 CSV（分号分隔，含 idx;id;sci_name;com_name 表头）；
      - GeoModel Labels.txt（制表符分隔的 speciesCode / 学名 / 俗名）；
      - 逗号分隔的两列或三列简表。
    """

    def __init__(self, path: Path) -> None:
        self.labels: list[SoundLabel] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                if self._parse_row(line):
                    continue

    def _parse_row(self, line: str) -> bool:
        """解析一行；表头行返回 True 但跳过。"""
        if ";" in line:
            parts = [item.strip() for item in line.split(";")]
            if parts[:4] == ["idx", "id", "sci_name", "com_name"]:
                return True
            if len(parts) >= 4:
                self._append(parts[1], parts[2], parts[3])
                return True
        parts = [item.strip() for item in line.split("\t")]
        if len(parts) == 1:
            parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 3:
            self._append(parts[0], parts[1], parts[2])
        elif len(parts) == 2:
            self._append(parts[0], parts[1], parts[1])
        elif parts:
            self._append(parts[0], parts[0], parts[0])
        return True

    def _append(self, code: str, scientific: str, common: str) -> None:
        self.labels.append(
            SoundLabel(
                index=len(self.labels),
                species_code=code,
                scientific_name=scientific,
                common_name=common,
            )
        )

    def __len__(self) -> int:
        return len(self.labels)

    def by_code(self) -> dict[str, SoundLabel]:
        return {label.species_code: label for label in self.labels}

    def index_by_scientific_name(self) -> dict[str, int]:
        """学名（小写去空格）到行序号的映射，用于跨分类体系关联。"""
        return {
            label.scientific_name.strip().lower(): label.index
            for label in self.labels
            if label.scientific_name
        }


def week_of_year(when: date | None = None) -> int:
    """把日期换算成 BirdNET 的 week（1..48）。"""
    day = (when or date.today()).timetuple().tm_yday
    week = min(WEEKS_IN_YEAR, day * WEEKS_IN_YEAR // 366 + 1)
    return max(1, week)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """读取 WAV，返回 (samples float32 单声道, sample_rate)。"""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"仅支持 16 位 PCM，当前位宽 {width * 8}")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """线性插值重采样（足够用于 48k 到 32k 这类整比例场景）。"""
    if source_rate == target_rate or data.size == 0:
        return data
    duration = data.shape[0] / float(source_rate)
    target_length = int(round(duration * target_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.linspace(0.0, duration, data.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(target_positions, source_positions, data).astype(np.float32)


def split_windows(data: np.ndarray, rate: int, window_seconds: float, overlap: float) -> list[np.ndarray]:
    """按时间窗切片，返回每个窗口的样本。"""
    window = int(round(window_seconds * rate))
    if window <= 0:
        return []
    if data.shape[0] < window:
        padded = np.zeros(window, dtype=np.float32)
        padded[: data.shape[0]] = data
        return [padded]
    step = max(1, int(round(window * (1.0 - overlap))))
    windows: list[np.ndarray] = []
    for start in range(0, data.shape[0] - window + 1, step):
        windows.append(data[start : start + window])
    return windows


class GeoModelFilter:
    """GeoModel 后过滤。默认走 CPU（ONNX Runtime），可选 NPU 后端。

    BirdNET 与 GeoModel 的分类体系不同（11,560 类 / 14,082 类），行序号不可
    直接对齐，因此这里按学名建立索引；GeoModel 未覆盖的物种按中性权重处理。
    """

    def __init__(self, model_path: Path, geo_labels: LabelTable, backend: str = "cpu") -> None:
        self.backend = backend
        self.geo_labels = geo_labels
        self._model_path = model_path
        self._index_by_name = geo_labels.index_by_scientific_name()
        self._session = self._build_session(model_path) if backend == "cpu" else None

    @staticmethod
    def _build_session(model_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    def index_of(self, scientific_name: str) -> int | None:
        """按学名查 GeoModel 行序号；未覆盖时返回 None。"""
        return self._index_by_name.get(scientific_name.strip().lower())

    def scores(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        """返回每个物种的出现概率向量。"""
        if self._session is None:
            raise RuntimeError("GeoModel 的 NPU 后端尚未配置，请使用 cpu 后端")
        inputs = np.asarray([[latitude, longitude, float(week)]], dtype=np.float32)
        output = self._session.run(None, {self._session.get_inputs()[0].name: inputs})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def rank(
        self,
        candidates: Iterable[tuple[str, str, float]],
        latitude: float,
        longitude: float,
        week: int,
    ) -> list[tuple[str, float, float]]:
        """按 鸟类置信度 × 地理概率 重排。

        candidates 每项为 (species_code, scientific_name, confidence)。
        返回 (code, 鸟类置信度, 地理概率) 列表，地理概率 -1 表示该物种未被
        GeoModel 覆盖（按中性权重，不做地理惩罚）。
        """
        geo = self.scores(latitude, longitude, week)
        ranked: list[tuple[float, str, float, float]] = []
        for code, scientific_name, confidence in candidates:
            index = self.index_of(scientific_name)
            geo_prob = -1.0
            weight = 1.0
            if index is not None and index < geo.shape[0]:
                geo_prob = float(geo[index])
                weight = max(geo_prob, 1e-6)
            ranked.append((confidence * weight, code, confidence, geo_prob))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [(code, confidence, geo_prob) for _, code, confidence, geo_prob in ranked]
