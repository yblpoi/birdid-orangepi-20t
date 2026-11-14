"""鸟鸣识别链路：BirdNET 物种识别 + GeoModel 地理/季节后过滤。

BirdNET 负责逐时间窗给出物种概率，GeoModel 依据 (纬度, 经度, 周) 给出
该物种在当地当时的出现概率，两者相乘后排序。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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
    """线性插值重采样（足够用于 48k 到 16k 这类整比例场景）。"""
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
    """GeoModel 后过滤。默认走 CPU（ONNX Runtime），可选 NPU 后端。"""

    def __init__(self, model_path: Path, labels: LabelTable, backend: str = "cpu") -> None:
        self.backend = backend
        self.labels = labels
        self._model_path = model_path
        self._session = self._build_session(model_path) if backend == "cpu" else None

    @staticmethod
    def _build_session(model_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    def scores(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        """返回每个物种的出现概率向量。"""
        if self._session is None:
            raise RuntimeError("GeoModel 的 NPU 后端尚未配置，请使用 cpu 后端")
        inputs = np.asarray([[latitude, longitude, float(week)]], dtype=np.float32)
        output = self._session.run(None, {self._session.get_inputs()[0].name: inputs})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def apply(
        self,
        scored: Iterable[tuple[str, float]],
        latitude: float,
        longitude: float,
        week: int,
    ) -> list[tuple[str, float, float]]:
        """按地理概率重排 BirdNET 结果，返回 (code, 鸟类概率, 地理概率)。"""
        geo = self.scores(latitude, longitude, week)
        by_code = self.labels.by_code()
        combined: list[tuple[str, float, float]] = []
        for code, confidence in scored:
            label = by_code.get(code)
            geo_prob = 0.0
            if label is not None and label.index < geo.shape[0]:
                geo_prob = float(geo[label.index])
            combined.append((code, confidence, geo_prob))
        combined.sort(key=lambda item: item[1] * max(item[2], 1e-6), reverse=True)
        return combined