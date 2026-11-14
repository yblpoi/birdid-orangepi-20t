"""鸟鸣识别链路：BirdNET 物种打分 + GeoModel 地理/季节后过滤。

BirdNET 输入是 3 秒 32 kHz 单声道波形，输出 11,560 类概率；GeoModel 以
(lat, lon, week) 为输入，输出同一分类体系下各物种的出现概率。两者相乘后排序，
显著降低地理上不可能的误报。
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sound import GeoModelFilter, LabelTable, read_wav, resample, split_windows

BIRDNET_SEGMENT_SECONDS = 3.0
BIRDNET_SAMPLE_RATE = 32000


@dataclass
class SoundResult:
    """鸟鸣识别结果条目。"""

    code: str
    scientific_name: str
    common_name: str
    confidence: float
    geo_score: float


class OnnxBirdNet:
    """BirdNET 声学模型的 ONNX 封装（CPU 或 NPU 后端统一接口）。"""

    def __init__(self, model_path: Path, backend: str = "cpu") -> None:
        self.backend = backend
        self._session = None
        if backend == "cpu":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(model_path), options, providers=["CPUExecutionProvider"]
            )
        else:
            from .acl_runner import AclModel

            self._acl_model = AclModel(model_path)

    def scores(self, samples: np.ndarray) -> np.ndarray:
        """对单个 3 秒窗口返回类别概率向量。"""
        feed = np.ascontiguousarray(samples[None, :], dtype=np.float32)
        if self._session is not None:
            name = self._session.get_inputs()[0].name
            output = self._session.run(None, {name: feed})[0]
        else:
            output = self._acl_model.infer(feed)[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        return vector

    def close(self) -> None:
        if self._session is None and getattr(self, "_acl_model", None) is not None:
            self._acl_model.close()


class SoundPipeline:
    """把 BirdNET 与 GeoModel 串成一次鸟鸣识别。"""

    def __init__(self, config) -> None:
        self._config = config
        sound = config.sound
        model_dir = config.resolve(sound.model_dir)
        self._labels = LabelTable(config.resolve(sound.labels_file))
        self._birdnet = OnnxBirdNet(model_dir / sound.birdnet_model, sound.birdnet_backend)
        self._geo = GeoModelFilter(
            model_dir / sound.geomodel_model,
            self._labels,
            backend=sound.geomodel_backend,
        )

    def close(self) -> None:
        """释放模型资源。"""
        self._birdnet.close()

    def predict(
        self,
        raw: bytes,
        latitude: float | None = None,
        longitude: float | None = None,
        week: int | None = None,
    ) -> list[Any] | None:
        """识别一段音频，返回懂鸟结果条目数组。"""
        samples = self._decode(raw)
        if samples.size == 0:
            return None
        windows = split_windows(
            samples,
            BIRDNET_SAMPLE_RATE,
            BIRDNET_SEGMENT_SECONDS,
            self._config.sound.overlap,
        )
        best: dict[str, float] = {}
        for window in windows:
            scores = self._birdnet.scores(window)
            indices = np.argsort(-scores)[: self._config.sound.top_k * 3]
            for index in indices:
                label = self._labels.labels[int(index)] if int(index) < len(self._labels) else None
                if label is None:
                    continue
                score = float(scores[index])
                if score < self._config.sound.min_confidence:
                    continue
                if score > best.get(label.species_code, 0.0):
                    best[label.species_code] = score
        if not best:
            return None

        scored = sorted(best.items(), key=lambda item: -item[1])[: self._config.sound.top_k]
        if latitude is not None and longitude is not None and week is not None:
            ranked = self._geo.apply(scored, latitude, longitude, week)
        else:
            ranked = [(code, score, 1.0) for code, score in scored]
        by_code = self._labels.by_code()
        entries: list[list[Any]] = []
        for code, confidence, geo_score in ranked[: self._config.sound.top_k]:
            label = by_code.get(code)
            if label is None:
                continue
            label_text = f"{label.common_name}|{label.common_name}|{label.scientific_name}"
            entries.append([round(confidence * 100.0, 1), label_text, label.index, "B"])
        if not entries:
            return None
        return [{"box": [], "list": entries}]

    def _decode(self, raw: bytes) -> np.ndarray:
        """把上传的音频解码成 32 kHz 单声道 float32。"""
        buffer = io.BytesIO(raw)
        try:
            with wave.open(buffer, "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            if width != 2:
                return np.zeros(0, dtype=np.float32)
            data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
        except wave.Error:
            return np.zeros(0, dtype=np.float32)
        return resample(data, rate, BIRDNET_SAMPLE_RATE)