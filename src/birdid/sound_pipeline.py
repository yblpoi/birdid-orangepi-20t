"""鸟鸣识别链路：BirdNET 物种打分 + GeoModel 地理/季节后过滤。

BirdNET 输入是 3 秒 32 kHz 单声道波形，输出 11,560 类概率；GeoModel 以
(lat, lon, week) 为输入，输出同一分类体系下各物种的出现概率。两者相乘后排序，
显著降低地理上不可能的误报。
"""

from __future__ import annotations

import io
import logging
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sound import GeoModelFilter, LabelTable, read_wav, resample, split_windows

BIRDNET_SEGMENT_SECONDS = 3.0
BIRDNET_SAMPLE_RATE = 32000

LOGGER = logging.getLogger(__name__)


@dataclass
class SoundResult:
    """鸟鸣识别结果条目。"""

    code: str
    scientific_name: str
    common_name: str
    confidence: float
    geo_score: float


class BirdNetModel:
    """BirdNET 声学模型封装。

    BirdNET v3 的 ONNX 图前端是 STFT + ReduceL2，昇腾 CANN 未提供这两种算子
    插件，整图无法转 .om。因此按 mel 特征边界把图切成两段：

      * 前端 wave -> mel（STFT/ReduceL2，10 个节点）留在 CPU 用 onnxruntime，
        单窗口约 14 ms；
      * 后端 mel -> 11,560 类 logits（CNN 主干，653 个节点）转成 .om 跑 NPU。

    两段拼接与整图推理逐元素一致（CPU 上实测 max abs diff = 0.0），而 CNN 主干
    单窗口从 91.5 ms（CPU）降到 12.4 ms（NPU）。
    """

    def __init__(
        self,
        onnx_path: Path,
        backend: str = "cpu",
        om_path: Path | None = None,
        front_onnx: Path | None = None,
    ) -> None:
        self.backend = "cpu"
        self._session = None
        self._acl_model = None
        self._front = None
        self._front_input = ""
        self._back_shape: tuple[int, ...] = ()
        self._mel_pool: ThreadPoolExecutor | None = None
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        want_npu = backend == "npu"
        hybrid = (
            want_npu
            and front_onnx is not None
            and front_onnx.exists()
            and om_path is not None
            and om_path.exists()
        )
        if hybrid:
            from .acl_runner import AclModel

            self._front = ort.InferenceSession(
                str(front_onnx), options, providers=["CPUExecutionProvider"]
            )
            self._front_input = self._front.get_inputs()[0].name
            self._acl_model = AclModel(om_path)
            self._back_shape = tuple(int(d) for d in self._acl_model.inputs[0].shape)
            self.backend = "hybrid-front-cpu-back-npu"
            # mel 前端是 CPU 上最重的一段（单窗口约 23 ms），四个窗口串行要 90+ ms。
            # onnxruntime 的 intra_op 线程对单个窗口帮助有限（实测 1/2/4 线程 27.9/23.4/20.8 ms），
            # 但把不同窗口分给多个线程跑可近线性扩展（4 线程 27.7 ms 跑完 4 个窗口），
            # 所以按窗口并行、NPU 主干仍然串行（它本来就单卡串行）。
            workers = min(4, max(1, (os.cpu_count() or 1)))
            self._mel_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mel")
            LOGGER.info(
                "BirdNET 混合后端：mel 前端 %s 走 CPU（%d 线程并行），CNN 主干 %s 走 NPU",
                front_onnx.name,
                workers,
                om_path.name,
            )
        elif want_npu and om_path is not None and om_path.exists():
            # 兼容整图 .om（保留给未来 CANN 支持 STFT 时使用）
            from .acl_runner import AclModel

            self._acl_model = AclModel(om_path)
            self.backend = "npu"
        else:
            if want_npu:
                LOGGER.warning(
                    "未找到 %s / %s，BirdNET 回退 CPU（推理会明显变慢）", front_onnx, om_path
                )
            self._session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
            self.backend = "cpu"

    def _mel(self, samples: np.ndarray) -> np.ndarray:
        """计算单窗口 mel 特征，输出形状 (框架数, 128)。"""
        feed = np.ascontiguousarray(samples[None, :], dtype=np.float32)
        mel = self._front.run(None, {self._front_input: feed})[0]
        return np.asarray(mel, dtype=np.float32)

    def _scores_from_mel(self, mel: np.ndarray) -> np.ndarray:
        """mel 特征走 NPU 主干，返回类别概率。"""
        if len(self._back_shape) == 2:
            feed = mel.reshape(self._back_shape)
        else:
            feed = mel.reshape((1,) + self._back_shape)
        output = self._acl_model.infer(np.ascontiguousarray(feed, dtype=np.float32))[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def mels(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """并行计算多个窗口的 mel 特征。

        onnxruntime 单窗口内部的并行度有限，但窗口之间互不依赖，所以用线程池
        按窗口并发（每个窗口一次 run），CPU 上的这段耗时随窗口数近线性摊薄。
        """
        if self._front is None:
            return []
        if self._mel_pool is None or len(windows) <= 1:
            return [self._mel(window) for window in windows]
        return list(self._mel_pool.map(self._mel, windows))

    def scores(self, samples: np.ndarray) -> np.ndarray:
        """对单个 3 秒窗口返回类别概率向量（模型内已含 sigmoid）。"""
        if self._front is not None and self._acl_model is not None:
            return self._scores_from_mel(self._mel(samples))
        feed = np.ascontiguousarray(samples[None, :], dtype=np.float32)
        if self._acl_model is not None:
            output = self._acl_model.infer(feed)[0]
        else:
            name = self._session.get_inputs()[0].name
            output = self._session.run(None, {name: feed})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def score_windows(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """对多个窗口依次返回类别概率：mel 并行算完，NPU 主干仍逐窗口串行。"""
        mels = self.mels(windows)
        results: list[np.ndarray] = []
        for index, window in enumerate(windows):
            if index < len(mels):
                results.append(self._scores_from_mel(mels[index]))
            else:
                results.append(self.scores(window))
        return results

    def close(self) -> None:
        """释放后端资源。"""
        if self._mel_pool is not None:
            self._mel_pool.shutdown(wait=False)
            self._mel_pool = None
        if self._acl_model is not None:
            self._acl_model.close()
            self._acl_model = None
        self._session = None
        self._front = None


class SoundPipeline:
    """把 BirdNET 与 GeoModel 串成一次鸟鸣识别。"""

    def __init__(self, config) -> None:
        self._config = config
        sound = config.sound
        model_dir = config.resolve(sound.model_dir)
        self._labels = LabelTable(config.resolve(sound.labels_file))
        self._birdnet = BirdNetModel(
            model_dir / sound.birdnet_model,
            sound.birdnet_backend,
            model_dir / sound.birdnet_om,
            model_dir / sound.birdnet_front_onnx,
        )
        # GeoModel 有自己的分类体系与标签表，不能复用 BirdNET 的标签。
        self._geo_labels = LabelTable(config.resolve(sound.geomodel_labels_file))
        self._geo = GeoModelFilter(
            model_dir / sound.geomodel_model,
            self._geo_labels,
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
        for scores in self._birdnet.score_windows(windows):
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

        by_code = self._labels.by_code()
        scored = sorted(best.items(), key=lambda item: -item[1])[: self._config.sound.top_k * 3]
        candidates = [
            (code, by_code[code].scientific_name, score)
            for code, score in scored
            if code in by_code
        ]
        if latitude is not None and longitude is not None and week is not None:
            ranked = self._geo.rank(candidates, latitude, longitude, week)
        else:
            ranked = [(code, score, -1.0) for code, _, score in candidates]
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