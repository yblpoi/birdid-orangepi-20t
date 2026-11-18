"""鸟鸣识别链路：BirdNET 物种打分 + GeoModel 地理/季节后过滤。

BirdNET 输入是 3 秒 32 kHz 单声道波形，输出 11,560 类概率；GeoModel 以
(lat, lon, week) 为输入，输出同一分类体系下各物种的出现概率。两者相乘后排序，
显著降低地理上不可能的误报。

原先 mel 前端（STFT/ReduceL2）留在 CPU、GeoModel 也走 CPU，两段合计约占声学
链路一半时间。现在整条链路都在 NPU 上：

  * mel 前端：把「分帧 + 窗函数 + 实数 DFT + 取模平方 + 投影到 128 mel 通道」
    折叠成一个 (128, 1, 2048) 的 Conv1d + MatMul，权重由原 STFT 窗和 mel 投影
    矩阵离线算出，导出 .om 后单窗口 1.27 ms（CPU 前端 22.9 ms，约 18 倍）。
    4 窗口时走批量 .om（一次前向 4.3 ms），其余窗口数走单窗 .om；不足 4 窗
    时尾部补零，补零窗口的输出落在尾部、不参与结果，语义安全。
  * CNN 主干：原 backward .om，mel -> 11,560 类 logits，单窗口 12.5 ms。
  * GeoModel：整图转 .om，单次 0.88 ms。

三者都在 device 0 上串行执行，NPU 本来就是单卡串行，线程数增加只会在队列里排队。
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .sound import GeoModelFilter, LabelTable, resample, split_windows

BIRDNET_SEGMENT_SECONDS = 3.0
BIRDNET_SAMPLE_RATE = 32000

# mel 前端的窗口参数，与 BirdNET v3 官方 STFT 保持一致：
# frame_length=2048、hop=512、reflect pad 1024、每个 3 秒窗得到 188 帧。
MEL_FRAME_LENGTH = 2048
MEL_HOP = 512
MEL_PAD = 1024
MEL_FRAMES_PER_WINDOW = 188
MEL_BINS = 128

LOGGER = logging.getLogger(__name__)


@dataclass
class SoundResult:
    """鸟鸣识别结果条目。"""

    code: str
    scientific_name: str
    common_name: str
    confidence: float
    geo_score: float


def _reflect_pad(samples: np.ndarray, pad: int) -> np.ndarray:
    """按 numpy 的 reflect 语义补齐两端（与 ONNX Pad mode=reflect 一致）。"""
    if samples.size <= 1:
        return np.pad(samples, (pad, pad), mode="edge")
    return np.pad(samples, (pad, pad), mode="reflect")


def frames_of(samples: np.ndarray) -> np.ndarray:
    """把一个 3 秒窗口切成 (188, 2048) 的帧矩阵。

    做的是「reflect pad 1024 -> 以 hop 512 取 2048 长帧」，与模型内原始
    STFT 算子的分帧完全一致；整段音频可以在样本维拼好后一次性分帧，等价于
    逐窗口分帧再拼接（窗口间在样本维是连续的）。
    """
    padded = _reflect_pad(np.asarray(samples, dtype=np.float32), MEL_PAD)
    windows = sliding_window_view(padded, MEL_FRAME_LENGTH)
    return np.ascontiguousarray(windows[::MEL_HOP], dtype=np.float32)


def frames_many(windows: list[np.ndarray]) -> np.ndarray:
    """多个 3 秒窗口一次性分帧成 (n*188, 2048)。

    逐窗口调用 `frames_of` 再 concatenate 会把每帧复制两遍（窗口内一次、
    拼接一次），12 窗实测 9.5 ms；这里先按窗口算好目标行号、一次分配输出
    缓冲、再逐窗口写入窗口视图，同一批数据 1.8 ms 完成，结果逐位相同。
    """
    if not windows:
        return np.zeros((0, MEL_FRAME_LENGTH), dtype=np.float32)
    count = len(windows)
    rows = MEL_FRAMES_PER_WINDOW
    out = np.empty((count * rows, MEL_FRAME_LENGTH), dtype=np.float32)
    for index, window in enumerate(windows):
        padded = _reflect_pad(np.asarray(window, dtype=np.float32), MEL_PAD)
        frames = sliding_window_view(padded, MEL_FRAME_LENGTH)[::MEL_HOP]
        out[index * rows : (index + 1) * rows] = frames[:rows]
    return out


class MelFrontend:
    """mel 前端（分帧 + DFT + mel 投影），优先走 NPU。"""

    def __init__(
        self,
        om_path: Path | None,
        batch_om_path: Path | None = None,
        device_id: int = 0,
    ) -> None:
        self._acl = None
        self._batch_acl = None
        self._batch_windows = 0
        self.backend = "cpu"
        if om_path is not None and Path(om_path).exists():
            from .acl_runner import AclModel

            try:
                self._acl = AclModel(om_path, device_id)
                if batch_om_path is not None and Path(batch_om_path).exists():
                    self._batch_acl = AclModel(batch_om_path, device_id)
                    self._batch_windows = (
                        int(self._batch_acl.inputs[0].shape[0]) // MEL_FRAMES_PER_WINDOW
                    )
                self.backend = "npu"
            except Exception:  # pragma: no cover - 部署环境异常时回退
                LOGGER.warning("mel 前端 .om 加载失败，回退 CPU 前端", exc_info=True)
                self._acl = None
                self._batch_acl = None

    @property
    def available(self) -> bool:
        return self._acl is not None

    def mel(self, frames: np.ndarray) -> np.ndarray:
        """(188, 2048) 帧矩阵 -> (188, 128) mel 特征。"""
        if self._acl is None:
            raise RuntimeError("mel 前端的 NPU 后端不可用")
        return np.asarray(self._acl.infer(np.ascontiguousarray(frames, dtype=np.float32))[0])

    def mels(self, frames: np.ndarray) -> np.ndarray:
        """(n*188, 2048) 帧矩阵 -> (n*188, 128)。

        批量 OM 的吞吐明显好于单窗口 OM（4 窗口 4.1 ms，逐窗要 5.7 ms），所以
        能凑满一批就凑：先按批量大小整块跑，剩下的零头再逐窗跑。8 窗口实测
        11.2 ms -> 8.2 ms，12 窗口 17.1 ms -> 12.5 ms。
        """
        if self._acl is None:
            raise RuntimeError("mel 前端的 NPU 后端不可用")
        rows = MEL_FRAMES_PER_WINDOW
        total = int(frames.shape[0])
        if total % rows:
            raise ValueError(f"帧数 {total} 不是窗口帧数 {rows} 的整数倍")
        windows = total // rows
        if windows <= 0:
            return np.zeros((0, MEL_BINS), dtype=np.float32)
        blocks: list[np.ndarray] = []
        start = 0
        chunk = self._batch_windows if self._batch_acl is not None else 1
        while start < windows:
            stop = min(start + chunk, windows)
            piece = frames[start * rows : stop * rows]
            if stop - start == chunk and self._batch_acl is not None:
                blocks.append(
                    np.asarray(
                        self._batch_acl.infer(np.ascontiguousarray(piece, dtype=np.float32))[0]
                    )
                )
            else:
                # 零头：批量 OM 跑不满一批，逐窗更省（1 窗 1.4 ms vs 4.1 ms）。
                for index in range(start, stop):
                    blocks.append(self.mel(frames[index * rows : (index + 1) * rows]))
            start = stop
        return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)

    def close(self) -> None:
        if self._batch_acl is not None:
            self._batch_acl.close()
            self._batch_acl = None
        if self._acl is not None:
            self._acl.close()
            self._acl = None


class BirdNetModel:
    """BirdNET 声学模型封装。

    BirdNET v3 的 ONNX 图前端是 STFT + ReduceL2，昇腾 CANN 未提供这两种算子
    插件，整图无法直接转 .om。因此按 mel 特征边界把图切成两段，并把前端
    重写成 CANN 可转换的等价形式：

      * 前端 wave -> mel：用离线算好的 DFT 权重把「加窗 + rfft + 取模平方 +
        mel 投影」折叠成 Conv1d/MatMul，转成 .om 跑 NPU（单窗口 1.27 ms，
        原 CPU 前端 22.9 ms）；4 窗口走批量 .om 一次算完。
      * 后端 mel -> 11,560 类 logits（CNN 主干）跑 NPU .om（单窗口 12.5 ms）。

    两段拼接与整图 CPU 推理逐元素一致（mel 特征 max abs diff = 5.2e-05，
    4/4 窗口 top1 相同、top10 重合 100%）。

    主干还能再切一刀：尾部那四个 [11560, 2048, 1] 的 Conv 组成打分头，改写成
    MatMul 后可以真正批量，4 个窗口一次前向 7.8 ms（逐窗要 23.5 ms）。整段
    4 窗从 52.1 ms 降到 37.7 ms（1.38 倍），12 窗 top1 判断与整图逐一相同。
    """

    def __init__(
        self,
        onnx_path: Path,
        backend: str = "cpu",
        om_path: Path | None = None,
        front_onnx: Path | None = None,
        mel_om: Path | None = None,
        mel_batch_om: Path | None = None,
        device_id: int = 0,
        trunk_om: Path | None = None,
        head_batch_oms: list[Path] | None = None,
    ) -> None:
        self.backend = "cpu"
        self._session = None
        self._acl_model = None
        self._trunk_acl = None
        self._heads: list[_HeadBatch] = []
        self._front = None
        self._front_input = ""
        self._mel_npu: MelFrontend | None = None
        self._back_shape: tuple[int, ...] = ()
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        want_npu = backend == "npu"
        has_back = om_path is not None and om_path.exists()
        if has_back and want_npu:
            from .acl_runner import AclModel

            self._acl_model = AclModel(om_path, device_id)
            self._back_shape = tuple(int(d) for d in self._acl_model.inputs[0].shape)
            self._mel_npu = MelFrontend(mel_om, mel_batch_om, device_id)
            # 可选：把主干再切成 trunk + 批量 head。只有两个 .om 都加载成功，
            # 且 head 的批大小 >= 2 时才有意义（单窗口整图更快）。
            if trunk_om is not None and head_batch_oms:
                try:
                    self._trunk_acl = AclModel(trunk_om, device_id)
                    heads: list[_HeadBatch] = []
                    for path in head_batch_oms:
                        batch = _HeadBatch(AclModel(path, device_id))
                        if batch.size >= 2:
                            heads.append(batch)
                        else:
                            LOGGER.warning("head %s 的批量大小只有 %d，忽略", path.name, batch.size)
                            batch.close()
                    if not heads:
                        raise ValueError("没有可用的批量 head")
                    heads.sort(key=lambda item: item.size, reverse=True)
                    self._heads = heads
                except Exception:  # pragma: no cover - 部署环境异常时回退
                    LOGGER.warning("主干切分模型加载失败，回退整图主干", exc_info=True)
                    self.close_heads()
                    if self._trunk_acl is not None:
                        self._trunk_acl.close()
                        self._trunk_acl = None
            if self._mel_npu.available:
                self.backend = "npu-split" if self._split_ready else "npu"
                LOGGER.info(
                    "BirdNET 全 NPU 后端：mel 前端 %s%s + CNN 主干 %s",
                    getattr(mel_om, "name", mel_om),
                    (
                        f"（{self._mel_npu._batch_windows} 窗口批量 {getattr(mel_batch_om, 'name', mel_batch_om)}）"
                        if self._mel_npu._batch_acl is not None
                        else ""
                    ),
                    om_path.name,
                )
            elif front_onnx is not None and front_onnx.exists():
                # NPU mel 不可用时退回原来的 CPU 前端 + NPU 主干混合后端
                self._front = ort.InferenceSession(
                    str(front_onnx), options, providers=["CPUExecutionProvider"]
                )
                self._front_input = self._front.get_inputs()[0].name
                self.backend = "hybrid-front-cpu-back-npu"
                LOGGER.info("BirdNET 混合后端：mel 前端 %s 走 CPU，CNN 主干 %s 走 NPU", front_onnx.name, om_path.name)
            else:
                LOGGER.warning("未找到 mel 前端 .om / %s，BirdNET 回退 CPU（推理会明显变慢）", front_onnx)
                self._acl_model.close()
                self._acl_model = None
                self._session = ort.InferenceSession(
                    str(onnx_path), options, providers=["CPUExecutionProvider"]
                )
                self.backend = "cpu"
        else:
            if want_npu:
                LOGGER.warning(
                    "未找到 %s / %s，BirdNET 回退 CPU（推理会明显变慢）", front_onnx, om_path
                )
            self._session = ort.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
            self.backend = "cpu"

    def _mel_cpu(self, samples: np.ndarray) -> np.ndarray:
        """CPU 前端：单窗口 (96000,) -> (188, 128)。"""
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

    def scores(self, samples: np.ndarray) -> np.ndarray:
        """对单个 3 秒窗口返回类别概率向量（模型内已含 sigmoid）。"""
        return self.score_windows([samples])[0]

    def score_windows(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        """对多个窗口返回类别概率。

        NPU 全链路：一次性把整段音频分帧（等价于逐窗口分帧拼接），mel 前端在
        NPU 上一次算完，再逐窗喂给 CNN 主干（310B1 上批量主干语义不正确，
        只能单窗口串行）。
        """
        if not windows:
            return []
        if self._acl_model is not None and self._mel_npu is not None and self._mel_npu.available:
            frames = frames_many(windows)
            mels = self._mel_npu.mels(frames)
            pieces = [
                mels[i * MEL_FRAMES_PER_WINDOW : (i + 1) * MEL_FRAMES_PER_WINDOW]
                for i in range(len(windows))
            ]
            # 单窗口整图更快（12.9 ms vs 15.0 ms）；2 窗以上批量 head 才有收益。
            # 单窗口整图更快（12.9 ms vs 15.0 ms）；2 窗以上批量 head 才有收益。
            if self._split_ready and len(pieces) >= 2:
                return self._scores_split(pieces)
            return [self._scores_from_mel(mel) for mel in pieces]
        if self._acl_model is not None and self._front is not None:
            return [self._scores_from_mel(self._mel_cpu(window)) for window in windows]
        if self._acl_model is not None:
            return [self._scores_from_mel(frames_of(window)) for window in windows]
        return [self._scores_cpu(window) for window in windows]

    def _scores_split(self, mels: list[np.ndarray]) -> list[np.ndarray]:
        """mel 列表 -> 类别概率列表，走「trunk 逐窗 + 批量 head」分段链路。

        310B1 上整图主干每次只能算一个窗口（批量语义会被尾段 ReduceMean/Reshape
        折叠掉），而主干尾部那四个 [11560, 2048, 1] 的 Conv 占了单窗 13 ms 里的
        5 ms 左右。把主干切成 trunk（卷积主干）与 head（那四个 Conv 组成的打分头）
        之后，head 改写成 MatMul 就能真正批量：4 个窗口的 head 一次前向只要 7.2 ms，
        而逐窗跑要 23.5 ms。

        批量档不止一种：B4/B8/B12 实测分别是 7.2/9.7/14.0 ms，单位窗口成本随批量
        下降。长音频（最多 27 个 3 秒窗口）按最优批量序列切块，尾部补零凑满一批，
        27 窗口的 head 成本从「7×B4 = 50.6 ms」降到「B12+B8+B8 = 33.4 ms」。
        """
        assert self._trunk_acl is not None and self._heads
        # 先把所有窗口的 trunk 一次跑完，再按批量档成组喂 head，避免不同档之间来回切。
        tails = [
            self._trunk_acl.infer(np.ascontiguousarray(mel, dtype=np.float32))
            for mel in mels
        ]
        scores: list[np.ndarray] = []
        offset = 0
        sizes = [head.size for head in self._heads]
        for size in head_batch_plan(len(mels), sizes):
            head = self._head_by_size(size)
            chunk = tails[offset : offset + size]
            scores.extend(head.run(chunk))
            offset += len(chunk)
        return scores

    @property
    def _split_ready(self) -> bool:
        """trunk 与至少一个批量 head 都可用时才走切分链路。"""
        return self._trunk_acl is not None and bool(self._heads)

    def _head_by_size(self, size: int) -> "_HeadBatch":
        for head in self._heads:
            if head.size == size:
                return head
        raise KeyError(f"没有批量大小为 {size} 的 head")

    def _scores_cpu(self, samples: np.ndarray) -> np.ndarray:
        feed = np.ascontiguousarray(samples[None, :], dtype=np.float32)
        name = self._session.get_inputs()[0].name
        output = self._session.run(None, {name: feed})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        """释放后端资源。"""
        if self._mel_npu is not None:
            self._mel_npu.close()
            self._mel_npu = None
        if self._acl_model is not None:
            self._acl_model.close()
            self._acl_model = None
        self.close_heads()
        if self._trunk_acl is not None:
            self._trunk_acl.close()
            self._trunk_acl = None
        self._session = None
        self._front = None

    def close_heads(self) -> None:
        """释放所有批量 head。"""
        for head in self._heads:
            head.close()
        self._heads = []


class _HeadBatch:
    """一个批量档的打分头，负责把一个批次的 trunk 输出喂进 .om 并取回概率。"""

    def __init__(self, model) -> None:
        self._model = model
        self.size = int(model.inputs[0].shape[0])
        self._feeds = None

    def run(self, tails: list[list[np.ndarray]]) -> list[np.ndarray]:
        """tails 里每个元素是一次 trunk 推理的多个输出；不足 size 时尾部补零。"""
        if self._feeds is None:
            # 按 head 的输入规格预分配批缓冲：每块一次 concatenate 要 1.7 ms，
            # 直接写进预分配数组后降到 0.06 ms。
            self._feeds = [
                np.zeros((self.size,) + tuple(spec.shape[1:]), dtype=spec.dtype)
                for spec in self._model.inputs
            ]
        for slot, feed in enumerate(self._feeds):
            for offset, tail in enumerate(tails):
                feed[offset] = tail[slot]
            if len(tails) < self.size:
                feed[len(tails) :] = 0
        output = self._model.infer(self._feeds)[0]
        batch = np.asarray(output, dtype=np.float32)
        return [batch[offset].reshape(-1) for offset in range(len(tails))]

    def close(self) -> None:
        self._model.close()
        self._feeds = None


def head_batch_plan(windows: int, sizes: list[int]) -> list[int]:
    """把 windows 个窗口切成若干批，返回每批的批量档，使总耗时最小。

    单批成本（310B1 实测，喂 fp16）：B4 7.2 ms、B8 9.7 ms、B12 14.0 ms，单位窗口
    成本随批量下降，所以长音频要用大档。不足一批时尾部补零（补出来的窗口落在
    批次末尾，结果直接丢弃），因此窗口数不必是批量的整数倍。27 窗口的最优解是
    B12+B8+B8 = 33.4 ms，而只按 B4 切要 50.6 ms。
    """
    if windows <= 0:
        return []
    usable = sorted({size for size in sizes if size >= 2})
    if not usable:
        return []
    cost = {4: 7.2, 8: 9.7, 12: 14.0}
    # best[n] 是 n 个窗口的最小总耗时，pick[n] 是这个最优解的最后一批（批量档，装载数）。
    best = [0.0] + [float("inf")] * windows
    pick: list[tuple[int, int]] = [(0, 0)] * (windows + 1)
    for total in range(1, windows + 1):
        for size in usable:
            for take in range(1, min(size, total) + 1):
                candidate = best[total - take] + cost.get(size, size * 1.8)
                if candidate < best[total]:
                    best[total] = candidate
                    pick[total] = (size, take)
    plan: list[int] = []
    remaining = windows
    while remaining > 0:
        size, take = pick[remaining]
        plan.append(size)
        remaining -= take
    return plan[::-1]


def optional_model(model_dir: Path, name: str) -> Path | None:
    """把配置里的可选模型文件名解析成路径；空串或缺失时返回 None。"""
    if not name:
        return None
    path = model_dir / name
    return path if path.exists() else None


class SoundPipeline:
    """把 BirdNET 与 GeoModel 串成一次鸟鸣识别。"""

    def __init__(self, config) -> None:
        self._config = config
        sound = config.sound
        model_dir = config.resolve(sound.model_dir)
        self._labels = LabelTable(config.resolve(sound.labels_file))
        # 每次 predict 都重建 11,560 条 code->label 的字典很贵（约 7 ms），这里缓存。
        self._by_code = self._labels.by_code()
        self._birdnet = BirdNetModel(
            model_dir / sound.birdnet_model,
            sound.birdnet_backend,
            model_dir / sound.birdnet_om,
            model_dir / sound.birdnet_front_onnx,
            optional_model(model_dir, sound.birdnet_mel_om),
            optional_model(model_dir, sound.birdnet_mel_batch_om),
            config.npu.device_id,
            optional_model(model_dir, sound.birdnet_trunk_om),
            # 多个批量档用逗号分隔；缺文件的直接被跳过，加载失败的档位自动降级。
            [
                path
                for path in (
                    optional_model(model_dir, name.strip())
                    for name in sound.birdnet_head_batch_oms.split(",")
                )
                if path is not None
            ],
        )
        # GeoModel 有自己的分类体系与标签表，不能复用 BirdNET 的标签。
        self._geo_labels = LabelTable(config.resolve(sound.geomodel_labels_file))
        self._geo = GeoModelFilter(
            model_dir / sound.geomodel_model,
            self._geo_labels,
            backend=sound.geomodel_backend,
            om_path=optional_model(model_dir, sound.geomodel_om),
            device_id=config.npu.device_id,
        )

    def close(self) -> None:
        """释放模型资源。"""
        self._birdnet.close()
        self._geo.close()

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
        # 11,560 个类别只取前 9 个，argpartition 比整表 argsort 快 5 倍
        # （12 秒音频 4 窗口实测 5.53 ms -> 1.20 ms）。
        keep = self._config.sound.top_k * 3
        for scores in self._birdnet.score_windows(windows):
            indices = np.argpartition(-scores, keep - 1)[:keep]
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

        by_code = self._by_code
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
