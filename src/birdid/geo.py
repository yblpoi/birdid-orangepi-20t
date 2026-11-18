"""GeoModel 地理先验：按 (纬度, 经度, 周) 给出各物种的出现概率。

GeoModel V3.0.4 是一个纯 MLP 小模型（15 MB），输入 3 个数值、输出 14,082 维
概率向量。CPU 上单次约 1.65 ms，转成 .om 后在 NPU 上约 0.88 ms。

该模型只解决「这个物种在某地某时是否可能出现」，用于给识别结果加权，
不改变模型本身的分类能力。92 张实测：BirdTrace top1 83.0% → 96.6%。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class GeoModel:
    """GeoModel 推理封装：默认 NPU（.om），缺少 .om 时回退 CPU（ONNX Runtime）。"""

    def __init__(
        self,
        model_path: Path,
        backend: str = "cpu",
        om_path: Path | None = None,
        device_id: int = 0,
    ) -> None:
        self.backend = "cpu"
        self._model_path = model_path
        self._session = None
        self._acl_model = None
        want_npu = backend != "cpu" and om_path is not None and Path(om_path).exists()
        if want_npu:
            from .acl_runner import AclModel

            try:
                self._acl_model = AclModel(om_path, device_id)
                self.backend = "npu"
                LOGGER.info("GeoModel 走 NPU：%s", Path(om_path).name)
            except Exception:  # pragma: no cover - 部署环境异常时回退
                LOGGER.warning("GeoModel 的 .om 加载失败，回退 CPU", exc_info=True)
                self._acl_model = None
        if self._acl_model is None:
            if backend != "cpu":
                LOGGER.warning("GeoModel 的 NPU 后端不可用（%s），回退 CPU", om_path)
            self._session = self._build_session(model_path)
            self.backend = "cpu"

    @staticmethod
    def _build_session(model_path: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])

    def scores(self, latitude: float, longitude: float, week: int) -> np.ndarray:
        """返回每个物种在 (lat, lon, week) 下的出现概率向量。"""
        feed = np.asarray([[latitude, longitude, float(week)]], dtype=np.float32)
        if self._acl_model is not None:
            output = self._acl_model.infer(feed)[0]
        else:
            name = self._session.get_inputs()[0].name
            output = self._session.run(None, {name: feed})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        self._session = None
        if self._acl_model is not None:
            self._acl_model.close()
            self._acl_model = None


def rerank(
    candidates: list[tuple[int, float]],
    geo: np.ndarray,
    geomodel_index_of,
    weight: float = 1.0,
) -> list[tuple[int, float]]:
    """按 分类置信度 × 地理概率^weight 重排候选。

    candidates 为 (类序号, 概率)。GeoModel 未覆盖的类别按中性权重 1.0 处理，
    即不做地理惩罚——否则会把只在别的大洲出现的物种无意义地抬上来。
    """
    scored: list[tuple[float, int]] = []
    total = 0.0
    for index, probability in candidates:
        row = geomodel_index_of(index)
        factor = 1.0
        if row is not None and 0 <= row < geo.shape[0]:
            factor = max(float(geo[row]), 1e-6) ** weight
        scored.append((probability * factor, index))
        total += probability
    blended = sum(score for score, _index in scored)
    # 重排只改变各候选的相对权重，不该凭空放大或缩小总置信度，
    # 因此把混合后的分数按原总和等比缩放回去，返回的概率依然可读。
    scale = total / blended if blended > 0 else 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    return [(index, score * scale) for score, index in scored]
