"""推理后端抽象：优先昇腾 ACL（.om），可回退 ONNX Runtime 便于本地联调。

生产环境使用 ACL 后端把模型跑在 NPU 上；ONNX Runtime 后端只用于在没有
.om 或没有 NPU 的开发机上做功能验证，性能与生产无关。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


class Runtime(Protocol):
    """模型推理后端接口。"""

    name: str

    def infer(self, feed: np.ndarray) -> list[np.ndarray]:
        """执行一次前向，返回输出张量列表。"""

    def close(self) -> None:
        """释放资源。"""


def build_runtime(model_dir: Path, om_name: str, onnx_name: str, device_id: int = 0) -> Runtime:
    """按优先级构造后端：有 .om 用 ACL，否则回退 ONNX Runtime。"""
    om_path = model_dir / om_name
    if om_path.exists():
        from .acl_runner import AclModel

        LOGGER.info("使用 ACL 后端: %s", om_path)
        return _AclRuntime(AclModel(om_path, device_id))
    onnx_path = model_dir / onnx_name
    if onnx_path.exists():
        LOGGER.warning("未找到 %s，回退 ONNX Runtime（仅用于功能验证）: %s", om_name, onnx_path)
        return _OnnxRuntime(onnx_path)
    raise FileNotFoundError(f"模型缺失：{om_path} 与 {onnx_path} 都不存在")


class _AclRuntime:
    """包装 AclModel，接口与其他后端一致。"""

    name = "acl"

    def __init__(self, model) -> None:
        self._model = model

    def infer(self, feed: np.ndarray) -> list[np.ndarray]:
        return self._model.infer(feed)

    def close(self) -> None:
        self._model.close()


class _OnnxRuntime:
    """ONNX Runtime CPU 后端，仅用于无 NPU 时的联调。"""

    name = "onnx"

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )

    def infer(self, feed: np.ndarray) -> list[np.ndarray]:
        name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {name: np.ascontiguousarray(feed, dtype=np.float32)})
        return [np.asarray(item, dtype=np.float32) for item in outputs]

    def close(self) -> None:
        self._session = None