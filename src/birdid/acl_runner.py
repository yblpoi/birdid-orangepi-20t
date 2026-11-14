"""昇腾 ACL 推理封装：加载 .om 模型并执行单批前向。

仅依赖 pyACL（`import acl`），不需要 MindSpore / PyTorch。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SET_ENV_HINT = "/usr/local/Ascend/cann-9.2.1/set_env.sh"


def _ensure_acl_env() -> None:
    """提示用户加载 CANN 环境变量（不修改进程环境，避免副作用）。"""
    if "ASCEND_HOME_PATH" not in os.environ and "LD_LIBRARY_PATH" not in os.environ:
        raise RuntimeError(
            f"未检测到 CANN 环境变量，请先 source {SET_ENV_HINT}"
        )


@dataclass
class TensorSpec:
    """模型输入/输出的张量描述。"""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class AclModel:
    """单个 .om 模型的推理会话。"""

    def __init__(self, model_path: Path, device_id: int = 0) -> None:
        _ensure_acl_env()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.device_id = device_id
        self._acl = __import__("acl")
        self._init_runtime()

    def _init_runtime(self) -> None:
        acl = self._acl
        self._acl.init()
        self._acl.rt.set_device(self.device_id)
        self._context, ret = acl.rt.create_context(self.device_id)
        _check(ret, "acl.rt.create_context")
        self._model_id, ret = acl.mdl.load_from_file(str(self.model_path))
        _check(ret, "acl.mdl.load_from_file")
        self._desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(self._desc, self._model_id), "acl.mdl.get_desc")
        self.inputs = self._read_specs(acl.mdl.get_num_inputs, acl.mdl.get_input_size_by_index)
        self.outputs = self._read_specs(acl.mdl.get_num_outputs, acl.mdl.get_output_size_by_index)

    def _read_specs(self, num_fn, size_fn) -> list[TensorSpec]:
        acl = self._acl
        specs: list[TensorSpec] = []
        for index in range(num_fn(self._desc)):
            size = size_fn(self._desc, index)
            specs.append(
                TensorSpec(
                    name=f"tensor_{index}",
                    shape=(size,),
                    dtype=np.dtype("float32"),
                )
            )
        return specs

    def infer(self, feed: np.ndarray) -> list[np.ndarray]:
        """执行一次推理，返回输出张量列表。"""
        acl = self._acl
        data = np.ascontiguousarray(feed, dtype=np.float32)
        dataset = acl.mdl.create_dataset()
        buffer = acl.rt.malloc(data.nbytes, 0)
        _check(acl.rt.memcpy(buffer, data.nbytes, data.tobytes(), data.nbytes, 1), "h2d")
        acl.mdl.add_dataset_buffer(dataset, _wrap_buffer(acl, buffer, data.nbytes))
        out_dataset = acl.mdl.create_dataset()
        out_buffers = []
        for spec in self.outputs:
            nbytes = int(np.prod(spec.shape)) * 4
            out_buf = acl.rt.malloc(nbytes, 0)
            out_buffers.append((out_buf, nbytes))
            acl.mdl.add_dataset_buffer(out_dataset, _wrap_buffer(acl, out_buf, nbytes))
        try:
            _check(acl.mdl.execute(self._model_id, dataset, out_dataset), "acl.mdl.execute")
            results = []
            for out_buf, nbytes in out_buffers:
                host = np.empty(nbytes // 4, dtype=np.float32)
                _check(acl.rt.memcpy(host.ctypes.data, nbytes, out_buf, nbytes, 2), "d2h")
                results.append(host)
            return results
        finally:
            acl.mdl.destroy_dataset(dataset)
            acl.mdl.destroy_dataset(out_dataset)
            for out_buf, _ in out_buffers:
                acl.rt.free(out_buf)
            acl.rt.free(buffer)

    def close(self) -> None:
        """释放模型与设备资源。"""
        acl = self._acl
        if getattr(self, "_model_id", None) is not None:
            acl.mdl.unload(self._model_id)
            self._model_id = None
        if getattr(self, "_desc", None) is not None:
            acl.mdl.destroy_desc(self._desc)
            self._desc = None
        if getattr(self, "_context", None) is not None:
            acl.rt.destroy_context(self._context)
            self._context = None
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def __enter__(self) -> "AclModel":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _check(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"ACL 调用失败: {what} 返回 {ret}")


def _wrap_buffer(acl, device_ptr: int, nbytes: int):
    """把设备内存包装成 aclDataBuffer。"""
    return acl.create_data_buffer(device_ptr, nbytes)
