"""昇腾 ACL 推理封装：加载 .om 模型并执行单批前向。

仅依赖 pyACL（`import acl`），不需要 MindSpore / PyTorch 运行时。
设计目标：
  - 一个进程内可创建多个 AclModel（共享同一 device context 由调用方保证）；
  - 每次推理复用设备侧内存，避免重复 malloc；
  - 输入按模型真实 dims 校验，输出按真实 shape 还原。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# CANN 环境变量提示；实际路径由部署环境决定，这里只做提示不写入
SET_ENV_HINT = "source ${ASCEND_HOME:-/usr/local/Ascend/cann}/set_env.sh"

# aclDataType -> numpy dtype
_DTYPE_MAP = {
    0: np.float32,   # ACL_FLOAT
    1: np.float16,   # ACL_FLOAT16
    2: np.int8,      # ACL_INT8
    3: np.int32,     # ACL_INT32
    4: np.uint8,     # ACL_UINT8
    6: np.int16,     # ACL_INT16
    8: np.int64,     # ACL_INT64
    9: np.uint64,    # ACL_UINT64
    11: np.float64,  # ACL_DOUBLE
    12: np.bool_,    # ACL_BOOL
}

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _check(ret: int, what: str) -> None:
    if ret != 0:
        raise RuntimeError(f"ACL 调用失败: {what} 返回 {ret}")


def _acl_module():
    """延迟导入 pyACL，未安装时给出可操作的错误。"""
    try:
        return __import__("acl")
    except ImportError as exc:  # pragma: no cover - 依赖部署环境
        raise RuntimeError(
            f"未找到 pyACL，请先执行: {SET_ENV_HINT}，并确认 "
            "$ASCEND_HOME/python/site-packages 在 PYTHONPATH 中"
        ) from exc


def _require_env() -> None:
    """校验 CANN 运行库是否可见，避免出现难以定位的段错误。"""
    if not os.environ.get("ASCEND_HOME_PATH") and not os.environ.get("LD_LIBRARY_PATH"):
        raise RuntimeError(f"未检测到 CANN 环境变量，请先执行: {SET_ENV_HINT}")


@dataclass(frozen=True)
class TensorSpec:
    """模型输入/输出的张量描述。"""

    index: int
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        size = 1
        for dim in self.shape:
            size *= int(dim)
        return size * self.dtype.itemsize


class AclModel:
    """单个 .om 模型的推理会话（非线程安全，需外部串行化）。"""

    def __init__(self, model_path, device_id: int = 0) -> None:
        _require_env()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.device_id = device_id
        self._acl = _acl_module()
        self._owns_context = False
        self._model_id = None
        self._desc = None
        self._context = None
        self._init_runtime()

    # ---------- 初始化 ----------

    def _init_runtime(self) -> None:
        acl = self._acl
        _check(acl.init(), "acl.init")
        _check(acl.rt.set_device(self.device_id), "acl.rt.set_device")
        # 复用已有 context；没有则创建并由本对象负责销毁
        context, ret = acl.rt.get_context()
        if ret != 0 or not context:
            context, ret = acl.rt.create_context(self.device_id)
            _check(ret, "acl.rt.create_context")
            self._owns_context = True
        self._context = context

        self._model_id, ret = acl.mdl.load_from_file(str(self.model_path))
        _check(ret, f"acl.mdl.load_from_file({self.model_path})")
        self._desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(self._desc, self._model_id), "acl.mdl.get_desc")

        self.inputs = self._read_specs(is_input=True)
        self.outputs = self._read_specs(is_input=False)
        LOGGER.info(
            "已加载模型 %s 输入=%s 输出=%s",
            self.model_path.name,
            [(s.name, s.shape) for s in self.inputs],
            [(s.name, s.shape) for s in self.outputs],
        )

    def _read_specs(self, is_input: bool) -> list[TensorSpec]:
        acl = self._acl
        count = (
            acl.mdl.get_num_inputs(self._desc) if is_input else acl.mdl.get_num_outputs(self._desc)
        )
        specs: list[TensorSpec] = []
        for index in range(count):
            if is_input:
                raw_shape, ret = acl.mdl.get_input_dims(self._desc, index)
                raw_name = acl.mdl.get_input_name_by_index(self._desc, index)
                code = acl.mdl.get_input_data_type(self._desc, index)
            else:
                raw_shape, ret = acl.mdl.get_output_dims(self._desc, index)
                raw_name = acl.mdl.get_output_name_by_index(self._desc, index)
                code = acl.mdl.get_output_data_type(self._desc, index)
            _check(ret, "acl.mdl.get_dims")
            dims = tuple(int(d) for d in raw_shape.get("dims", []) if int(d) > 0)
            dtype = np.dtype(_DTYPE_MAP.get(int(code), np.float32))
            specs.append(
                TensorSpec(
                    index=index,
                    name=str(raw_name) if raw_name else f"tensor_{index}",
                    shape=dims,
                    dtype=dtype,
                )
            )
        return specs

    # ---------- 推理 ----------

    def infer(self, feed: np.ndarray) -> list[np.ndarray]:
        """执行一次前向。feed 为输入张量（单输入模型）或输入列表。"""
        feeds = [feed] if isinstance(feed, np.ndarray) else list(feed)
        if len(feeds) != len(self.inputs):
            raise ValueError(f"输入数量不匹配: 需要 {len(self.inputs)}，实际 {len(feeds)}")
        acl = self._acl
        dataset = acl.mdl.create_dataset()
        in_buffers: list[tuple[int, int]] = []
        out_dataset = acl.mdl.create_dataset()
        out_buffers: list[tuple[int, int]] = []
        try:
            for spec, data in zip(self.inputs, feeds):
                arr = np.ascontiguousarray(data, dtype=spec.dtype)
                if spec.shape and tuple(arr.shape) != spec.shape:
                    raise ValueError(
                        f"输入 {spec.name} 形状不匹配: 需要 {spec.shape}，实际 {arr.shape}"
                    )
                nbytes = int(arr.nbytes)
                dev_ptr, ret = acl.rt.malloc(nbytes, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, "acl.rt.malloc(input)")
                in_buffers.append((dev_ptr, nbytes))
                _check(
                    acl.rt.memcpy(
                        dev_ptr, nbytes, arr.ctypes.data, nbytes, ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "acl.rt.memcpy(h2d)",
                )
                acl.mdl.add_dataset_buffer(dataset, acl.create_data_buffer(dev_ptr, nbytes))

            for spec in self.outputs:
                nbytes = spec.nbytes
                dev_ptr, ret = acl.rt.malloc(nbytes, ACL_MEM_MALLOC_HUGE_FIRST)
                _check(ret, "acl.rt.malloc(output)")
                out_buffers.append((dev_ptr, nbytes))
                acl.mdl.add_dataset_buffer(
                    out_dataset, acl.create_data_buffer(dev_ptr, nbytes)
                )

            _check(
                acl.mdl.execute(self._model_id, dataset, out_dataset), "acl.mdl.execute"
            )

            results: list[np.ndarray] = []
            for spec, (dev_ptr, nbytes) in zip(self.outputs, out_buffers):
                host = np.empty(nbytes, dtype=np.uint8)
                _check(
                    acl.rt.memcpy(
                        host.ctypes.data,
                        nbytes,
                        dev_ptr,
                        nbytes,
                        ACL_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "acl.rt.memcpy(d2h)",
                )
                results.append(host.view(spec.dtype).reshape(spec.shape))
            return results
        finally:
            acl.mdl.destroy_dataset(dataset)
            acl.mdl.destroy_dataset(out_dataset)
            for dev_ptr, _ in in_buffers + out_buffers:
                acl.rt.free(dev_ptr)

    # ---------- 释放 ----------

    def close(self) -> None:
        acl = self._acl
        if self._model_id is not None:
            acl.mdl.unload(self._model_id)
            self._model_id = None
        if self._desc is not None:
            acl.mdl.destroy_desc(self._desc)
            self._desc = None
        if self._context is not None and self._owns_context:
            acl.rt.destroy_context(self._context)
        self._context = None
        try:
            acl.rt.reset_device(self.device_id)
        except Exception:  # pragma: no cover - 释放阶段尽量不抛
            LOGGER.debug("reset_device 失败，忽略", exc_info=True)

    def __enter__(self) -> "AclModel":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()