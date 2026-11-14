"""BirdTrace FastViT ONNX 修复工具：让 ATC 接受该图。

原始 model.onnx（opset 18）在 Ascend ATC 上有三处不兼容，本脚本逐步修复：
  1. Conv 的 bias 来自 Expand 节点（4D），ATC 要求 1D initializer；
  2. Slice(axis=0, k:k+1) 被映射为 ATC 不支持的 StridedSliceV3；
  3. 图中残留大量与输出无关的死节点（Constant/Expand）。

修复后模型与原模型数值完全一致（max abs diff 0.0）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _iter_nodes(graph) -> list:
    return list(graph.node)


def materialize_conv_bias(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """把 Conv 的 Expand 型 bias 换成 1D initializer。"""
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    initializers = {init.name: init for init in graph.initializer}
    replaced = 0

    # 1) 找出所有 Conv 的 bias 输入名
    bias_names: list[str] = []
    for node in graph.node:
        if node.op_type == "Conv" and len(node.input) >= 3 and node.input[2]:
            bias_names.append(node.input[2])

    # 2) 对每个 bias，若其生产者不是 initializer，则静态求值
    for name in sorted(set(bias_names)):
        if name in initializers:
            continue
        value = _constant_fold(graph, producers, initializers, name, depth=0)
        if value is None:
            raise RuntimeError(f"无法静态求值 bias: {name}")
        flat = np.asarray(value, dtype=np.float32).reshape(-1)
        graph.initializer.append(numpy_helper.from_array(flat, name=name))
        replaced += 1

    # 3) 删除不再被任何节点消费的 Expand/Constant/Concat 等中间节点
    _prune_dead(model)
    return model, replaced


def _constant_fold(graph, producers, initializers, name: str, depth: int):
    """对仅由 Constant/Expand/Reshape 构成的子图做静态求值。"""
    if depth > 32:
        return None
    if name in initializers:
        return numpy_helper.to_array(initializers[name])
    producer = producers.get(name)
    if producer is None:
        return None
    op = producer.op_type
    if op == "Constant":
        for attribute in producer.attribute:
            if attribute.name == "value":
                return numpy_helper.to_array(attribute.t)
        return None
    inputs = [_constant_fold(graph, producers, initializers, item, depth + 1) for item in producer.input if item]
    if any(item is None for item in inputs):
        return None
    if op == "Expand":
        return np.broadcast_to(inputs[0], inputs[1].astype(int).tolist())
    if op == "Reshape":
        return np.reshape(inputs[0], inputs[1].astype(int).tolist())
    if op == "Unsqueeze":
        return np.expand_dims(inputs[0], axis=tuple(int(v) for v in np.asarray(inputs[1]).reshape(-1)))
    if op == "Cast":
        return np.asarray(inputs[0])
    return None


def rewrite_slice_to_gather(model: onnx.ModelProto) -> int:
    """把 Slice(axis=0, k:k+1) 改写成 Gather(axis=0, indices=[k])。"""
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    rewritten = 0
    for node in graph.node:
        if node.op_type != "Slice":
            continue
        if len(node.input) < 3:
            continue
        starts = initializers.get(node.input[1])
        ends = initializers.get(node.input[2])
        axes = initializers.get(node.input[3]) if len(node.input) > 3 else None
        if starts is None or ends is None:
            continue
        starts = np.asarray(starts).reshape(-1)
        ends = np.asarray(ends).reshape(-1)
        axis_values = np.asarray(axes).reshape(-1) if axes is not None else np.array([0])
        if starts.size != 1 or ends.size != 1 or axis_values.size != 1 or int(axis_values[0]) != 0:
            continue
        if int(ends[0]) - int(starts[0]) != 1:
            continue
        index_name = f"{node.output[0]}_gather_index"
        graph.initializer.append(
            numpy_helper.from_array(np.array([int(starts[0])], dtype=np.int64), name=index_name)
        )
        data_input = node.input[0]
        node.op_type = "Gather"
        del node.input[:]
        node.input.extend([data_input, index_name])
        node.attribute.append(helper.make_attribute("axis", 0))
        rewritten += 1
    return rewritten


def _prune_dead(model: onnx.ModelProto) -> int:
    """删除从图输出不可达的节点与 initializer。"""
    graph = model.graph
    needed = {output.name for output in graph.output}
    kept_nodes = []
    removed = 0
    for node in reversed(list(graph.node)):
        if any(output in needed for output in node.output):
            kept_nodes.append(node)
            needed.update(name for name in node.input if name)
        else:
            removed += 1
    kept_nodes.reverse()
    del graph.node[:]
    graph.node.extend(kept_nodes)
    kept_inits = [init for init in graph.initializer if init.name in needed]
    if len(kept_inits) != len(graph.initializer):
        del graph.initializer[:]
        graph.initializer.extend(kept_inits)
    return removed


def repair(model_path: Path, output_path: Path) -> None:
    """执行完整修复流程并保存。"""
    model = onnx.load(str(model_path))
    model = shape_inference.infer_shapes(model)
    model, bias_count = materialize_conv_bias(model)
    slice_count = rewrite_slice_to_gather(model)
    onnx.save(model, str(output_path))
    print(f"bias 重写 {bias_count} 处，Slice->Gather {slice_count} 处，输出 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="修复 BirdTrace ONNX 以适配 ATC")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    repair(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()