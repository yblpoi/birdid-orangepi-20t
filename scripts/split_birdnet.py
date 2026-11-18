"""把 BirdNET v3 ONNX 按 mel 边界切成前端与 CNN 主干两段。

为什么需要切分：整图含有 STFT 与 ReduceL2，昇腾 CANN 的 ONNX 算子插件里
没有这两个算子（atc 报 Not_Supported_Operator E13010），整图无法转 .om。
切开后 CNN 主干 653 个节点可以正常转 .om 上 NPU，两段拼接与整图推理结果在 CPU 上逐元素一致。

这一步只负责「按 mel 张量切开」；要把前端也搬上 NPU，接着跑
scripts/build_birdnet_split.py（把 STFT 展开成两次 MatMul，并把主干再切成 trunk + head）。

用法::

    python scripts/split_birdnet.py --input models/birdnet_v3_fp16.onnx \
        --front-out models/birdnet_v3_fp16_front.onnx \
        --back-out models/birdnet_v3_fp16_back.onnx --samples 96000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MEL_TENSOR = "node_mm_output_cast_0"


def _fixed_input(model: onnx.ModelProto, name: str, samples: int) -> onnx.ModelProto:
    """把动态 batch/samples 维度写死，便于后续算子折叠与 atc 指定 shape。"""
    for item in model.graph.input:
        if item.name != name:
            continue
        dims = item.type.tensor_type.shape.dim
        if len(dims) >= 2:
            dims[0].dim_value = 1
            dims[1].dim_value = samples
    return model


def _prune(graph, inputs: list[str], outputs: list[str], value_info, opset_import, ir_version: int):
    """按反向可达性裁剪出以 inputs 为输入、outputs 为输出的子图。"""
    producer = {out: node for node in graph.node for out in node.output}
    keep: set[str] = set()
    todo = list(outputs)
    while todo:
        name = todo.pop()
        if name in keep or name in inputs:
            continue
        keep.add(name)
        node = producer.get(name)
        if node is None:
            continue
        todo.extend([i for i in node.input if i])
    nodes = [n for n in graph.node if any(o in keep for o in n.output)]
    used = {i for n in nodes for i in n.input if i}
    inits = [i for i in graph.initializer if i.name in used]

    def info(name: str):
        value = value_info.get(name)
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim] if value else None
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)

    graph_out = helper.make_graph(nodes, "sub", [info(n) for n in inputs], [info(n) for n in outputs], inits)
    model = helper.make_model(graph_out, opset_imports=opset_import)
    model.ir_version = ir_version
    onnx.checker.check_model(model)
    return model


def split(src: Path, front_out: Path, back_out: Path, samples: int = 96000) -> tuple[int, int, list[int]]:
    """切分模型并返回 (前端节点数, 主干节点数)。"""
    model = _fixed_input(onnx.load(str(src)), "input", samples)
    # 先做常量折叠：整图里的 Shape/Gather/Reshape 会残留动态 shape，
    # 折叠后两段子图都能拿到静态形状，atc 也更稳。
    try:
        from onnxsim import simplify

        model, ok = simplify(
            model,
            overwrite_input_shapes={"input": [1, samples]},
            dynamic_input_shape=False,
            perform_optimization=True,
        )
        if not ok:
            raise RuntimeError("onnxsim 简化失败")
    except ImportError:
        print("未安装 onnxsim，跳过常量折叠（atc 对动态 shape 更敏感）")

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    value_info = {
        v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
    }
    graph = model.graph
    mel_shape = [d.dim_value for d in value_info[MEL_TENSOR].type.tensor_type.shape.dim]
    print(f"mel 张量 {MEL_TENSOR} 形状 {mel_shape}")
    opset_import = list(model.opset_import)
    front = _prune(graph, ["input"], [MEL_TENSOR], value_info, opset_import, model.ir_version)
    back = _prune(
        graph, [MEL_TENSOR], ["predictions", "embeddings_out"], value_info, opset_import, model.ir_version
    )
    onnx.save(front, str(front_out))
    onnx.save(back, str(back_out))
    return len(front.graph.node), len(back.graph.node), mel_shape


def main() -> None:
    parser = argparse.ArgumentParser(description="按 mel 边界切分 BirdNET v3 ONNX")
    parser.add_argument("--input", required=True, help="整图 ONNX（birdnet_v3_fp16.onnx）")
    parser.add_argument("--front-out", required=True, help="前端（wave -> mel）输出路径")
    parser.add_argument("--back-out", required=True, help="主干（mel -> logits）输出路径")
    parser.add_argument("--samples", type=int, default=96000, help="3 秒 32 kHz 的采样点数")
    args = parser.parse_args()
    front, back, shape = split(Path(args.input), Path(args.front_out), Path(args.back_out), args.samples)
    print(f"前端节点 {front}，主干节点 {back}，mel 形状 {shape}")
    print("下一步：atc --model=<back> --input_shape=" + MEL_TENSOR + ":" + ",".join(str(d) for d in shape))


if __name__ == "__main__":
    main()