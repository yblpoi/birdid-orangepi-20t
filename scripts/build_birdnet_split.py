"""从 BirdNET v3 的 front/back 两段 ONNX 生成「全 NPU」链路需要的中间 ONNX。

服务实际用的是这套产物（见 docs/deploy.md 4.3）：

    models/birdnet_v3_fp16_mel.om             mel 前端（单窗口）
    models/birdnet_v3_fp16_mel_b4.om          mel 前端（4 窗口批量）
    models/birdnet_v3_fp16_trunk.om           卷积主干
    models/birdnet_v3_fp16_head_b{4,8,12}.om  打分头（批量）

它们的来历都不直观，所以集中在这一个脚本里：

1. **mel 前端**：原始 front ONNX 用 STFT 算子，CANN 没有对应插件（E13010），
   无法转 .om。这里把「加窗 + rfft + 取模平方 + mel 投影」解析展开：
   加窗融化进 DFT 矩阵，只保留真正被 mel 矩阵用到的 bin，最终只剩两次 MatMul。
2. **trunk / head**：按 mel 之后的张量边界把 CNN 主干切成两段。head 里那四个
   [11560, 2048, 1] 的 Conv 本来是逐帧点积，改写成 MatMul 后才能真批量。
   批量档 b4/b8/b12 的 .om 分别喂 4/8/12 个窗口，服务按窗口数挑最优组合。

用法::

    python scripts/build_birdnet_split.py \\
        --front models/birdnet_v3_fp16_front.onnx \\
        --back models/birdnet_v3_fp16_back_ir7.onnx \\
        --out-dir models

生成完再用 scripts/convert_om.sh 逐个转 .om（见 docs/deploy.md）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

FLEN = 2048   # STFT 窗长
HOP = 512     # 帧移
FRAMES = 188  # 一个 3 秒窗口的帧数

# 主干在 mel 之后按这批张量切开
CUT_OUT = ["transpose_14", "transpose_17", "mul_2891", "node_mean_31_input_cast_0", "silu_101"]


def _init_map(model: onnx.ModelProto) -> dict:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def build_mel(front_path: Path, out_dir: Path, batch: int) -> None:
    """把 front ONNX 里的 STFT 展开成两次 MatMul。"""
    front = onnx.load(str(front_path))
    init = _init_map(front)
    window = init["node_STFT_39_input_cast_2"].astype(np.float64)   # (2048,)
    matrix = init["node_mm_input_cast_1"].astype(np.float64)        # (1025,128)

    used = np.flatnonzero(np.abs(matrix).sum(axis=1) > 0)
    lo, hi = int(used[0]), int(used[-1]) + 1
    if not np.array_equal(used, np.arange(lo, hi)):
        raise SystemExit("mel 矩阵用到的 bin 不连续，不能直接切片：" + str(lo) + ".." + str(hi - 1))

    ticks = np.arange(FLEN)
    bins = np.arange(lo, hi)
    angle = 2.0 * np.pi * np.outer(ticks, bins) / FLEN
    cos_part = window[:, None] * np.cos(angle)    # 加窗融化进矩阵
    sin_part = -(window[:, None] * np.sin(angle))
    dft = np.concatenate([cos_part, sin_part], axis=1).astype(np.float32)   # (2048, 2*nbins)
    sub = matrix[lo:hi, :].astype(np.float32)                              # (nbins, 128)
    nbins = len(bins)

    def emit(frames: int, path: Path) -> None:
        nodes = [
            helper.make_node("MatMul", ["frames", "D"], ["both"]),
            helper.make_node("Slice", ["both", "s0", "s1", "ax"], ["re"]),
            helper.make_node("Slice", ["both", "s2", "s3", "ax"], ["im"]),
            helper.make_node("Mul", ["re", "re"], ["re2"]),
            helper.make_node("Mul", ["im", "im"], ["im2"]),
            helper.make_node("Add", ["re2", "im2"], ["power"]),
            helper.make_node("MatMul", ["power", "Msub"], ["mel"]),
        ]
        inits = [
            numpy_helper.from_array(dft, "D"),
            numpy_helper.from_array(sub, "Msub"),
            numpy_helper.from_array(np.array([0], np.int64), "s0"),
            numpy_helper.from_array(np.array([nbins], np.int64), "s1"),
            numpy_helper.from_array(np.array([nbins], np.int64), "s2"),
            numpy_helper.from_array(np.array([2 * nbins], np.int64), "s3"),
            numpy_helper.from_array(np.array([1], np.int64), "ax"),
        ]
        graph = helper.make_graph(
            nodes,
            "mel_dft_b" + str(frames // FRAMES),
            [helper.make_tensor_value_info("frames", TensorProto.FLOAT, [frames, FLEN])],
            [helper.make_tensor_value_info("mel", TensorProto.FLOAT, [frames, 128])],
            inits,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        print("[mel] " + path.name + ": frames=(" + str(frames) + "," + str(FLEN) + ") "
              "bins=" + str(lo) + ".." + str(hi - 1) + "  大小 " + ("%.1f" % (path.stat().st_size / 1e6)) + " MB")

    emit(FRAMES, out_dir / "mel_dft.onnx")
    if batch >= 2:
        emit(FRAMES * batch, out_dir / ("mel_dft_b" + str(batch) + ".onnx"))

    check_mel(front_path, dft, sub, nbins)


def check_mel(front_path: Path, dft: np.ndarray, sub: np.ndarray, nbins: int) -> None:
    """用 float64 展开式对齐 ORT 前端，确认展开没有语义漂移。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[mel] 跳过数值校验（没装 onnxruntime）")
        return

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(96000).astype(np.float32)   # 3 s @ 32 kHz，正好 188 帧
    padded = np.pad(signal, (1024, 1024), mode="reflect")
    index = (np.arange(FLEN)[None, :] + HOP * np.arange(FRAMES)[:, None]).astype(np.int32)
    frames = padded[index]

    both = frames.astype(np.float64) @ dft.astype(np.float64)
    power = both[:, :nbins] ** 2 + both[:, nbins:] ** 2
    mel = power @ sub.astype(np.float64)

    session = ort.InferenceSession(str(front_path), providers=["CPUExecutionProvider"])
    ref = session.run(None, {session.get_inputs()[0].name: signal[None, :]})[0]
    diff = float(np.abs(mel - ref).max())
    print("[mel] 展开式 vs ORT 前端最大绝对差 " + ("%.2e" % diff))


def build_trunk_head(back_path: Path, out_dir: Path, batches: list) -> None:
    """按可达性把 CNN 主干切成 trunk / head，再把 head 的 Conv 改写成 MatMul。"""
    model = onnx.load(str(back_path))
    graph = model.graph
    nodes = list(graph.node)
    producer = {out: idx for idx, node in enumerate(nodes) for out in node.output}

    need = set(CUT_OUT)
    trunk_idx = set()
    changed = True
    while changed:
        changed = False
        for idx, node in enumerate(nodes):
            if idx in trunk_idx or not any(out in need for out in node.output):
                continue
            trunk_idx.add(idx)
            changed = True
            need.update(name for name in node.input if name)

    head_idx = [idx for idx in range(len(nodes)) if idx not in trunk_idx]
    frontier = []
    for idx in head_idx:
        for name in nodes[idx].input:
            if not name or name in frontier:
                continue
            src = producer.get(name)
            if src is not None and src in trunk_idx:
                frontier.append(name)

    inferred = shape_inference.infer_shapes(onnx.load(str(back_path)), strict_mode=False)
    shapes = {}
    for item in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shapes[item.name] = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
    for name in frontier:
        if name not in shapes:
            raise SystemExit("切分边界 " + name + " 推不出形状")

    used_in = set()
    for idx in trunk_idx:
        used_in.update(nodes[idx].input)
    trunk_graph = helper.make_graph(
        [nodes[i] for i in sorted(trunk_idx)],
        "trunk",
        [graph.input[0]],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shapes[name]) for name in frontier],
        [item for item in graph.initializer if item.name in used_in],
    )
    trunk = helper.make_model(trunk_graph, opset_imports=list(model.opset_import))
    trunk.ir_version = 8
    onnx.checker.check_model(trunk)
    onnx.save(trunk, str(out_dir / "trunk.onnx"))
    print("[trunk] " + str(len(trunk_idx)) + " 节点，边界 " + str(frontier))

    used_in = set()
    for idx in head_idx:
        used_in.update(nodes[idx].input)
    head_graph = helper.make_graph(
        [nodes[i] for i in head_idx],
        "head",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT16, shapes[name]) for name in frontier],
        [helper.make_tensor_value_info(item.name, item.type.tensor_type.elem_type,
                                       [dim.dim_value for dim in item.type.tensor_type.shape.dim])
         for item in graph.output],
        [item for item in graph.initializer if item.name in used_in],
    )
    head = helper.make_model(head_graph, opset_imports=list(model.opset_import))
    head.ir_version = 8
    onnx.checker.check_model(head)
    head_path = out_dir / "head.onnx"
    onnx.save(head, str(head_path))
    print("[head] " + str(len(head_idx)) + " 节点")

    for batch in batches:
        emit_head(head_path, batch, out_dir / ("head_b" + str(batch) + ".onnx"), shapes)


def emit_head(head_path: Path, batch: int, path: Path, shapes: dict) -> None:
    """把 head 里的 Conv(11560,2048,1) 换成 MatMul，让批维真正生效。"""
    model = onnx.load(str(head_path))
    graph = model.graph
    inits = {item.name: item for item in graph.initializer}
    nodes = []
    arrays = {}

    for node in graph.node:
        if node.op_type != "Conv":
            nodes.append(node)
            for name in node.input:
                if name in inits and name not in arrays:
                    arrays[name] = inits[name]
            continue

        weight = numpy_helper.to_array(inits[node.input[1]])   # [11560, 2048, 1]
        bias = numpy_helper.to_array(inits[node.input[2]])     # [11560]
        out_channels, in_channels = weight.shape[0], weight.shape[1]
        length = shapes[node.input[0]][2]
        tag = node.output[0]
        arrays[tag + "_W"] = numpy_helper.from_array(
            np.ascontiguousarray(weight.reshape(out_channels, in_channels).T).astype(np.float16), tag + "_W")
        arrays[tag + "_B"] = numpy_helper.from_array(bias.astype(np.float16), tag + "_B")
        arrays[tag + "_2d"] = numpy_helper.from_array(np.array([batch * length, in_channels], np.int64), tag + "_2d")
        arrays[tag + "_3d"] = numpy_helper.from_array(np.array([batch, length, out_channels], np.int64), tag + "_3d")

        nodes.extend([
            helper.make_node("Transpose", [node.input[0]], [tag + "_T"], perm=[0, 2, 1]),
            helper.make_node("Reshape", [tag + "_T", tag + "_2d"], [tag + "_F"]),
            helper.make_node("MatMul", [tag + "_F", tag + "_W"], [tag + "_M"]),
            helper.make_node("Add", [tag + "_M", tag + "_B"], [tag + "_A"]),
            helper.make_node("Reshape", [tag + "_A", tag + "_3d"], [tag + "_3"]),
            helper.make_node("Transpose", [tag + "_3"], [tag], perm=[0, 2, 1]),
        ])

    inputs = []
    for item in graph.input:
        dims = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        dims[0] = batch
        inputs.append(helper.make_tensor_value_info(item.name, item.type.tensor_type.elem_type, dims))
    outputs = []
    for item in graph.output:
        dims = [dim.dim_value for dim in item.type.tensor_type.shape.dim]
        dims[0] = batch
        outputs.append(helper.make_tensor_value_info(item.name, item.type.tensor_type.elem_type, dims))

    built = helper.make_graph(nodes, "head_b" + str(batch), inputs, outputs, list(arrays.values()))
    model = helper.make_model(built, opset_imports=list(model.opset_import))
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    print("[head] " + path.name + ": " + str(len(nodes)) + " 节点，批量 " + str(batch))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", default="models/birdnet_v3_fp16_front.onnx")
    parser.add_argument("--back", default="models/birdnet_v3_fp16_back_ir7.onnx")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 12],
                        help="head 的批量档，服务按窗口数挑最优组合")
    parser.add_argument("--mel-batch", type=int, default=4,
                        help="mel 前端的批量档，与服务里的 birdnet_mel_batch_om 对应")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_mel(Path(args.front), out_dir, args.mel_batch)
    build_trunk_head(Path(args.back), out_dir, args.batches)
    print("")
    print("下一步：用 scripts/convert_om.sh 把生成的 ONNX 逐个转成 .om（见 docs/deploy.md 4.3）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
