"""裁剪 YOLO26-seg ONNX：只保留检测头 output0，去掉分割原型 output1。

服务只需要检测框（懂鸟规范返回 box + list，不需要 mask），而 output1
(1,32,160,160) 每次推理要拷 3.2 MB 回主机、并驱动一整条 prototype 分支。
310B1 上实测：640x640 单图 114.3 ms -> 101.6 ms，检测结果逐位相同。

用法：
    python scripts/prune_yolo_outputs.py models/yolo26m-seg.onnx models/yolo26m-seg-box.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def prune(model: onnx.ModelProto, keep: tuple[str, ...] = ("output0",)) -> onnx.ModelProto:
    """删除除 keep 之外的图输出，并做反向可达性裁剪。"""
    graph = model.graph
    kept = [o for o in graph.output if o.name in keep]
    if not kept:
        raise ValueError(f"图里没有输出 {keep}，实际输出: {[o.name for o in graph.output]}")
    del graph.output[:]
    graph.output.extend(kept)

    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.output:
            producer[name] = node

    needed: set[int] = set()
    stack = [o.name for o in graph.output]
    while stack:
        name = stack.pop()
        node = producer.get(name)
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        stack.extend(node.input)

    nodes = [n for n in graph.node if id(n) in needed]
    used: set[str] = set()
    for node in nodes:
        used.update(node.input)
    inits = [i for i in graph.initializer if i.name in used]

    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    return model


def main() -> None:
    """命令行入口：裁剪并保存。"""
    parser = argparse.ArgumentParser(description="裁剪 YOLO26-seg ONNX 到只输出检测框")
    parser.add_argument("source", help="原始 seg ONNX（含 output0 与 output1）")
    parser.add_argument("target", help="输出的单输出 ONNX")
    parser.add_argument("--keep", default="output0", help="保留的输出名，逗号分隔")
    args = parser.parse_args()

    keep = tuple(name.strip() for name in args.keep.split(",") if name.strip())
    model = onnx.load(args.source)
    before = len(model.graph.node)
    prune(model, keep)
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    print(
        f"已裁剪 {args.source} -> {target}：节点 {before} -> {len(model.graph.node)}，"
        f"输出 {[o.name for o in model.graph.output]}，{target.stat().st_size} 字节"
    )


if __name__ == "__main__":
    main()
