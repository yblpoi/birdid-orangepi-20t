"""导出 YOLO26m-seg 为 ONNX（在 x86/Apple 开发机或任意有 torch 的机器执行）。"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """导出固定 640x640 输入的 seg ONNX。"""
    parser = argparse.ArgumentParser(description="导出 YOLO26-seg ONNX")
    parser.add_argument("--weights", default="yolo26m-seg.pt", help="官方权重文件名")
    parser.add_argument("--output", default="models/yolo26m-seg.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=False,
        nms=False,
        dynamic=False,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(exported).replace(target)
    print(f"已导出 {target} ({target.stat().st_size} 字节)")


if __name__ == "__main__":
    main()