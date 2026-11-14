"""端到端延迟基准：分别测量检测、分类、鸟鸣三条链路的耗时。

用法（板上）：
    PYTHONPATH=src python scripts/benchmark.py --rounds 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from birdid.classifier import SpeciesRegistry
from birdid.config import load_config
from birdid.image_pipeline import ImagePipeline


def _percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。"""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(ratio * (len(ordered) - 1)))))
    return ordered[index]


def _summarise(name: str, samples: list[float]) -> str:
    """输出一行统计。"""
    return (
        f"{name}: mean={statistics.mean(samples):.1f}ms "
        f"p50={_percentile(samples, 0.5):.1f}ms "
        f"p90={_percentile(samples, 0.9):.1f}ms "
        f"min={min(samples):.1f}ms max={max(samples):.1f}ms"
    )


def bench_image(rounds: int, image_path: Path | None, config_path: str | None) -> None:
    """测量图片链路。"""
    from PIL import Image

    config = load_config(config_path)
    registry = SpeciesRegistry(config.resolve(config.npu.classes_file))
    pipeline = ImagePipeline(config, registry)
    if image_path is None:
        array = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
        image = Image.fromarray(array)
    else:
        image = Image.open(image_path)
        image.load()
    pipeline.predict(image)  # 预热，剔除首次加载开销
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter()
        pipeline.predict(image)
        samples.append((time.perf_counter() - started) * 1000.0)
    print(_summarise("image", samples))
    pipeline.close()


def bench_sound(rounds: int, config_path: str | None) -> None:
    """测量鸟鸣链路（3 秒 32kHz 合成音）。"""
    from birdid.sound_pipeline import SoundPipeline

    config = load_config(config_path)
    pipeline = SoundPipeline(config)
    rate = 32000
    timeline = np.arange(rate * 3) / rate
    signal = (0.2 * np.sin(2 * np.pi * 3000 * timeline)).astype(np.float32)
    frames = (signal * 32767).astype("<i2").tobytes()
    header = b"RIFF" + (36 + len(frames)).to_bytes(4, "little") + b"WAVEfmt "
    header += (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    header += (1).to_bytes(2, "little") + rate.to_bytes(4, "little")
    header += (rate * 2).to_bytes(4, "little") + (2).to_bytes(2, "little")
    header += (16).to_bytes(2, "little") + b"data" + len(frames).to_bytes(4, "little")
    raw = header + frames
    pipeline.predict(raw, None, None, None)
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter()
        pipeline.predict(raw, None, None, None)
        samples.append((time.perf_counter() - started) * 1000.0)
    print(_summarise("sound", samples))
    pipeline.close()


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="birdid 延迟基准")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", choices=["image", "sound"], default=None)
    args = parser.parse_args(argv)
    if args.only in (None, "image"):
        bench_image(args.rounds, args.image, args.config)
    if args.only in (None, "sound"):
        bench_sound(args.rounds, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())