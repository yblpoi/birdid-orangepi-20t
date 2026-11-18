"""板上自检：确认鸟鸣链路真的全在 NPU 上跑，并给出延迟 / 吞吐 / 泄漏三项基准。

原来的四个一次性探测脚本（placement 检查、窗口数扫描、并发压测、长时间 soak）合并到这里，
以后换板子或改模型只跑这一个文件就够。

用法（板上）：
    PYTHONPATH=src python scripts/check_npu.py                # 跑全部
    PYTHONPATH=src python scripts/check_npu.py --only placement latency
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from birdid.config import load_config
from birdid.sound_pipeline import SoundPipeline

SAMPLE_RATE = 32000
WINDOW_SECONDS = 3
WINDOW_COUNTS = (1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 20, 24, 27)


def _synthesise(raw: bytes, windows: int) -> bytes:
    """把一段真实录音按整数倍拼长，凑出指定窗口数的 WAV 字节。"""
    with wave.open(io.BytesIO(raw), "rb") as handle:
        data = handle.readframes(handle.getnframes())
    span = int(SAMPLE_RATE * WINDOW_SECONDS) * 2
    buffer = bytearray()
    while len(buffer) < span * windows:
        buffer += data
    payload = bytes(buffer[: span * windows])
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(payload)
    return out.getvalue()


def check_placement(pipeline: SoundPipeline) -> bool:
    """确认没有任何 ORT 会话还在承接计算，返回是否全部落在 NPU 上。"""
    birdnet = pipeline._birdnet
    geo = pipeline._geo
    mel = birdnet._mel_npu
    print("== 设备归属 ==")
    print(f"birdnet backend        : {birdnet.backend}")
    print(f"  整图主干 ORT 会话    : {'None（好）' if birdnet._session is None else '存在（在跑 CPU！）'}")
    print(f"  前端 ORT 会话        : {'None（好）' if birdnet._front is None else '存在（在跑 CPU！）'}")
    print(f"  mel 前端             : {mel.backend if mel else None}")
    print(f"  主干 AclModel        : {'有' if birdnet._trunk_acl else '无'}")
    print(f"  head 批量档          : {[head.size for head in birdnet._heads]}")
    print(f"geo backend            : {geo.backend}")
    print(f"  geo ORT 会话         : {'None（好）' if geo._session is None else '存在（在跑 CPU！）'}")
    pure = (
        birdnet._session is None
        and birdnet._front is None
        and geo._session is None
        and mel is not None
        and mel.backend == "npu"
        and geo.backend == "npu"
    )
    print(f"=> 全 NPU：{'是' if pure else '否'}")
    return pure


def bench_latency(pipeline: SoundPipeline, raw: bytes, rounds: int) -> None:
    """按窗口数扫一遍端到端延迟。"""
    print("== 延迟（每个 3 秒窗口）==")
    for windows in WINDOW_COUNTS:
        payload = _synthesise(raw, windows)
        for _ in range(3):
            pipeline.predict(payload)
        samples: list[float] = []
        for _ in range(rounds):
            started = time.perf_counter()
            pipeline.predict(payload)
            samples.append((time.perf_counter() - started) * 1000.0)
        samples.sort()
        # 合成音是单频正弦，识别不出鸟属正常；这里只关心延迟，标签可有可无。
        result = pipeline.predict(payload)
        top = result[0]["list"] if result else []
        name = top[0][1].split("|")[-1] if top else "-"
        median = samples[len(samples) // 2]
        print(f"N={windows:2d}  min={samples[0]:7.2f}  med={median:7.2f} ms   {name}")


def bench_concurrency(pipeline: SoundPipeline, raw: bytes, levels: list[int]) -> None:
    """进程内多线程压测，验证吞吐是否被单卡串行锁死。"""
    print("== 并发（12 秒音频）==")
    for _ in range(3):
        pipeline.predict(raw)
    for level in levels:
        latencies = [0.0] * level

        def work(index: int) -> None:
            started = time.perf_counter()
            pipeline.predict(raw)
            latencies[index] = (time.perf_counter() - started) * 1000.0

        threads = [threading.Thread(target=work, args=(i,)) for i in range(level)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = (time.perf_counter() - started) * 1000.0
        ordered = sorted(latencies)
        print(
            f"{level} 线程: 中位 {ordered[len(ordered) // 2]:.0f} ms  "
            f"最大 {ordered[-1]:.0f} ms  吞吐 {level / (wall / 1000.0):.2f} req/s"
        )


def bench_soak(pipeline: SoundPipeline, raw: bytes, rounds: int) -> None:
    """连跑若干次，看有没有内存泄漏与标签漂移。"""
    print(f"== 稳定性（{rounds} 次）==")
    import resource

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    baseline: list[str] | None = None
    started = time.perf_counter()
    for index in range(rounds):
        names = [entry[1].split("|")[-1] for entry in pipeline.predict(raw)[0]["list"]]
        if baseline is None:
            baseline = names
        elif names != baseline:
            print(f"  第 {index} 次标签漂移：{names}")
            baseline = names
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  平均 {elapsed / rounds * 1000:.1f} ms/次，RSS {before} → {after} KB（+{(after - before) / 1024:.1f} MB）")
    print(f"  最终标签 {baseline}")


def main(argv: list[str] | None = None) -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description="birdid 板上自检")
    parser.add_argument("--config", default=None)
    parser.add_argument("--audio", type=Path, default=None, help="真实录音；缺省用 3 秒合成音")
    parser.add_argument("--rounds", type=int, default=8, help="延迟基准每档的重复次数")
    parser.add_argument("--soak", type=int, default=60, help="soak 的重复次数")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["placement", "latency", "concurrency", "soak"],
        default=None,
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pipeline = SoundPipeline(config)
    chosen = set(args.only or ["placement", "latency", "concurrency", "soak"])
    ok = True

    if args.audio is not None and args.audio.exists():
        raw = args.audio.read_bytes()
    else:
        timeline = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        signal = (0.2 * np.sin(2 * np.pi * 3000 * timeline)).astype(np.float32)
        frames = (signal * 32767).astype("<i2").tobytes()
        out = io.BytesIO()
        with wave.open(out, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(frames)
        raw = out.getvalue()

    try:
        if "placement" in chosen:
            ok = check_placement(pipeline) and ok
        if "latency" in chosen:
            bench_latency(pipeline, raw, args.rounds)
        if "concurrency" in chosen:
            bench_concurrency(pipeline, raw, [1, 2, 4, 8])
        if "soak" in chosen:
            bench_soak(pipeline, raw, args.soak)
    finally:
        pipeline.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
