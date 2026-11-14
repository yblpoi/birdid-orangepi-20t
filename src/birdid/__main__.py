"""启动服务：python -m birdid 或 scripts/run_server.sh。"""

from __future__ import annotations

import argparse
import os

from .app import create_app
from .config import load_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，缺省回退到 BIRDID_CONFIG 环境变量。"""
    parser = argparse.ArgumentParser(prog="birdid", description="鸟类识别 API 服务")
    parser.add_argument(
        "--config",
        default=os.environ.get("BIRDID_CONFIG"),
        help="配置文件路径，默认 configs/default.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """读取配置并启动 uvicorn。"""
    import uvicorn

    args = _parse_args(argv)
    config = load_config(args.config)
    uvicorn.run(
        create_app(args.config),
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        access_log=config.logging.access_log,
        workers=1,  # NPU 会话不可跨进程共享，单进程内用线程池并发
    )


if __name__ == "__main__":
    main()