"""启动服务：python -m birdid 或 scripts/serve.sh。"""

from __future__ import annotations

import os

from .app import create_app
from .config import load_config


def main() -> None:
    """读取配置并启动 uvicorn。"""
    import uvicorn

    config_path = os.environ.get("BIRDID_CONFIG")
    config = load_config(config_path)
    uvicorn.run(
        create_app(config_path),
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        access_log=config.logging.access_log,
        workers=1,  # NPU 会话不可跨进程共享，单进程内用线程池并发
    )


if __name__ == "__main__":
    main()