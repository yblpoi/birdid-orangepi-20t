#!/bin/bash
# 在昇腾开发板上启动 birdid 服务。所有路径相对仓库根目录。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANNN_SET_ENV="${CANNN_SET_ENV:-/usr/local/Ascend/cann-9.2.1/set_env.sh}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${BIRDID_CONFIG:-$REPO_ROOT/configs/default.yaml}"

if [ -f "$CANNN_SET_ENV" ]; then
  # shellcheck disable=SC1090
  source "$CANNN_SET_ENV"
else
  echo "warn: 未找到 CANN 环境脚本 $CANNN_SET_ENV" >&2
fi

# pyACL 的 Python 包不在默认 sys.path 上，需要显式加入。
export PYTHONPATH="$REPO_ROOT/src:${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.1}/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m birdid --config "$CONFIG"