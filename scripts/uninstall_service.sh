#!/bin/bash
# 卸载 birdid 的 systemd 服务：停服务、取消开机自启、删单元文件。
#
#   sudo scripts/uninstall_service.sh            # 保留 /etc/birdid 里的密钥文件
#   sudo scripts/uninstall_service.sh --purge    # 连密钥文件一起删
#
# 仓库代码与模型不受影响。重装用 scripts/install_service.sh。
set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-birdid}
ENV_DIR=${BIRDID_ENV_DIR:-/etc/birdid}
ENV_FILE="$ENV_DIR/$SERVICE_NAME.env"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME.service"

PURGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --purge) PURGE=1; shift ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[birdid] $*"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "[birdid] 错误: 需要 root，请用 sudo 运行" >&2
  exit 1
fi

if systemctl list-unit-files "$SERVICE_NAME.service" >/dev/null 2>&1; then
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "停止 $SERVICE_NAME"
    systemctl stop "$SERVICE_NAME" || true
  fi
  systemctl disable "$SERVICE_NAME" >/dev/null 2>&1 || true
else
  log "服务 $SERVICE_NAME 未安装，继续清理残留文件"
fi

if [ -f "$UNIT_PATH" ]; then
  log "删除 $UNIT_PATH"
  rm -f "$UNIT_PATH"
fi
systemctl daemon-reload
systemctl reset-failed "$SERVICE_NAME" >/dev/null 2>&1 || true

# 服务异常退出后可能留下子进程，一并清掉。
if pgrep -f "python -m birdid --config" >/dev/null 2>&1; then
  log "清理残留的 birdid 进程"
  pkill -f "python -m birdid --config" || true
fi

if [ "$PURGE" -eq 1 ] && [ -f "$ENV_FILE" ]; then
  log "删除 $ENV_FILE"
  rm -f "$ENV_FILE"
  rmdir "$ENV_DIR" 2>/dev/null || true
elif [ -f "$ENV_FILE" ]; then
  log "保留 $ENV_FILE（加 --purge 可一并删除）"
fi

log "已卸载。仓库文件与模型未改动。"