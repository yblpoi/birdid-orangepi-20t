#!/bin/bash
# 一键把 birdid 装成 systemd 服务（开机自启 + 崩溃自动拉起）。
#
#   sudo scripts/install_service.sh              # 用仓库属主身份运行服务
#   sudo scripts/install_service.sh --user foo   # 指定运行用户
#   sudo scripts/install_service.sh --dry-run    # 只打印将要写入的内容
#
# 装完可用 systemctl status birdid / journalctl -u birdid -f 查看，
# 卸载用 scripts/uninstall_service.sh。
set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-birdid}
ENV_DIR=${BIRDID_ENV_DIR:-/etc/birdid}
ENV_FILE="${ENV_DIR}/${SERVICE_NAME}.env"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="${REPO_ROOT}/scripts/birdid.service.in"

SERVICE_USER=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --user) SERVICE_USER="${2:?--user 需要参数}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

die() { echo "[birdid] 错误: $*" >&2; exit 1; }
log() { echo "[birdid] $*"; }

# ---- 前置检查 -------------------------------------------------------------
command -v systemctl >/dev/null 2>&1 || die "本机没有 systemd，无法安装服务"
[ -f "${TEMPLATE}" ] || die "缺少模板 ${TEMPLATE}"
if [ -z "${SERVICE_USER}" ]; then
  SERVICE_USER="$(stat -c %U "${REPO_ROOT}")"
fi
id "${SERVICE_USER}" >/dev/null 2>&1 || die "用户 ${SERVICE_USER} 不存在"
SERVICE_GROUP="$(id -gn "${SERVICE_USER}")"

if [ "$(id -u)" -ne 0 ] && [ "${DRY_RUN}" -eq 0 ]; then
  die "需要 root，请用 sudo 运行"
fi
if [ "${DRY_RUN}" -eq 0 ] && [ ! -x "${REPO_ROOT}/.venv/bin/python" ] && ! command -v python3 >/dev/null 2>&1; then
  die "既没有 .venv/bin/python 也没有 python3，先按 docs/deploy.md 第 2 节装依赖"
fi

# run_server.sh 里 CANNN_SET_ENV 的默认值；这里只检查不覆盖。
CANNN_SET_ENV=${CANNN_SET_ENV:-/usr/local/Ascend/cann-9.2.1/set_env.sh}
if [ ! -f "${CANNN_SET_ENV}" ]; then
  log "警告: 没找到 CANN 环境脚本 ${CANNN_SET_ENV}，服务可能起不来（可用 CANNN_SET_ENV 覆盖）"
fi

# 渲染模板：只替换三个占位符，不做其它改动。
RENDERED="$(sed -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
                -e "s|@SERVICE_USER@|${SERVICE_USER}|g" \
                -e "s|@SERVICE_GROUP@|${SERVICE_GROUP}|g" "${TEMPLATE}")"

if [ "${DRY_RUN}" -eq 1 ]; then
  log "--dry-run：将写入 ${UNIT_PATH}"
  printf '%s\n' "${RENDERED}"
  log "--dry-run：环境变量模板 ${ENV_FILE}"
  exit 0
fi

# ---- 停掉手工起的旧进程 ---------------------------------------------------
if pgrep -f "python -m birdid --config" >/dev/null 2>&1 && ! systemctl is-active --quiet "${SERVICE_NAME}"; then
  log "发现手工启动的 birdid 进程，先停掉它们"
  pkill -f "python -m birdid --config" || true
  for _ in $(seq 1 20); do
    pgrep -f "python -m birdid --config" >/dev/null 2>&1 || break
    sleep 0.5
  done
fi

# ---- 写单元文件 -----------------------------------------------------------
log "写入 ${UNIT_PATH}（用户 ${SERVICE_USER}:${SERVICE_GROUP}，仓库 ${REPO_ROOT}）"
printf '%s\n' "${RENDERED}" > "${UNIT_PATH}"
chmod 644 "${UNIT_PATH}"

# 密钥只从环境变量读，这里只放一个空模板，权限收紧。
if [ ! -f "${ENV_FILE}" ]; then
  mkdir -p "${ENV_DIR}"
  cat > "${ENV_FILE}" <<'ENVEOF'
# birdid 服务的环境变量。取消注释并填值后 systemctl restart birdid。
# 不填则服务不校验 api_key（内网自用可以直接留空）。
# BIRDID_API_KEY=
ENVEOF
  chmod 600 "${ENV_FILE}"
  log "已生成 ${ENV_FILE}（留空表示不校验 api_key）"
else
  log "沿用已有的 ${ENV_FILE}"
fi

# ---- 启动 -----------------------------------------------------------------
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}" >/dev/null
systemctl restart "${SERVICE_NAME}"

log "等待服务就绪…"
READY=0
for _ in $(seq 1 60); do
  if systemctl is-active --quiet "${SERVICE_NAME}" && curl -fsS --max-time 2 "http://127.0.0.1:8080/healthz" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [ "${READY}" -eq 1 ]; then
  log "服务已就绪：systemctl status ${SERVICE_NAME} / journalctl -u ${SERVICE_NAME} -f"
else
  log "服务已启动但 /healthz 还没响应，看日志：journalctl -u ${SERVICE_NAME} -n 80 --no-pager"
  exit 1
fi