#!/usr/bin/env bash
# 把 ONNX 转成昇腾 .om（soc_version=Ascend310B1）。
#
# 用法：scripts/convert_om.sh <onnx 文件> <输出前缀> [--input-shape image:1,3,384,384]
#
# BirdTrace FastViT 的 ONNX 需要先做三步图修复（见 scripts/repair_birdtrace_onnx.py）：
#   1. 4D bias 的 Conv 改写成 1D initializer；
#   2. Slice(axis=0, k:k+1) 改写成 Gather；
#   3. 剪除与输出无关的死节点。
set -euo pipefail

CANNN_SET_ENV=${CANNN_SET_ENV:-/usr/local/Ascend/cann-9.2.1/set_env.sh}
SOC_VERSION=${SOC_VERSION:-Ascend310B1}

if [ $# -lt 2 ]; then
  echo "用法: $0 <onnx 文件> <输出前缀> [atc 额外参数...]" >&2
  exit 2
fi

MODEL=$1
OUTPUT=$2
shift 2

if [ -f "${CANNN_SET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${CANNN_SET_ENV}"
fi

echo "[birdid] atc 转换 $(basename "${MODEL}") -> ${OUTPUT}.om (soc=${SOC_VERSION})"
atc \
  --model="${MODEL}" \
  --framework=5 \
  --output="${OUTPUT}" \
  --soc_version="${SOC_VERSION}" \
  --log=error \
  "$@"

echo "[birdid] 完成: ${OUTPUT}.om ($(stat -c %s "${OUTPUT}.om") 字节)"