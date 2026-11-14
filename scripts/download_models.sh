#!/usr/bin/env bash
# 下载模型权重。所有 URL 可通过环境变量覆盖，默认走国内镜像。
#
# 用法：scripts/download_models.sh [目标目录]
#
# 环境变量：
#   BIRDID_MODEL_DIR    权重输出目录（默认 models）
#   HF_MIRROR           Hugging Face 镜像（默认 https://hf-mirror.com）
#   GH_PROXY            GitHub 加速前缀（默认 https://gh-proxy.com/）
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
TARGET_DIR=${1:-${BIRDID_MODEL_DIR:-${REPO_ROOT}/models}}
HF_MIRROR=${HF_MIRROR:-https://hf-mirror.com}
GH_PROXY=${GH_PROXY:-https://gh-proxy.com/}

mkdir -p "${TARGET_DIR}"
echo "[birdid] 权重目录: ${TARGET_DIR}"

# 镜像下载失败时回退直连。
fetch() {
  local url=$1
  local out=$2
  if [ -s "${out}" ]; then
    echo "[birdid] 已存在，跳过: $(basename "${out}")"
    return 0
  fi
  echo "[birdid] 下载: ${url}"
  if ! curl -fSL --retry 5 --retry-delay 3 -C - -o "${out}.part" "${url}"; then
    echo "[birdid] 镜像失败，尝试 GitHub 加速" >&2
    curl -fSL --retry 5 --retry-delay 3 -C - -o "${out}.part" "${GH_PROXY}${url}"
  fi
  mv "${out}.part" "${out}"
}

# BirdTrace-FastViT-MA36-10K：鸟类分类（10,480 类）
BIRDTRACE_REPO="https://huggingface.co/warden123/BirdTrace-FastViT-MA36-10K/resolve/main"
fetch "${HF_MIRROR}/warden123/BirdTrace-FastViT-MA36-10K/resolve/main/model.onnx" \
  "${TARGET_DIR}/birdtrace.model.onnx"
fetch "${HF_MIRROR}/warden123/BirdTrace-FastViT-MA36-10K/resolve/main/classes.json" \
  "${TARGET_DIR}/birdtrace_classes.json"

# YOLO26m-seg：门禁与裁剪（需本地用 ultralytics 导出 ONNX，再 ATC 转 .om）
echo "[birdid] YOLO26m-seg 请用 scripts/export_yolo.py 导出，避免直接下载未校验权重。"

# BirdNET v3 声学模型（CC BY-NC-SA 4.0）
BIRDNET_URL="https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_FP16.onnx"
fetch "${BIRDNET_URL}" "${TARGET_DIR}/birdnet_v3_fp16.onnx"
fetch "https://zenodo.org/records/20703646/files/BirdNET+_V3.0-preview3.1_Global_11K_Labels.csv" \
  "${TARGET_DIR}/birdnet_v3_labels.csv"

# GeoModel v3.0.4 地理/季节先验（Apache-2.0）
GEOMODEL_URL="https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/_Geomodel_V3.0.4_Global_14K_FP32.onnx"
fetch "${GEOMODEL_URL}" "${TARGET_DIR}/geomodel_v3_fp32.onnx"
fetch "https://github.com/birdnet-team/geomodel/releases/download/v3.0.4/_Geomodel_V3.0.4_Global_14K_Labels.txt" \
  "${TARGET_DIR}/geomodel_v3_labels.txt"

echo "[birdid] 权重下载完成。下一步：scripts/convert_om.sh 生成昇腾 .om"