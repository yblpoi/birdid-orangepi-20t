# 部署说明（香橙派 AI Pro 20T / 昇腾 310B1）

## 1. 前置条件

- 已安装 CANN（本项目在 CANN 9.2.1 上验证），环境脚本默认
  `/usr/local/Ascend/cann-9.2.1/set_env.sh`，可用 `CANNN_SET_ENV` 覆盖。
- NNAL 只启用 asdsip（本硬件不支持 atb）。
- Python 3.10 + numpy/pillow/fastapi/uvicorn/python-multipart/pyyaml/onnxruntime。

## 2. 安装依赖

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

国内建议加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

## 3. 准备模型权重

权重不进仓库（见 `.gitignore` 与 `NOTICE`），执行下载脚本：

```bash
BIRDID_MODEL_DIR=models scripts/download_models.sh
```

脚本默认使用 `https://hf-mirror.com` 与 `https://gh-proxy.com` 加速。
YOLO26-seg 需要自行导出：

```bash
python scripts/export_yolo.py --weights yolo26m-seg.pt --output models/yolo26m-seg.onnx
```

## 4. 转 OM 模型（NPU 推理用）

```bash
scripts/convert_om.sh models/yolo26m-seg.onnx models/yolo26m-seg \
    --input_shape 'images:1,3,640,640'
scripts/convert_om.sh models/birdtrace.model.onnx models/fastvit \
    --input_shape 'image:1,3,384,384'
scripts/convert_om.sh models/birdnet_v3_fp16.onnx models/birdnet_v3_fp16 \
    --input_shape 'input:1,96000'
```

BirdTrace 的原始 ONNX 需要先做图修复，否则 ATC 会因 4D bias 的 Conv 报
`input bias shape should be 1D or 4D`：

```bash
python scripts/repair_birdtrace_onnx.py models/birdtrace_raw.onnx models/birdtrace.model.onnx
```

## 5. 启动服务

```bash
BIRDID_API_KEY=... scripts/run_server.sh
```

`scripts/run_server.sh` 会 source CANN 环境、把 `src` 与 pyACL 的 site-packages
加入 `PYTHONPATH`，再以单进程（NPU 会话不可跨进程共享）启动 uvicorn。

## 6. 验证

```bash
curl -s localhost:8080/healthz
curl -s -H 'api_key: ...' -F image=@bird.jpg -F upload=1 localhost:8080/dongniao
```

基准测试：`PYTHONPATH=src python scripts/benchmark.py --rounds 10`。

## 7. 排查

- `npu-smi info` 显示 `Health: Alarm` 是该开发板的硬件设计缺陷，可忽略。
- ATC 转换被强杀后会残留 forkserver 子进程并占满内存，必须清理后再重跑。
- 模型缺失时服务会自动回退 ONNX Runtime（CPU），日志里会有 warning。