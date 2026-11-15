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

BirdTrace 的 `classes.json` 只有英文俗名，而 GeoModel 按学名索引，需要离线生成一次映射表
（也是这一步让图片链路能拿到学名标签）：

```bash
python scripts/build_species_map.py \
  models/birdtrace_classes.json \
  models/geomodel_v3_labels.txt \
  models/birdtrace_species_map.json
```

正常输出 `classes=10480 matched=9257 (88.3%)`。匹配率低于 85% 通常是 GeoModel 标签表版本不对。

## 4. 转 OM 模型（NPU 推理用）

三个模型各有各的坑，按顺序执行即可。转换前先确认：

```bash
export TE_PARALLEL_COMPILER=4   # 板子 4 核，默认 8 路并行会把内存打爆导致 ATC 假死
source /usr/local/Ascend/cann-9.2.1/set_env.sh
```

### 4.1 YOLO26-seg

先裁掉分割原型分支：服务只用检测框（懂鸟规范返回 box + list，不需要 mask），
而原来的 `output1` 每次推理要往主机拷 3.2 MB，并驱动一条 prototype 卷积分支。
裁掉后 640x640 单图 114.3 ms → 101.6 ms，检测结果逐位相同：

```bash
python scripts/export_yolo.py --weights yolo26m-seg.pt --output models/yolo26m-seg.onnx
python scripts/prune_yolo_outputs.py models/yolo26m-seg.onnx models/yolo26m-seg-box.onnx
scripts/convert_om.sh models/yolo26m-seg-box.onnx models/yolo26m-seg \
    --input_shape 'images:1,3,640,640'
```

裁完只有 `output0` 一个输出，服务侧解码逻辑不变（`decode_predictions` 只读 `output0`）。

### 4.2 BirdTrace FastViT-MA36

原始 ONNX 需先做图修复，否则 ATC 先报
`input bias shape should be 1D or 4D`（36 个 Conv 的 bias 来自 Expand 子图），
再报 `No supported Ops kernel ... StridedSliceV3`（18 个 k:k+1 的 Slice）：

```bash
python scripts/repair_birdtrace_onnx.py models/birdtrace_raw.onnx models/birdtrace.model.onnx
```

修完还要用 onnxsim 把动态 shape 折叠成静态，否则 OM 能转出来但
`acl.mdl.load_from_file` 返回 500002（GE 的 init graph 在 310B1 上推出垃圾维度）：

```bash
python -c "import onnx, onnxsim; m = onnx.load('models/birdtrace.model.onnx'); \
  s, ok = onnxsim.simplify(m, overwrite_input_shapes={'image': [1,3,384,384]}, \
  dynamic_input_shape=False, perform_optimization=True); onnx.save(s, 'models/fastvit_sim.onnx')"
scripts/convert_om.sh models/fastvit_sim.onnx models/fastvit_sim --input_shape 'image:1,3,384,384'
```

### 4.3 BirdNET v3

BirdNET 整图**不能**直接转：310B1 没有 `STFT` 与 `ReduceL2` 的 ONNX 算子插件
（`Not_Supported_Operator(E13010)`）。做法是把 mel 之前的 STFT 前端留在 CPU：

```bash
python scripts/split_birdnet.py --input models/birdnet_v3_fp16.onnx \
    --front-out models/birdnet_v3_fp16_front.onnx \
    --back-out models/birdnet_v3_fp16_back.onnx
python -c "import onnx; m = onnx.load('models/birdnet_v3_fp16_back.onnx'); \
  m.ir_version = 7; onnx.save(m, 'models/birdnet_v3_fp16_back_ir7.onnx')"
scripts/convert_om.sh models/birdnet_v3_fp16_back_ir7.onnx models/birdnet_v3_fp16_back \
    --input_shape 'node_mm_output_cast_0:188,128'
```

`ir_version` 必须降到 7，否则 ATC 解析 `ir 10 / opset 18` 的切片会报
`E19999: Internal error!`；opset 也不要降到 11（`Reshape` 的 `allowzero` 属性不被识别）。

服务会自动识别：`models/` 下有 `birdnet_v3_fp16_back.om` 与 `birdnet_v3_fp16_front.onnx`
就走混合后端（CPU mel + NPU 主干），否则整体回退 CPU。

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