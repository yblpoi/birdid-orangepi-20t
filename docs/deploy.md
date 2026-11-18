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

门禁用 **n 版**（`yolo26n-seg`）：310B1 上同图单次前向 32.4 ms → 12.2 ms，约 2.65 倍加速，
m 版只是同一套流程的历史默认值。两个版本都要先裁掉分割原型分支：服务只用检测框
（懂鸟规范返回 box + list，不需要 mask），而原来的 `output1` 每次推理要往主机拷 3.2 MB，
并驱动一条 prototype 卷积分支。裁掉后 640x640 单图 114.3 ms → 101.6 ms，检测结果逐位相同：

```bash
# n 版（现默认）；换成 m 版只需把文件名里的 n 换成 m
python scripts/export_yolo.py --weights yolo26n-seg.pt --output models/yolo26n-seg.onnx
python scripts/prune_yolo_outputs.py models/yolo26n-seg.onnx models/yolo26n-seg-box.onnx
scripts/convert_om.sh models/yolo26n-seg-box.onnx models/yolo26n-seg \
    --input_shape 'images:1,3,640,640'
```

裁完只有 `output0` 一个输出，服务侧解码逻辑不变（`decode_predictions` 只读 `output0`）。
转换成功时日志末行是 `ATC run success, welcome to the next use.`；夹在中间的
`Performance_Not_Optimal_Operator(W11001)` 是 310B1 对少数算子的常规提示，可以忽略。

转完把 `.om` 放进 `models/`（`models/` 里已有历史版本时用符号链接指向真实文件即可），
并让 `configs/default.yaml` 的 `npu.seg_model` / `npu.seg_onnx` 分别指向 `.om` 与 `.onnx`。

**鸟类低分兜底**：`npu.seg_bird_conf_threshold`（默认 0.02）只对 COCO `bird` 这一类单独放宽阈值，
其余类别继续用 `seg_conf_threshold`（默认 0.25）。小目标、翻拍屏幕上拍的照片，鸟的分数会掉到
5%~15%，按整表阈值过滤就会只剩同位置的 person 框 → 表现成「照片里明明是鸟却报成人」。
非鸟图片上 `bird` 的噪声长期低于 0.6%，放宽到 0.02 不会把背景误判成鸟；
同一位置的鸟框与人框也不会互相压制（NMS 已按类别分组抑制），两边的框都能进候选。

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
（`Not_Supported_Operator(E13010): No operator plugin is registered for Op: node_STFT_39`）。

先把整图切成前后两段（`scripts/split_birdnet.py`），再用 `scripts/build_birdnet_split.py`
把这两段变成能上 NPU 的五个模型：

```bash
# 1) 整图 → front（STFT 前端）+ back（CNN 主干）
python scripts/split_birdnet.py --input models/birdnet_v3_fp16.onnx \
    --front-out models/birdnet_v3_fp16_front.onnx \
    --back-out models/birdnet_v3_fp16_back.onnx
python -c "import onnx; m = onnx.load('models/birdnet_v3_fp16_back.onnx'); \
  m.ir_version = 7; onnx.save(m, 'models/birdnet_v3_fp16_back_ir7.onnx')"

# 2) front → mel 前端（把 STFT 展开成两次 MatMul）
#    back  → trunk（卷积主干）+ head_b{4,8,12}（打分头，批量）
python scripts/build_birdnet_split.py \
    --front models/birdnet_v3_fp16_front.onnx \
    --back models/birdnet_v3_fp16_back_ir7.onnx \
    --out-dir models

# 3) 逐个转 OM
scripts/convert_om.sh models/mel_dft.onnx models/birdnet_v3_fp16_mel
scripts/convert_om.sh models/mel_dft_b4.onnx models/birdnet_v3_fp16_mel_b4
scripts/convert_om.sh models/trunk.onnx models/birdnet_v3_fp16_trunk
for b in 4 8 12; do
    scripts/convert_om.sh models/head_b$b.onnx models/birdnet_v3_fp16_head_b$b
done
```

三步各自绕开一个坑，都不直观：

1. **`ir_version` 必须降到 7**，否则 ATC 解析 `ir 10 / opset 18` 的切片会报
   `E19999: Internal error!`；opset 也不要降到 11（`Reshape` 的 `allowzero` 属性不被识别）。
2. **mel 前端的 STFT 要手工展开**。原图里 `STFT` 算子没有 CANN 插件，但这个前端其实等价于
   「加窗分帧 → rfft → 取模平方 → mel 矩阵投影」，而 mel 矩阵只有 1025 个 bin 里的 957 个非零
   （连续区间 3..959）。于是把窗函数融进 DFT 矩阵、只算这 957 个 bin，
   整条前端折叠成 `MatMul(frames[188,2048] @ D[2048,1914]) → |·|² → MatMul(mel[957,128])` 两个矩阵乘，
   ATC 就能转。脚本会用 float64 展开式对齐 ORT 前端做自检，正常打印
   `最大绝对差 7.3e-05`（float32 精度量级）。
3. **主干要切成 trunk + head 才能批量**。head 里四个 `[11560, 2048, 1]` 的 Conv 是逐帧点积，
   批量维会被后续的 ReduceMean 折叠掉，直接把批维改成 4 只会得到「只算第一个窗口」的假批量。
   改写成 `Transpose → Reshape → MatMul → Add → Reshape → Transpose` 后才真正批量：
   4 个窗口一次前向 7.2 ms，逐窗要 23.5 ms。B8/B12 同理（9.7 / 14.0 ms），
   服务按窗口数挑最优组合，27 个窗口的头部成本从 50.6 ms 降到 33.4 ms。

服务会自动识别 `models/` 下的产物并逐级降级：

| 找到的文件 | 后端 | 3 秒音频 |
| --- | --- | --- |
| `birdnet_v3_fp16_mel.om` + `trunk.om` + `head_b*.om` | `npu-split`（全 NPU） | 15.8 ms |
| `birdnet_v3_fp16_mel.om` + `birdnet_v3_fp16_back.om` | `npu`（全 NPU，整图主干） | 35.9 ms |
| `birdnet_v3_fp16_front.onnx` + `birdnet_v3_fp16_back.om` | `hybrid-front-cpu-back-npu` | 105 ms |
| 都没有 | `cpu` | 更慢 |

`configs/default.yaml` 里 `birdnet_head_batch_oms` 是逗号分隔的多个批量档，
少放几个不会报错，只是长音频的头部成本会上升；全部缺失时自动回退整图主干。

## 5. 启动服务

前台调试（关掉终端就停）：

```bash
BIRDID_API_KEY=... scripts/run_server.sh
```

`scripts/run_server.sh` 会 source CANN 环境、把 `src` 与 pyACL 的 site-packages
加入 `PYTHONPATH`，再以单进程（NPU 会话不可跨进程共享）启动 uvicorn。

### 5.1 装成 systemd 服务（推荐）

一键安装，开机自启、崩溃 5 秒后自动拉起、日志进 journald：

```bash
sudo scripts/install_service.sh            # 用仓库属主身份运行服务
sudo scripts/install_service.sh --user foo # 指定运行用户
scripts/install_service.sh --dry-run       # 只渲染 unit 内容，不落盘、不要 root
```

脚本做的事：按 `scripts/birdid.service.in` 渲染出 `/etc/systemd/system/birdid.service`；
若还在跑手工起的 `python -m birdid` 会先把它停掉（否则 8080 被占，新服务起不来）；
建一个权限 600 的 `/etc/birdid/birdid.env` 空模板；然后 `daemon-reload` + `enable` + `restart`，
轮询 `/healthz` 直到就绪（最多约 2 分钟）。

api_key 只走环境变量，不写进 unit 也不写进仓库：

```bash
sudo sed -i 's|^# BIRDID_API_KEY=.*|BIRDID_API_KEY=<你的 key>|' /etc/birdid/birdid.env
sudo systemctl restart birdid
```

日常运维：

```bash
systemctl status birdid          # 状态
journalctl -u birdid -f          # 跟日志
sudo systemctl restart birdid    # 改完配置重启
sudo scripts/uninstall_service.sh         # 卸载，保留 /etc/birdid/birdid.env
sudo scripts/uninstall_service.sh --purge # 连密钥文件一起删
```

卸载只动 unit 与环境变量文件，仓库代码、模型、归档目录都不受影响。

## 6. 验证

```bash
curl -s localhost:8080/healthz
curl -s -H 'api_key: ...' -F image=@bird.jpg -F upload=1 localhost:8080/dongniao
```

基准测试：`PYTHONPATH=src python scripts/benchmark.py --rounds 10`。

换板子、改模型或调完 `configs/default.yaml` 之后，先跑一遍设备归属自检：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=src:$PYTHONPATH python scripts/check_npu.py --audio your.wav
```

它会先打印 BirdNET 与 GeoModel 的 backend、以及各条链路上还留没留 ORT 会话
（有会话就说明回退了 CPU），最后一行 `=> 全 NPU：是` 才算配置生效；
随后按 1/2/3/4/5/8/9/12/16/17/20/24/27 个窗口扫端到端延迟、用 1/2/4/8 线程压吞吐、
连跑若干次看内存与标签漂移。`--only placement latency` 可以只跑其中几项。

`PYTHONPATH` 必须是 `src:$PYTHONPATH`，只写 `src` 会把 CANN 的 site-packages 顶掉
（报 `No module named 'acl'`）。

复现 README 里的两套精度数字（都要联网拉 iNaturalist 素材）：

```bash
# 图片：抓真实照片，再用 HTTP 链路逐张识别（脚本只负责取图与写清单）
python scripts/fetch_eval_set.py models/birdtrace_classes.json taxa.txt /tmp/birdimg img_manifest.txt

# 声音：抓带音频的观测，ffmpeg 统一转 32 kHz 单声道后逐段识别
python scripts/fetch_sound_eval.py models/birdtrace_classes.json taxa.txt /tmp/birdsnd sound_manifest.txt
PYTHONPATH=src python scripts/eval_sound.py sound_manifest.txt --verbose
```

`eval_sound.py` 会同时打印「不加地理先验」「加地理先验」「top5」三行命中率。

## 7. 排查

- `npu-smi info` 显示 `Health: Alarm` 是该开发板的硬件设计缺陷，可忽略。
- ATC 转换被强杀后会残留 forkserver 子进程并占满内存，必须清理后再重跑。
- 模型缺失时服务会自动回退 ONNX Runtime（CPU），日志里会有 warning。