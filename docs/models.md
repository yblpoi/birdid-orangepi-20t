# 模型与权重

仓库**不包含**任何权重文件，均需本地下载后放入 `models/`（该目录已被 `.gitignore` 忽略）。

| 文件 | 来源 | 授权 | 备注 |
| --- | --- | --- | --- |
| `yolo26m-seg.pt` / `.onnx` / `.om` | https://github.com/ultralytics/assets | AGPL-3.0 | 门禁 + 鸟类区域裁剪 |
| `model.onnx` → `fastvit.om` | https://github.com/wardens/BirdTrace-FastViT-MA36-10K | CC BY-NC 4.0 | 鸟类物种分类，10480 类 |
| `birdnet.onnx`（或 TFLite） | https://github.com/birdnet-team/BirdNET-Analyzer | CC BY-NC-SA 4.0 | 鸟鸣识别 |
| BirdNET 标签表 | 同上 | CC BY-NC-SA 4.0 | 物种编码与中英文名 |
| `geomodel_fp32.onnx` | https://github.com/birdnet-team/geomodel | Apache-2.0 | 地理/季节后过滤，14082 物种 |
| Geomodel 标签表 | 同上 | Apache-2.0 | speciesCode/学名/英文名 |

## 获取方式

使用 `scripts/download_models.sh`，默认走 `https://hf-mirror.com` 与 `https://gh-proxy.com` 加速。
下载后请校验文件大小与 SHA256（脚本会打印期望值）。

## NPU 转换与精度（310B1 实测）

| 模型 | 转换要点 | NPU 精度对齐 |
| --- | --- | --- |
| YOLO26m-seg | 直接转换，`--input_shape 'images:1,3,640,640'` | 原始张量最大差 596.8，但解码后的框/分数/类别完全一致 |
| BirdTrace FastViT | 修 bias → Slice 改 Gather → 剪死节点 → onnxsim 静态折叠 | logits 最大绝对差 0.2357，top1/top5 完全一致 |
| BirdNET v3 | 切 mel 前端，主干 `ir_version=7` 后转换 | 每个 3 s 窗口 top1 一致，2~4 名有偏移 |

### 三个关键坑（都会导致「转换成功但跑不起来」或「根本转不动」）

1. **ATC 假死不是死锁，是内存超发**。默认并行编译 8 路（`te_fusion/parallel_compilation.py`
   的 `get_multi_process_count()`），4 核 23 GB 无 swap 的板子会被打爆。设 `TE_PARALLEL_COMPILER=4`
   后，同一张 992 节点的图从 90 分钟不出结果变成 8 分 44 秒完成。
2. **`--jit_compile=0` 转出的 OM 可能加载失败**（`acl.mdl.load_from_file` 返回 500002 =
   `ACL_ERROR_GE_FAILURE`）。一定要实测加载，不能只看 ATC 报 success。查因要开
   `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_SLOG_PRINT_TO_STDOUT=1` 抓 GE 的 InferShape 行。
3. **ACL context 是线程绑定的**。`acl.rt.malloc` 返回 107002 表示当前线程没有 context，
   多线程服务必须在工作线程里 `acl.rt.set_device` + `acl.rt.set_context`（已在 `acl_runner.py` 处理）。

### 精度对比的口径

不要把 NPU 与 CPU 的**原始张量**直接对比：YOLO26-seg 的 32 个 mask 系数在低置信度行上差异极大（596.8），
但解码后的检测框、分数、类别完全一致。验收要比较**业务语义**（框 + 类别 + 排序），不是浮点数。

### 鸟鸣链路精度（72 段录音）

| 配置 | top1 | top5 |
| --- | --- | --- |
| BirdNET v3（CPU mel + NPU 主干） | 58 / 72 = 80.6% | — |
| 再叠 GeoModel 地理/季节先验 | 60 / 72 = 83.3% | 68 / 72 = 94.4% |

录音取自 iNaturalist 研究级观测，统一转 32 kHz 单声道并截前 12 秒（4 个窗口取最大概率）。
错例集中在两类：一是标注物种只是背景个体（例如标注 Eurasian Blue Tit 的片段里 Common Blackbird 更响），
二是录音里有多个个体同时鸣叫。GeoModel 在声音链路上同样是只增不减。

注意 BirdNET 与 BirdTrace 对同一物种的英文俗名可能不同（BirdNET 用 Eurasian Chaffinch，
BirdTrace 用 Common Chaffinch），评测时必须按同一规范折叠，否则会把正确结果算成错误。

## 授权提示

BirdTrace 与 BirdNET 的权重为**非商用**授权。若需商用，请自行替换为可商用模型或另行取得授权。
代码本身以 AGPL-3.0 发布，不改变模型权重的授权。
