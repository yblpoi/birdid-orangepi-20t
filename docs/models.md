# 模型与权重

仓库**不包含**任何权重文件，均需本地下载后放入 `models/`（该目录已被 `.gitignore` 忽略）。

| 文件 | 来源 | 授权 | 备注 |
| --- | --- | --- | --- |
| `yolo26n-seg.pt` / `.onnx` / `.om` | https://github.com/ultralytics/assets | AGPL-3.0 | 门禁 + 鸟类区域裁剪（现默认，快） |
| `yolo26m-seg.pt` / `.onnx` / `.om` | https://github.com/ultralytics/assets | AGPL-3.0 | 同上，历史默认值（准，慢 2.6 倍） |
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
| BirdNET v3 | 切 mel 前端，主干 `ir_version=7` 后转换（`scripts/split_birdnet.py`） | 每个 3 s 窗口 top1 一致，2~4 名有偏移 |
| BirdNET mel 前端 | 原始 ONNX 的 STFT 算子无 ATC 插件（`Not_Supported_Operator(E13010): No operator plugin is registered for Op: node_STFT_39, optype: ai.onnx::18::STFT`），故把「加窗 + rfft + 取模平方 + mel 投影」解析展开后折叠成两次 MatMul 再转换 | 与 ORT CPU 前端最大绝对差 5.2e-5，4/4 窗口 top1 相同、top10 重合 100% |
| BirdNET 切分主干 | trunk / head 按 mel 之后的张量边界切，head 的 Conv 改写成 MatMul 才能批量 | 12 窗口 top1 与整图 12/12 一致；head 批量输出与逐窗逐位相同 |

上面后三个模型的 ONNX 都由 `scripts/build_birdnet_split.py` 从 front/back 两段生成
（用法见 `docs/deploy.md` 4.3）。脚本会用 float64 展开式自检 mel 前端（最大绝对差 7.3e-05），
重新生成的 ONNX 与在板 OM 都与仓库里在用的产物逐位相同。

### 鸟鸣链路的批量策略（有收益，与图像链路相反）

图像链路的批推理没有收益（见下），但鸟鸣链路里 **打分头批量化是最大的一笔提速**。
BirdNET 主干尾部有四个 `[11560, 2048, 1]` 的 Conv，组成最终的打分头；把主干切成
trunk（卷积主干）与 head（打分头）后，把 head 的 Conv 改写成 MatMul 就能真正批量：

| 环节 | 单窗口 | 4 窗口逐窗 | 4 窗口一次批量 |
| --- | --- | --- | --- |
| head（打分头） | 6.5 ms | 23.5 ms | 7.2 ms |
| mel 前端 | 1.3 ms | 5.7 ms | 4.1 ms |
| trunk（卷积主干） | 7.6 ms | 30.4 ms | 30.4 ms（批量无收益） |

批量档不止一种，单位窗口成本随批量下降（B4/B8/B12 = 7.2/9.7/14.0 ms），所以按窗口数
挑最优组合；不足一批时尾部补零，补出来的窗口落在批次末尾直接丢弃，实测输出与逐窗逐位相同。
27 窗口的 head 成本由此从「7×B4 = 50.6 ms」降到「B12+B8+B8 = 33.4 ms」。

**trunk 批量没有收益**：把 4 份 trunk 展开成一张 4 输入 16 输出的图（trunkU4.om），
单次 29.9 ms vs 4× 单窗 30.4 ms，只快 1.6%，说明 310B1 上卷积主干已经吃满算力。

### 批推理没有收益（实测）

NPU 上把 seg 模型固定成批维度重转（batch=4，输入 `4x3x640x640`），单次 execute 132.2 ms，
折合每图 33.0 ms；而 batch=1 的 box-only 模型单图 32.1 ms —— **几乎完全相同**。吞吐 30.3 张/秒
与单张模型的 31 张/秒一致，说明瓶颈在算子计算本身而不是下发开销，攒批只会增加排队延迟。
因此服务不做微批处理，请求保持一对一推理。

### 前处理优化（CPU 侧，与 NPU 无关）

端到端 98.1 ms 里 NPU 只占 57 ms，其余全是 PIL 前处理，因此这块单独做过对照：

| 改动 | 结果 |
| --- | --- |
| 裁剪 + 分类器缩放合并成一次 `Image.resize(box=...)` | 20.0 ms → 13.8 ms，逐元素 maxdiff 0.0 |
| 分类器 resize 双三次 → 双线性 | 快约 40%，top1 不降反升（91.4% → 92.9%） |
| letterbox 用 `reduce()` 逐级下采样替代 `reducing_gap=2.0` | 24.1 ms → 9.6 ms，但 1280px 图上 3 张结果变化、命中 -1，**不采用** |
| 批推理（batch=4/8） | 每图耗时与 batch=1 相同，只增延迟，**不采用** |

注意 letterbox 的 `reducing_gap=2.0` 只在原图显著大于目标尺寸时才触发缩小；iNaturalist 评测集的最长边都不超过 1024，所以在该集合上对比两种实现完全无差异（93 张全部同结果），必须先把图放大到 1280px 才能看出差异 —— 评测集尺寸分布本身会掩盖这类改动。

### 粗粒度坐标的影响（area 字段）

懂鸟的 `area` 地区码只能给出城市级代表坐标，用 92 张评测图做过坐标粗化实验
（把真实拍摄坐标分别取整到 1/3/5/10 度后重跑地理先验）：

| 坐标精度 | top1 命中 |
| --- | --- |
| 原始坐标 | 96.6% |
| 取整到 1 度 | 96.6% |
| 取整到 3 度 | 97.7% |
| 取整到 5 度 | 93.2% |
| 取整到 10 度 | 89.8% |

城市级坐标（1~3 度内）几乎无损，5 度以上才开始掉 —— 这正好是 `area` 场景的实际精度。
降级时把 `geo_weight` 调小也能回收一部分（10 度 + weight 0.3 时回到 94.3%）。

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
