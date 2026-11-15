# birdid-orangepi-20t

在香橙派 AI Pro 20T（昇腾 Ascend 310B1）上运行的鸟类识别 API，兼容懂鸟（Dongniao）动物识别协议。

- 图像链路：YOLO26-seg 门禁/裁剪 → BirdTrace-FastViT-MA36-10K 分类，全部走昇腾 NPU
- 声音链路：BirdNET v3 识别 + GeoModel 地理/季节后过滤
- API：`POST /dongniao`，响应恒为 `[code, payload]` 二元组

## 快速开始

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
BIRDID_MODEL_DIR=models scripts/download_models.sh
scripts/run_server.sh
```

完整步骤见 [docs/deploy.md](docs/deploy.md)，协议见 [docs/api.md](docs/api.md)，
模型来源与授权见 [docs/models.md](docs/models.md)。

## 目录

```
configs/   服务与区域配置
src/birdid/  服务实现（协议、模型、存储、执行器）
scripts/  下载、导出、转换、压测、启动脚本
tests/    单元与 HTTP 层测试
docs/     协议、模型、部署文档
```

## 实测性能（310B1，单次推理）

| 环节 | 输入 | NPU 执行 | 同板 CPU 执行 | 加速比 |
| --- | --- | --- | --- | --- |
| YOLO26m-seg 检测（裁掉 mask 分支） | 1x3x640x640 | 32 ms | 4577 ms | ~140x |
| BirdTrace FastViT-MA36 分类 | 1x3x384x384 | 25.7 ms | 1933.9 ms | 75x |
| BirdNET 主干（mel 之后） | 188x128 | 12.5 ms | 91.5 ms | 7.3x |
| BirdNET 混合链路（含 CPU mel 前端） | 3 s 音频 | 35.9 ms | 105.4 ms | 2.9x |

上表是**纯推理**阶段；整条图片链路端到端（解码 + letterbox + 检测 + 裁剪 + 分类）1280x960 单图 98.1 ms，
其中 NPU 占 57 ms，其余是 PIL 前处理（大图已按 1/2~1/4 抽样解码抵消）。
前处理做过三处优化，都验证过数值等价或精度不降：
裁掉 seg 模型的 `output1`（1x32x160x160 的 mask 原型）让 640x640 单图 114.3 ms → 101.6 ms，检测框与置信度逐位相同；
裁剪与分类器缩放的两次重采样合并成一次 `Image.resize(box=...)`，该段 20.0 ms → 13.8 ms、maxdiff 0.0；
分类器 resize 用双线性替代双三次，快约 40% 且 top1 不降。
输入降到 576/512 虽然更快（104.3/94.6 ms）但 top1 掉到 82.6%/75.5%，故保持 640。

API 吞吐（`POST /dongniao` 图片，Mac 侧打流，全部 code=1000 零失败）：

| workers | 并发 | 吞吐 | 平均延迟 | p99 |
| --- | --- | --- | --- | --- |
| 2 | 4 | 10.14 req/s | 378.5 ms | 611.2 ms |
| 4 | 4 | 11.22 req/s | 340.0 ms | 538.5 ms |
| 4 | 8 | 12.33 req/s | 612.9 ms | 880.9 ms |
| 6 | 8 | 12.29 req/s | 599.8 ms | 924.4 ms |

吞吐被 NPU 单卡串行执行锁死（`AclModel.infer` 多线程只排队不提吞吐，实测 threads=1/2/4 恒为 17.4 inf/s），
并发上去后延迟线性增长即排队效应；继续加 `server.workers` 不再有收益。

既然排队只会把延迟推高到客户端超时，服务对队列深度做了硬性上限（`server.max_queue`，默认 32）：
打满时新请求立刻返回 `[1007, "server busy"]`，让调用方按限流语义快速退避，而不是拿到一个
迟迟出不来结果的 resultid。

精度对齐：BirdTrace 的 NPU 与 CPU logits 最大绝对差 0.2357（logits 量级 -50~-60），top1 与 top5 完全一致；
BirdNET 主干每个 3 秒窗口的 top1 一致，第 2~4 名存在顺序与分值偏移（详见 `docs/models.md`）。

## 识别精度（92 张真实照片实测）

评测集来自 iNaturalist 研究级观测（49 张训练集 + 43 张留出集，覆盖 90+ 物种），
全部按懂鸟协议走 `POST /dongniao` 图片链路，只统计 top1 精确命中。

| 配置 | 命中 / 总数 | 命中率 |
| --- | --- | --- |
| 整图直接分类（不裁剪） | 61 / 92 | 66.3% |
| YOLO 裁剪，边框 0.08（旧默认） | 66 / 92 | 71.7% |
| YOLO 裁剪，边框 0.55（现默认） | 76 / 92 | 82.6% |
| 上述 + GeoModel 地理先验 | 85 / 88 | 96.6% |

三点结论：

1. **紧贴检测框裁剪反而更差**：边框 0 在训练集上与整图持平（67.3%），留 0.50~0.60 是平台区（82.6%），
   留到 0.70 又开始回落（76.1%）。旧默认 0.08 明显偏低，改到 0.55 后两类评测集分别 +12.2 / +9.3 个百分点。
2. **按框面积自适应的边距策略会过拟合**：在 49 张训练集上刷到 85.7%，换到 43 张留出集只有 76.7%~79.1%，
   还不如固定常数，因此保留单一常数。
3. **地理先验专治「同属近缘种」**：不开地理重排时的错例几乎全是 Eurasian Nuthatch vs. 其他 Sitta、
   American vs. European Goldfinch 这类地缘替代种；叠上 GeoModel 后修好 12 个、打坏 0 个。
   反向对照（把 97 个拍摄坐标在标签之间随机打乱 8 次）gain 全部为负（均值 -12.1），说明增益不是随机抖动。

## 鸟鸣精度（72 段录音实测）

评测集来自 iNaturalist 研究级带音频观测（25 个欧洲常见物种，每物种最多 3 段，
用 ffmpeg 统一转成 32 kHz 单声道并截前 12 秒），逐段走声音链路，只统计 top1 精确命中
（已折叠 Common/Eurasian Chaffinch、Common/Eurasian Blackbird 两对同物异名）。

| 配置 | top1 命中 / 总数 | 命中率 |
| --- | --- | --- |
| 不加地理先验 | 58 / 72 | 80.6% |
| + GeoModel 地理/季节先验 | 60 / 72 | 83.3% |
| 上述的 top5 命中率 | 68 / 72 | 94.4% |

结论：录音本身含大量背景鸣声与重叠个体，最典型的错例是整段只有背景物种在叫
（如标注 Eurasian Blue Tit 的片段里 Common Blackbird 更响），属于标注与音频内容不一致，
而不是模型退化；地理先验在声音链路上同样只增不减（+2）。

## 状态

图像与声音链路均已端到端跑通：YOLO26m-seg 与 BirdTrace 全量走 NPU；BirdNET 采用「CPU mel 前端 + NPU CNN 主干」混合后端；GeoModel 走 CPU。

## 许可证

代码以 AGPL-3.0 发布。模型权重授权各不相同（其中若干为非商用），本项目不随仓库分发权重，详见 `NOTICE` 与 `docs/models.md`。