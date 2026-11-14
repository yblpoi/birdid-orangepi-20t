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

| 环节 | 输入 | NPU | 同板 CPU | 加速比 |
| --- | --- | --- | --- | --- |
| YOLO26m-seg 检测 | 1x3x640x640 | 46.6 ms | 4577 ms | 98x |
| BirdTrace FastViT-MA36 分类 | 1x3x384x384 | 25.7 ms | 1933.9 ms | 75x |
| BirdNET 主干（mel 之后） | 188x128 | 12.5 ms | 91.5 ms | 7.3x |
| BirdNET 混合链路（含 CPU mel 前端） | 3 s 音频 | 35.9 ms | 105.4 ms | 2.9x |

精度对齐：BirdTrace 的 NPU 与 CPU logits 最大绝对差 0.2357（logits 量级 -50~-60），top1 与 top5 完全一致；
BirdNET 主干每个 3 秒窗口的 top1 一致，第 2~4 名存在顺序与分值偏移（详见 `docs/models.md`）。

## 状态

图像与声音链路均已端到端跑通：YOLO26m-seg 与 BirdTrace 全量走 NPU；BirdNET 采用「CPU mel 前端 + NPU CNN 主干」混合后端；GeoModel 走 CPU。

## 许可证

代码以 AGPL-3.0 发布。模型权重授权各不相同（其中若干为非商用），本项目不随仓库分发权重，详见 `NOTICE` 与 `docs/models.md`。