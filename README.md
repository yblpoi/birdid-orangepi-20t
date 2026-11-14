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

## 实测性能（310B1，单张 640x640）

| 环节 | NPU | CPU |
| --- | --- | --- |
| YOLO26m-seg 检测 | 46.6 ms | 4577 ms |

BirdTrace / BirdNET 的 NPU 转换与对齐结果见 `docs/models.md`。

## 状态

图像与声音链路均已跑通；正在完成 BirdTrace 分类模型的全量 OM 转换与 NPU 精度对齐。

## 许可证

代码以 AGPL-3.0 发布。模型权重授权各不相同（其中若干为非商用），本项目不随仓库分发权重，详见 `NOTICE` 与 `docs/models.md`。