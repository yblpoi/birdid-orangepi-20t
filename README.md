# birdid-orangepi-20t

在香橙派 AI Pro 20T（昇腾 Ascend 310B1）上运行的鸟类识别 API，兼容懂鸟（Dongniao）动物识别协议。

- 图像链路：YOLO26-seg 门禁/裁剪 → BirdTrace-FastViT-MA36-10K 分类，全程昇腾 NPU 推理
- 声音链路：BirdNET 识别 + GeoModel 地理/季节后过滤
- API：`POST /dongniao`，响应为 `[code, payload]` 元组

## 状态

项目正在开发中。

## 许可证

代码以 AGPL-3.0 发布。模型权重的授权各不相同（其中若干为非商用），本项目不随仓库分发权重，详见 `NOTICE` 与 `docs/models.md`。
