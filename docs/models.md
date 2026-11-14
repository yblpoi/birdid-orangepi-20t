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

## 授权提示

BirdTrace 与 BirdNET 的权重为**非商用**授权。若需商用，请自行替换为可商用模型或另行取得授权。
代码本身以 AGPL-3.0 发布，不改变模型权重的授权。
