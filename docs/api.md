# API 说明

服务实现懂鸟动物识别 API 兼容协议：单个接口 `POST /dongniao`，响应体恒为 `[code, payload]` 二元组。

## 认证

请求头 `api_key`，或表单字段 `api_key`，二选一。服务端未配置密钥（环境变量 `BIRDID_API_KEY` 为空）时跳过校验。

## 请求分发

服务根据表单字段决定处理分支：

| 字段 | Content-Type | 行为 |
| --- | --- | --- |
| `image` + `upload` | multipart/form-data | 上传图片识别，≤2MB，边长 11~8192 |
| `sound` + `upload` | multipart/form-data | 上传鸟鸣识别，≤5MB |
| `url` | x-www-form-urlencoded | 服务端下载图片（5s 超时，最多 2 次重定向） |
| `sound_url` | x-www-form-urlencoded | 服务端下载音频（≤5MB，10s，最多 2 次重定向） |
| `resultid` | 任意 | 查询识别结果 |
| `animalid` (+ `class`) | 任意 | 查询物种资料 |
| `quota` | 任意 | 查询剩余配额 |

## 响应码

| code | 含义 |
| --- | --- |
| 1000 | 成功（受理返回 resultid；查询返回结果数组） |
| 1001 | 处理中 / URL 下载失败 |
| 1002 | 音频 URL 下载或解码失败 |
| 1007 | 配额耗尽或限流（HTTP 503） |
| 1010 | 未识别到动物 |

## 结果结构

```json
[1000, [{"box": [31, 9, 379, 356],
        "list": [[97.6, "北美红松鼠|North American Red Squirrel|Tamiasciurus hudsonicus", 6534, "M"]]}]]
```

- `box`：裁剪框，`[x1, y1, x2, y2]`
- `list`：`[置信度, "中文名|英文名|学名", animalid, class]`，按置信度降序

## 非鸟类处理

识别到鸟类时裁剪并送入分类器；整图未检出鸟类时，返回检测到的具体类别与置信度，例如
`[1000, [{"box": [...], "list": [[99, "狗|Dog|Canis lupus familiaris", 0, "M"]]}]]`，
完全无动物时返回 `[1010, "No animals"]`。

## 扩展字段（非懂鸟规范）

图片与声音两条链路都支持可选的 `latitude` / `longitude` / `week` 表单字段，用于叠加 GeoModel 地理与季节先验：

- `latitude`：-90 ~ 90，缺失或非法则跳过地理过滤；
- `longitude`：-180 ~ 180，与 `latitude` 同时生效；
- `week`：1 ~ 48（缺省按当天换算），非法值按当天换算。

图片链路的做法是：先取 top-N（N = `max_detections`）候选，按 `分类置信度 x 地理概率^geo_weight` 重排，
再截断到 `top_k` 返回；重排后会把分数等比缩放回原来的总和，所以返回的置信度仍然可读。
92 张带真实拍摄坐标的评测图上，BirdTrace top1 命中率 83.0% -> 96.6%，修好的 12 个样本全部是同属/近缘种混淆，零误伤。

BirdNET 与 GeoModel 的分类体系不同（11,560 类 / 14,082 类），BirdTrace 与 GeoModel 也不同（10,480 类 / 14,082 类），
服务一律按**学名**关联（BirdTrace 侧由 `scripts/build_species_map.py` 离线生成映射），
GeoModel 未覆盖的物种按中性权重处理，不做地理惩罚。

## 声音结果结构

声音识别没有检测框，`box` 为空数组：

```json
[1000, [{"box": [], "list": [[66.2, "Wing-banded Wren|Wing-banded Wren|Microcerculus bambla", 6505, "B"]]}]]
```

`animalid` 为 BirdNET 标签表的行序号，仅在本服务内有意义。
