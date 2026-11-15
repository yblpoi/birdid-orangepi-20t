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
| 1007 | 配额耗尽、限流或服务排队已满（本地部署返回 HTTP 200 + `"server busy"`） |
| 1010 | 未识别到动物 |

## 结果结构

```json
[1000, [{"box": [31, 9, 379, 356],
        "list": [[97.6, "北美红松鼠|North American Red Squirrel|Tamiasciurus hudsonicus", 6534, "M"]]}]]
```

- `box`：裁剪框，`[x1, y1, x2, y2]`
- `list`：`[置信度, "中文名|英文名|学名", animalid, class]`，按置信度降序

## 过载保护

推理队列有硬性上限（`server.max_queue`，默认 32）。NPU 单卡串行执行，排队过深只会让
每个请求的等待时间线性增长直到客户端超时，因此队列打满时新请求立刻返回
`[1007, "server busy"]`，客户端按限流语义退避重试即可，不必等一个永远不会及时完成的 resultid。

## 非鸟类处理

识别到鸟类时裁剪并送入分类器；整图未检出鸟类时，返回检测到的具体类别与置信度，例如
`[1000, [{"box": [...], "list": [[99, "狗|Dog|Canis lupus familiaris", 0, "M"]]}]]`，
完全无动物时返回 `[1010, "No animals"]`。
`class` 字段的语义：只有客户端**显式**传入 `class` 时才按类别过滤返回结果（例如只传 `B` 就只保留鸟类条目）；
不传或传空时返回全部检出结果，这样"图里没有鸟但有狗"仍能如实返回狗的类别与置信度。

## 下载与上传的体积上限

`url` / `sound_url` 由服务端拉取，除了 HTTP 头里的 `Content-Length` 预检，读取过程中还会累计字节数，
超过 `max_upload_image_mb`（默认 2）或 `max_upload_sound_mb`（默认 5）立即中止，避免恶意或异常 URL 把进程内存打爆。

## 扩展字段（非懂鸟规范）

图片与声音两条链路都支持可选的 `latitude` / `longitude` / `week` 表单字段，用于叠加 GeoModel 地理与季节先验：

- `latitude`：-90 ~ 90，缺失或非法则跳过地理过滤；
- `longitude`：-180 ~ 180，与 `latitude` 同时生效；
- `week`：1 ~ 48（缺省按当天换算），非法值按当天换算。

懂鸟规范里的 `area` 地区码也能用：只给 `area` 而不给经纬度时，服务会取该地区的代表坐标
（见 `configs/regions.yaml`，例如 `CN-BJ` -> 39.90, 116.40）参与地理先验；
未知地区码忽略，行为与不传完全一致。显式经纬度优先于 `area`。

只给 `area` 时地理先验用更保守的 `npu.area_geo_weight`（默认 0.5）：显式经纬度是用户现场的真实位置，
可以直接用满权重；`area` 只能映射到一个省级代表点，满权重会把「同属近缘种」按省份分布强行拉开。
实测一张英国拍摄的蓝山雀照片，`area=CN-BJ` 在满权重下会给出 60% 的 Azure Tit（错误），降到 0.5 后
恢复为 95.1% 的 Eurasian Blue Tit。

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
