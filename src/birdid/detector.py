"""YOLO26-seg 门禁与裁剪。

推理走 NPU（.om）；后处理在 CPU 上完成：解析 end2end 检测输出（xyxy + 置信度
+ 类别 + 掩码系数），执行 NMS，并还原 letterbox 坐标。

YOLO26 的导出头部为 NMS-free 的 one2one 分支，输出形状 (1, 300, 38)：
  第 0~3 列 = x1, y1, x2, y2（letterbox 输入坐标系，像素）
  第 4 列   = 置信度（已过 sigmoid）
  第 5 列   = 类别下标（COCO 80 类）
  第 6~37 列 = 32 个掩码系数（本服务只需检测框，不使用掩码）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import LetterboxResult


@dataclass(frozen=True)
class Detection:
    """单个检测结果。"""

    box: tuple[float, float, float, float]   # 原图坐标系 x1,y1,x2,y2
    score: float
    class_id: int
    class_name: str
    is_bird: bool


# COCO 80 类名称，与 YOLO26-seg 导出顺序一致。
COCO_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)

BIRD_CLASS_ID = 14
BIRD_CLASS_NAME = "bird"

# 中文名，输出给懂鸟协议的 names 字段用；缺失时回退英文名。
COCO_NAMES_ZH: dict[int, str] = {
    0: "人", 1: "自行车", 2: "汽车", 3: "摩托车", 4: "飞机", 5: "公交车", 6: "火车", 7: "卡车",
    8: "船", 9: "红绿灯", 10: "消防栓", 11: "停车标志", 12: "停车计时器", 13: "长椅",
    14: "鸟", 15: "猫", 16: "狗", 17: "马", 18: "羊", 19: "牛", 20: "大象", 21: "熊",
    22: "斑马", 23: "长颈鹿", 24: "背包", 25: "雨伞", 26: "手提包", 27: "领带", 28: "行李箱",
    29: "飞盘", 30: "滑雪板", 31: "单板滑雪", 32: "球", 33: "风筝", 34: "棒球棒",
    35: "棒球手套", 36: "滑板", 37: "冲浪板", 38: "网球拍", 39: "瓶子", 40: "酒杯", 41: "杯子",
    42: "叉子", 43: "刀", 44: "勺子", 45: "碗", 46: "香蕉", 47: "苹果", 48: "三明治", 49: "橙子",
    50: "西兰花", 51: "胡萝卜", 52: "热狗", 53: "披萨", 54: "甜甜圈", 55: "蛋糕", 56: "椅子",
    57: "沙发", 58: "盆栽", 59: "床", 60: "餐桌", 61: "马桶", 62: "电视", 63: "笔记本电脑",
    64: "鼠标", 65: "遥控器", 66: "键盘", 67: "手机", 68: "微波炉", 69: "烤箱", 70: "烤面包机",
    71: "水槽", 72: "冰箱", 73: "书", 74: "时钟", 75: "花瓶", 76: "剪刀", 77: "毛绒玩具",
    78: "吹风机", 79: "牙刷",
}


def class_name(class_id: int) -> str:
    """COCO 类别英文名。"""
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return f"class_{class_id}"


def class_name_zh(class_id: int) -> str:
    """COCO 类别中文名。"""
    return COCO_NAMES_ZH.get(class_id, class_name(class_id))


def decode_predictions(
    raw: np.ndarray,
    letterbox_info: LetterboxResult,
    original_size: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    bird_threshold: float | None = None,
) -> list[Detection]:
    """把 YOLO26-seg 的原始输出解码为原图坐标系的检测框。

    bird_threshold 给鸟门禁单独一个（更低的）阈值：小目标、逆光、屏幕翻拍等
    场景下鸟类的分数会掉到 5%~15%，而 COCO 里其它类别在同一张图上往往同时
    给出错框。给鸟单独放宽阈值，既能把这类图送进物种分类，又不会让非鸟图
    误触发——实测非鸟图鸟类分数长期低于 1%。
    """
    preds = np.asarray(raw, dtype=np.float32)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.ndim != 2 or preds.shape[1] < 6:
        raise ValueError(f"无法识别的检测输出形状: {preds.shape}")

    boxes_xyxy = preds[:, :4]
    confidences = preds[:, 4]
    class_ids = preds[:, 5].round().astype(np.int64)

    if bird_threshold is None:
        keep = confidences >= conf_threshold
    else:
        thresholds = np.full(class_ids.shape, conf_threshold, dtype=np.float32)
        thresholds[class_ids == BIRD_CLASS_ID] = min(bird_threshold, conf_threshold)
        keep = confidences >= thresholds
    if not np.any(keep):
        return []
    boxes_xyxy = boxes_xyxy[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    order = np.argsort(-confidences)
    boxes_xyxy = boxes_xyxy[order]
    confidences = confidences[order]
    class_ids = class_ids[order]

    selected = nms(boxes_xyxy, confidences, iou_threshold, max_detections, class_ids)

    orig_w, orig_h = original_size
    pad_x, pad_y = letterbox_info.pad_x, letterbox_info.pad_y
    ratio = letterbox_info.ratio
    results: list[Detection] = []
    for index in selected:
        dx1, dy1, dx2, dy2 = boxes_xyxy[index]
        # 还原 letterbox：先减 padding，再除以缩放比
        ox1 = (dx1 - pad_x) / ratio
        oy1 = (dy1 - pad_y) / ratio
        ox2 = (dx2 - pad_x) / ratio
        oy2 = (dy2 - pad_y) / ratio
        ox1, oy1 = max(0.0, float(ox1)), max(0.0, float(oy1))
        ox2, oy2 = min(float(orig_w), float(ox2)), min(float(orig_h), float(oy2))
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        cid = int(class_ids[index])
        results.append(
            Detection(
                box=(ox1, oy1, ox2, oy2),
                score=float(confidences[index]),
                class_id=cid,
                class_name=class_name(cid),
                is_bird=cid == BIRD_CLASS_ID,
            )
        )
    return results


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """计算两两 IoU 矩阵。"""
    x1 = boxes[:, None, 0]
    y1 = boxes[:, None, 1]
    x2 = boxes[:, None, 2]
    y2 = boxes[:, None, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    inter_x1 = np.maximum(x1, x1.T)
    inter_y1 = np.maximum(y1, y1.T)
    inter_x2 = np.minimum(x2, x2.T)
    inter_y2 = np.minimum(y2, y2.T)
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    union = areas + areas.T - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int,
    class_ids: np.ndarray | None = None,
) -> list[int]:
    """标准非极大值抑制，返回保留下标。

    给定 class_ids 时按类别分组抑制：不同类别的框即使大面积重叠也互不压制。
    门禁为了让小目标鸟过线把鸟的阈值放得很低，同一位置常有置信度更高的人、
    背景大框，类别无关的抑制会把刚捞回来的鸟框整片压掉。
    """
    if boxes.shape[0] == 0:
        return []
    ious = iou_matrix(boxes)
    if class_ids is None:
        same_class = np.ones(ious.shape, dtype=bool)
    else:
        same_class = class_ids[:, None] == class_ids[None, :]
    suppressed = np.zeros(boxes.shape[0], dtype=bool)
    keep: list[int] = []
    for index in range(boxes.shape[0]):
        if suppressed[index]:
            continue
        keep.append(index)
        if len(keep) >= max_detections:
            break
        suppressed |= (ious[index] > iou_threshold) & same_class[index]
        suppressed[index] = False
    return keep


def pick_targets(detections: list[Detection], max_targets: int) -> tuple[list[Detection], list[Detection]]:
    """分离鸟类与非鸟类检测结果。"""
    birds = [d for d in detections if d.is_bird][:max_targets]
    others = [d for d in detections if not d.is_bird][:max_targets]
    return birds, others