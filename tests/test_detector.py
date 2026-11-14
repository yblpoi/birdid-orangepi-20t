"""检测后处理单元测试：验证 end2end 输出解码与 NMS。"""

from __future__ import annotations

import numpy as np

from birdid.detector import COCO_NAMES, decode_predictions, nms
from birdid.preprocess import LetterboxResult


def _letterbox_identity(size: int) -> LetterboxResult:
    return LetterboxResult(image=np.zeros((1, 3, size, size), dtype=np.float32), ratio=1.0, pad_x=0.0, pad_y=0.0)


def test_decode_reads_end2end_columns() -> None:
    raw = np.zeros((1, 300, 38), dtype=np.float32)
    raw[0, 0, :6] = [10.0, 20.0, 110.0, 220.0, 0.9, 14.0]
    detections = decode_predictions(raw, _letterbox_identity(640), (640, 640), 0.25, 0.45, 5)
    assert len(detections) == 1
    assert detections[0].is_bird
    assert detections[0].class_name == "bird"
    assert detections[0].box == (10.0, 20.0, 110.0, 220.0)


def test_decode_returns_empty_below_threshold() -> None:
    raw = np.zeros((1, 300, 38), dtype=np.float32)
    raw[0, 0, :6] = [0.0, 0.0, 10.0, 10.0, 0.01, 14.0]
    assert decode_predictions(raw, _letterbox_identity(640), (640, 640), 0.25, 0.45, 5) == []


def test_decode_undoes_letterbox_padding() -> None:
    raw = np.zeros((1, 300, 38), dtype=np.float32)
    raw[0, 0, :6] = [100.0, 100.0, 200.0, 200.0, 0.8, 14.0]
    info = LetterboxResult(image=np.zeros((1, 3, 640, 640), dtype=np.float32), ratio=0.5, pad_x=20.0, pad_y=10.0)
    detections = decode_predictions(raw, info, (1280, 1280), 0.25, 0.45, 5)
    assert detections[0].box == (160.0, 180.0, 360.0, 380.0)


def test_nms_suppresses_overlap() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.45, 5) == [0, 2]


def test_coco_bird_index_is_14() -> None:
    assert COCO_NAMES[14] == "bird"