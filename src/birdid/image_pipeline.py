"""图片识别链路：YOLO26-seg 门禁裁剪 - FastViT 物种分类。

流程：
  1. 图片 letterbox 到 640x640，送入 YOLO26-seg（NPU）得到检测框。
  2. 若存在鸟类框，按 crop_margin_ratio 外扩裁剪，逐框送入 FastViT（NPU）分类。
  3. 若不存在鸟类框，返回非鸟类的具体类别与置信度（例如 dog 99%）。
  4. 完全没有检测目标时返回 None 的占位结果，由上层转成 [1010, "No animals"]。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .runtime import build_runtime
from .classifier import SpeciesPrediction, SpeciesRegistry, scientific_name
from .detector import Detection, class_name, class_name_zh, decode_predictions, pick_targets
from .preprocess import crop_with_margin, letterbox, resize_for_classifier


@dataclass
class ImageResult:
    """图片识别结果。"""

    entries: list[dict[str, Any]]
    has_bird: bool


class ImagePipeline:
    """封装两个 NPU 模型，串成一次图片识别。"""

    def __init__(self, config, registry: SpeciesRegistry) -> None:
        self._config = config
        self._registry = registry
        npu = config.npu
        model_dir = config.resolve(npu.model_dir)
        self._seg = build_runtime(
            model_dir, npu.seg_model, npu.seg_onnx, npu.device_id
        )
        self._cls = build_runtime(
            model_dir, npu.cls_model, npu.cls_onnx, npu.device_id
        )

    def close(self) -> None:
        """释放 NPU 资源。"""
        self._seg.close()
        self._cls.close()

    def predict(self, image: Image.Image) -> ImageResult:
        """对整图执行门禁与分类。"""
        npu = self._config.npu
        source = image.convert("RGB")
        boxed = letterbox(source, npu.seg_input_size)
        raw = self._seg.infer(boxed.image)[0]
        detections = decode_predictions(
            raw,
            boxed,
            source.size,
            npu.seg_conf_threshold,
            npu.seg_iou_threshold,
            npu.max_detections,
        )
        birds, others = pick_targets(detections, npu.max_detections)
        if birds:
            entries = [self._classify_bird(source, det) for det in birds]
            return ImageResult(entries=entries, has_bird=True)
        if others:
            entries = [self._describe_non_bird(det) for det in others]
            return ImageResult(entries=entries, has_bird=False)
        return ImageResult(entries=[], has_bird=False)

    def _classify_bird(self, image: Image.Image, detection: Detection) -> dict[str, Any]:
        """裁剪鸟类区域并分类，返回懂鸟结果条目。"""
        crop = crop_with_margin(image, detection.box, self._config.npu.crop_margin_ratio)
        tensor = resize_for_classifier(crop, self._config.npu.cls_input_size)
        logits = self._cls.infer(tensor)[0]
        predictions = self._registry.top_k(logits, self._config.sound.top_k)
        items = [self._species_item(prediction) for prediction in predictions]
        return {"box": [round(v) for v in detection.box], "list": items}

    @staticmethod
    def _species_item(prediction: SpeciesPrediction) -> list[Any]:
        """构造 [置信度, "中文|英文|学名", animalid, class]。"""
        latin = scientific_name(prediction.species_name)
        label = f"{latin}|{latin}|{latin}"
        confidence = round(prediction.confidence, 1)
        return [confidence, label, prediction.index, "B"]

    @staticmethod
    def _describe_non_bird(detection: Detection) -> dict[str, Any]:
        """非鸟类目标：返回具体类别与置信度。"""
        english = class_name(detection.class_id)
        chinese = class_name_zh(detection.class_id)
        label = f"{chinese}|{english}|{english}"
        confidence = round(detection.score * 100.0, 1)
        return {
            "box": [round(v) for v in detection.box],
            "list": [[confidence, label, detection.class_id, "M"]],
        }