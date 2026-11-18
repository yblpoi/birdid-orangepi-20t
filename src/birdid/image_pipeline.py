"""图片识别链路：YOLO26-seg 门禁裁剪 - FastViT 物种分类。

流程：
  1. 图片 letterbox 到 640x640，送入 YOLO26-seg（NPU）得到检测框。
  2. 若存在鸟类框，按 crop_margin_ratio 外扩裁剪，逐框送入 FastViT（NPU）分类。
  3. 若不存在鸟类框，返回非鸟类的具体类别与置信度（例如 dog 99%）。
  4. 完全没有检测目标时返回 None 的占位结果，由上层转成 [1010, "No animals"]。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geo import GeoModel, rerank
from .runtime import build_runtime
from .classifier import SpeciesMap, SpeciesPrediction, SpeciesRegistry, load_species_map, scientific_name
from .detector import Detection, class_name, class_name_zh, decode_predictions, pick_targets
from .preprocess import crop_resize_for_classifier, letterbox


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
        self._species_map: SpeciesMap | None = load_species_map(
            config.resolve(npu.species_map_file) if npu.geo_weight > 0 else None
        )
        self._geo: GeoModel | None = None
        if self._species_map is not None:
            geo_path = config.resolve(config.sound.model_dir) / config.sound.geomodel_model
            if geo_path.exists():
                geo_om = config.sound.geomodel_om
                geo_om_path = config.resolve(config.sound.model_dir) / geo_om if geo_om else None
                self._geo = GeoModel(
                    geo_path,
                    config.sound.geomodel_backend,
                    geo_om_path,
                    npu.device_id,
                )
            else:
                logging.warning("未找到 %s，图片链路不做地理重排", geo_path)

    def close(self) -> None:
        """释放 NPU 资源。"""
        self._seg.close()
        self._cls.close()
        if self._geo is not None:
            self._geo.close()

    def predict(
        self,
        image: Image.Image,
        box_scale: float = 1.0,
        geo: tuple = (None, None, None),
        geo_weight: float | None = None,
    ) -> ImageResult:
        """对整图执行门禁与分类。

        box_scale 把检测框坐标换算回原图：解码时若按 1/2 或 1/4 降采样，
        传 2.0 或 4.0 即可，保证返回的 box 始终是原图像素坐标。
        geo 为可选的 (纬度, 经度, 周)，三者齐全时用 GeoModel 做地理重排。
        """
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
            bird_threshold=npu.seg_bird_conf_threshold,
        )
        birds, others = pick_targets(detections, npu.max_detections)
        if birds:
            entries = [
                self._classify_bird(source, det, box_scale, geo, geo_weight) for det in birds
            ]
            return ImageResult(entries=entries, has_bird=True)
        if others:
            entries = [self._describe_non_bird(det, box_scale) for det in others]
            return ImageResult(entries=entries, has_bird=False)
        return ImageResult(entries=[], has_bird=False)

    def _classify_bird(
        self,
        image: Image.Image,
        detection: Detection,
        box_scale: float = 1.0,
        geo: tuple = (None, None, None),
        geo_weight: float | None = None,
    ) -> dict[str, Any]:
        """裁剪鸟类区域并分类，返回懂鸟结果条目。"""
        npu = self._config.npu
        tensor = crop_resize_for_classifier(
            image, detection.box, npu.crop_margin_ratio, npu.cls_input_size
        )
        logits = self._cls.infer(tensor)[0]
        # 地理重排需要更大的候选池：先取 max_detections 个，重排后再截 top_k。
        candidates = self._registry.top_k(logits, max(npu.max_detections, self._config.sound.top_k))
        predictions = self._apply_geo(candidates, geo, geo_weight)[: self._config.sound.top_k]
        items = [self._species_item(prediction) for prediction in predictions]
        box = [round(v * box_scale) for v in detection.box]
        return {"box": box, "list": items}

    def _apply_geo(
        self,
        predictions: list[SpeciesPrediction],
        geo: tuple,
        geo_weight: float | None = None,
    ) -> list[SpeciesPrediction]:
        """用 GeoModel 地理先验重排候选；条件不足时原样返回。"""
        if self._geo is None or self._species_map is None:
            return predictions
        latitude, longitude, week = geo
        if latitude is None or longitude is None or week is None:
            return predictions
        weight = self._config.npu.geo_weight if geo_weight is None else geo_weight
        if weight <= 0:
            return predictions
        spatial = self._geo.scores(latitude, longitude, week)
        ranked = rerank(
            [(p.index, p.confidence) for p in predictions],
            spatial,
            self._species_map.geomodel_index,
            weight,
        )
        by_index = {p.index: p for p in predictions}
        adjusted: list[SpeciesPrediction] = []
        for index, confidence in ranked:
            base = by_index[index]
            adjusted.append(
                SpeciesPrediction(
                    index=base.index,
                    species_id=base.species_id,
                    species_name=base.species_name,
                    confidence=confidence,
                )
            )
        return adjusted

    @staticmethod
    def _species_item(prediction: SpeciesPrediction) -> list[Any]:
        """构造 [置信度, "中文|英文|学名", animalid, class]。"""
        latin = scientific_name(prediction.species_name)
        label = f"{latin}|{latin}|{latin}"
        confidence = round(prediction.confidence, 1)
        return [confidence, label, prediction.index, "B"]

    @staticmethod
    def _describe_non_bird(
        detection: Detection, box_scale: float = 1.0
    ) -> dict[str, Any]:
        """非鸟类目标：返回具体类别与置信度。"""
        english = class_name(detection.class_id)
        chinese = class_name_zh(detection.class_id)
        label = f"{chinese}|{english}|{english}"
        confidence = round(detection.score * 100.0, 1)
        return {
            "box": [round(v * box_scale) for v in detection.box],
            "list": [[confidence, label, detection.class_id, "M"]],
        }