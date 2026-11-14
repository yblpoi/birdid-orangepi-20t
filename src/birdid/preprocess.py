"""图像预处理：解码、letterbox、归一化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxResult:
    """letterbox 结果，含还原到原图所需的缩放与偏移。"""

    image: np.ndarray      # (1, 3, size, size) float32，RGB，取值 0~1
    ratio: float
    pad_x: float
    pad_y: float


def letterbox(image: Image.Image, size: int) -> LetterboxResult:
    """等比缩放并居中填充到 size×size。"""
    source = image.convert("RGB")
    width, height = source.size
    ratio = min(size / width, size / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = source.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))[None, ...]
    return LetterboxResult(image=np.ascontiguousarray(chw), ratio=ratio, pad_x=pad_x, pad_y=pad_y)


def normalize_imagenet(chw: np.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]) -> np.ndarray:
    """按 ImageNet 均值方差归一化（BirdTrace 的 ONNX 已内嵌归一化，仅在自定义导出时使用）。"""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((chw - mean_arr) / std_arr)


def crop_with_margin(image: Image.Image, box: tuple[float, float, float, float], margin_ratio: float) -> Image.Image:
    """按比例外扩裁剪框并裁剪，保留一部分边框以提升分类效果。"""
    width, height = image.size
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    mx, my = box_w * margin_ratio, box_h * margin_ratio
    left = max(0, int(round(x1 - mx)))
    top = max(0, int(round(y1 - my)))
    right = min(width, int(round(x2 + mx)))
    bottom = min(height, int(round(y2 + my)))
    if right <= left or bottom <= top:
        return image.convert("RGB")
    return image.crop((left, top, right, bottom)).convert("RGB")


def resize_for_classifier(image: Image.Image, size: int) -> np.ndarray:
    """分类器输入：短边等比缩放后中心裁剪，返回 (1,3,size,size)。"""
    source = image.convert("RGB")
    width, height = source.size
    scale = size / min(width, height)
    new_size = (max(size, int(round(width * scale))), max(size, int(round(height * scale))))
    resized = source.resize(new_size, Image.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1))[None, ...])
