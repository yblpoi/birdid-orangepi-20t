"""图像预处理：letterbox、归一化、裁剪与分类器输入构造。

这些函数都在 CPU 上跑，是端到端延迟里不可忽视的一块。实现上刻意避免
多余的整图拷贝：调用方传进来的已经是 RGB，内部不再重复 convert；letterbox
也不再先合成一张 uint8 画布再转 float，而是直接写进目标 float32 张量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# letterbox 填充色，与 YOLO 训练时的 114 灰一致。
PAD_VALUE = 114
PAD_NORMALIZED = PAD_VALUE / 255.0
INV_255 = np.float32(1.0 / 255.0)


@dataclass(frozen=True)
class LetterboxResult:
    """letterbox 结果，含还原到原图所需的缩放与偏移。"""

    image: np.ndarray      # (1, 3, size, size) float32，RGB，取值 0~1
    ratio: float
    pad_x: float
    pad_y: float


def _as_rgb(image: Image.Image) -> Image.Image:
    """确保是 RGB；已经是 RGB 时直接返回，避免整图拷贝。"""
    return image if image.mode == "RGB" else image.convert("RGB")


def _to_chw(source: Image.Image, size: int, offset: tuple[int, int] | None) -> np.ndarray:
    """把 PIL 图像转成 (1,3,size,size) float32；offset 给定时贴到该位置，其余填灰。"""
    patch = np.multiply(np.asarray(source).transpose(2, 0, 1), INV_255, dtype=np.float32)
    if offset is None and patch.shape[1] == size and patch.shape[2] == size:
        return np.ascontiguousarray(patch[None, ...])
    canvas = np.empty((1, 3, size, size), dtype=np.float32)
    canvas.fill(PAD_NORMALIZED)
    height, width = patch.shape[1], patch.shape[2]
    left, top = offset if offset is not None else ((size - width) // 2, (size - height) // 2)
    canvas[0, :, top:top + height, left:left + width] = patch
    return canvas


def letterbox(image: Image.Image, size: int) -> LetterboxResult:
    """等比缩放并居中填充到 size×size。"""
    source = _as_rgb(image)
    width, height = source.size
    ratio = min(size / width, size / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = source.resize((new_w, new_h), Image.BILINEAR, reducing_gap=2.0)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    chw = _to_chw(resized, size, (int(pad_x), int(pad_y)))
    return LetterboxResult(image=chw, ratio=ratio, pad_x=pad_x, pad_y=pad_y)


def normalize_imagenet(chw: np.ndarray, mean: tuple[float, float, float], std: tuple[float, float, float]) -> np.ndarray:
    """按 ImageNet 均值方差归一化（BirdTrace 的 ONNX 已内嵌归一化，仅在自定义导出时使用）。"""
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.ascontiguousarray((chw - mean_arr) / std_arr)


def margin_box(
    image: Image.Image,
    box: tuple[float, float, float, float],
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """把检测框按比例外扩并夹到图内，返回整数裁剪框。"""
    width, height = image.size
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * margin_ratio
    my = (y2 - y1) * margin_ratio
    left = max(0, int(round(x1 - mx)))
    top = max(0, int(round(y1 - my)))
    right = min(width, int(round(x2 + mx)))
    bottom = min(height, int(round(y2 + my)))
    return left, top, right, bottom


def crop_with_margin(image: Image.Image, box: tuple[float, float, float, float], margin_ratio: float) -> Image.Image:
    """按比例外扩裁剪框并裁剪，保留一部分边框以提升分类效果。"""
    source = _as_rgb(image)
    left, top, right, bottom = margin_box(source, box, margin_ratio)
    if right <= left or bottom <= top:
        return source
    return source.crop((left, top, right, bottom))


def resize_for_classifier(image: Image.Image, size: int) -> np.ndarray:
    """分类器输入：短边等比缩放后中心裁剪，返回 (1,3,size,size)。

    用双线性而非双三次：实测两者 top1 一致（双线性略高），但双线性在
    aarch64 上快约 40%，是这条链路里性价比最高的一处替换。
    """
    source = _as_rgb(image)
    width, height = source.size
    scale = size / min(width, height)
    new_size = (max(size, int(round(width * scale))), max(size, int(round(height * scale))))
    resized = source.resize(new_size, Image.BILINEAR, reducing_gap=2.0)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    return _to_chw(cropped, size, None)


def crop_resize_for_classifier(
    image: Image.Image,
    box: tuple[float, float, float, float],
    margin_ratio: float,
    size: int,
) -> np.ndarray:
    """外扩裁剪 + 短边等比缩放 + 中心裁剪，一步完成。

    语义等价于 crop_with_margin 后接 resize_for_classifier，但只做一次重采样：
    后者「先缩放到 new_size 再中心裁剪」等价于对缩放图上的一块中心区域采样，
    所以这里把该区域换算回原图坐标，用一次 Image.resize(box=...) 直接得到结果。
    省掉一次整块裁剪图的中间分配与一次多余的重采样。
    """
    source = _as_rgb(image)
    width, height = source.size
    left, top, right, bottom = margin_box(source, box, margin_ratio)
    if right <= left or bottom <= top:
        left, top, right, bottom = 0, 0, width, height
    crop_w, crop_h = right - left, bottom - top
    scale = size / min(crop_w, crop_h)
    scaled_w = max(size, int(round(crop_w * scale)))
    scaled_h = max(size, int(round(crop_h * scale)))
    off_x = (scaled_w - size) // 2
    off_y = (scaled_h - size) // 2
    center = (
        left + off_x / scale,
        top + off_y / scale,
        left + (off_x + size) / scale,
        top + (off_y + size) / scale,
    )
    resized = source.resize((size, size), Image.BILINEAR, box=center, reducing_gap=2.0)
    return _to_chw(resized, size, None)
