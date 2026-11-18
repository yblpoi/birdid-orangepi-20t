"""配置加载。所有路径相对于仓库根目录解析，密钥只从环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    workers: int
    max_upload_image_mb: int
    max_upload_sound_mb: int
    max_image_side: int
    min_image_side: int
    max_queue: int


@dataclass(frozen=True)
class AuthConfig:
    api_key_env: str
    header_name: str
    form_field: str

    @property
    def api_key(self) -> str | None:
        value = os.environ.get(self.api_key_env, "").strip()
        return value or None


@dataclass(frozen=True)
class StorageConfig:
    result_ttl_seconds: int
    backend: str
    sqlite_path: str


@dataclass(frozen=True)
class NpuConfig:
    device_id: int
    model_dir: str
    seg_model: str
    cls_model: str
    seg_onnx: str
    cls_onnx: str
    classes_file: str
    seg_input_size: int
    cls_input_size: int
    decode_max_side: int
    seg_conf_threshold: float
    seg_iou_threshold: float
    crop_margin_ratio: float
    max_detections: int
    species_map_file: str
    geo_weight: float
    area_geo_weight: float
    # 鸟门禁单独的低阈值：小目标与翻拍图上鸟类分数会掉到 5%~15%，
    # 用与其它类别相同的 0.25 会直接漏检并掉进非鸟分支。
    seg_bird_conf_threshold: float = 0.02


@dataclass(frozen=True)
class SoundConfig:
    model_dir: str
    birdnet_model: str
    birdnet_om: str
    birdnet_front_onnx: str
    geomodel_model: str
    birdnet_backend: str
    geomodel_backend: str
    sample_rate: int
    window_seconds: float
    overlap: float
    top_k: int
    min_confidence: float
    geomodel_threshold: float
    labels_file: str
    geomodel_labels_file: str
    # 可选：NPU 侧的 .om 模型名；留空或缺文件时自动回退 CPU。
    geomodel_om: str = ""
    birdnet_mel_om: str = ""
    birdnet_mel_batch_om: str = ""
    # 可选：把 CNN 主干切成 trunk + head 两段。head 走批量化后，4 窗口只需
    # 一次前向，比整图逐窗串行快约 1.38 倍（见 sound_pipeline 注释）。
    birdnet_trunk_om: str = ""
    # 逗号分隔的多个批量档，例如 "head_b4.om,head_b8.om,head_b12.om"。
    # 批量越大单位窗口成本越低，按窗口数自动挑最优组合（见 head_batch_plan）。
    birdnet_head_batch_oms: str = ""



@dataclass(frozen=True)
class LoggingConfig:
    level: str
    access_log: bool


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    auth: AuthConfig
    storage: StorageConfig
    npu: NpuConfig
    sound: SoundConfig
    logging: LoggingConfig
    regions: dict[str, str]
    region_coords: dict[str, tuple[float, float]]

    def resolve(self, relative: str) -> Path:
        """把配置里的相对路径解析为仓库内的绝对路径。"""
        path = Path(relative)
        return path if path.is_absolute() else REPO_ROOT / path


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是映射")
    return value


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """加载 YAML 配置；未给出的路径使用仓库内默认配置。"""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    regions_raw = {}
    regions_file = REPO_ROOT / "configs" / "regions.yaml"
    if regions_file.exists():
        with regions_file.open("r", encoding="utf-8") as handle:
            regions_raw = (yaml.safe_load(handle) or {}).get("regions") or {}

    regions: dict[str, str] = {}
    region_coords: dict[str, tuple[float, float]] = {}
    for code, value in regions_raw.items():
        key = str(code)
        if isinstance(value, dict):
            regions[key] = str(value.get("name") or key)
            lat, lon = value.get("lat"), value.get("lon")
            if lat is not None and lon is not None:
                region_coords[key] = (float(lat), float(lon))
        else:
            regions[key] = str(value)

    return Config(
        server=ServerConfig(**_section(raw, "server")),
        auth=AuthConfig(**_section(raw, "auth")),
        storage=StorageConfig(**_section(raw, "storage")),
        npu=NpuConfig(**_section(raw, "npu")),
        sound=SoundConfig(**_section(raw, "sound")),
        logging=LoggingConfig(**_section(raw, "logging")),
        regions=regions,
        region_coords=region_coords,
    )
