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

    return Config(
        server=ServerConfig(**_section(raw, "server")),
        auth=AuthConfig(**_section(raw, "auth")),
        storage=StorageConfig(**_section(raw, "storage")),
        npu=NpuConfig(**_section(raw, "npu")),
        sound=SoundConfig(**_section(raw, "sound")),
        logging=LoggingConfig(**_section(raw, "logging")),
        regions={str(k): str(v) for k, v in regions_raw.items()},
    )
