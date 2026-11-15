"""HTTP 层测试：用替身 AppState 验证协议分支，不加载真实模型。"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from birdid import app as app_module
from birdid.protocol import CODE_NO_ANIMALS, CODE_OK, CODE_PROCESSING, CODE_QUOTA


class _StubExecutor:
    """替身任务执行器：同步跑完，返回固定 resultid。"""

    def __init__(self) -> None:
        self.payloads: dict[str, Any] = {}
        self.call_count = 0

    def submit(self, kind: str, class_code: str, work, accepted_payload: Any = None) -> str:
        del kind, class_code, accepted_payload
        self.call_count += 1
        outcome = work()
        result_id = f"HH-B_test_{self.call_count}"
        self.payloads[result_id] = outcome
        return result_id

    def get(self, result_id: str) -> dict | None:
        if result_id not in self.payloads:
            return None
        outcome = self.payloads[result_id]
        payload = [CODE_NO_ANIMALS, "No animals"] if outcome is None else [CODE_OK, outcome]
        return {"state": "done", "payload": payload}

    @property
    def pending(self) -> int:
        return 0

    def saturated(self) -> bool:
        return False

    def shutdown(self) -> None:
        return None


class _StubImagePipeline:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def predict(self, image, box_scale=1.0, geo=(None, None, None)):
        self.seen.append(image.size)

        class _Result:
            entries = [
                {
                    "box": [1, 2, 3, 4],
                    "list": [[99.9, "Turdus merula|Eurasian Blackbird|Turdus merula", 7, "B"]],
                }
            ]

        return _Result()

    def close(self) -> None:
        return None


class _StubSoundPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def predict(self, raw, latitude=None, longitude=None, week=None):
        self.calls.append((len(raw), latitude, longitude, week))
        return [{"box": [], "list": [[88.0, "Turdus merula|Eurasian Blackbird|Turdus merula", 3, "B"]]}]

    def close(self) -> None:
        return None


class _StubState:
    def __init__(self, config) -> None:
        self.config = config
        self.executor = _StubExecutor()
        self.image_pipeline = _StubImagePipeline()
        self.sound_pipeline = _StubSoundPipeline()

    def close(self) -> None:
        return None


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "AppState", _StubState)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "server:",
                "  host: 127.0.0.1",
                "  port: 8080",
                "  workers: 1",
                "  max_upload_image_mb: 2",
                "  max_upload_sound_mb: 5",
                "  max_image_side: 8192",
                "  min_image_side: 11",
                "  max_queue: 4",
                "auth:",
                "  api_key_env: BIRDID_TEST_KEY",
                "  header_name: api_key",
                "  form_field: api_key",
                "storage:",
                "  result_ttl_seconds: 60",
                "  backend: memory",
                "  sqlite_path: data/test.db",
                "npu:",
                "  device_id: 0",
                "  model_dir: models",
                "  seg_model: seg.om",
                "  cls_model: cls.om",
                "  seg_onnx: seg.onnx",
                "  cls_onnx: cls.onnx",
                "  classes_file: models/classes.json",
                "  seg_input_size: 640",
                "  cls_input_size: 384",
                "  seg_conf_threshold: 0.25",
                "  seg_iou_threshold: 0.45",
                "  crop_margin_ratio: 0.55",
                "  decode_max_side: 1280",
                "  max_detections: 5",
                "  species_map_file: models/species_map.json",
                "  geo_weight: 1.0",
                "sound:",
                "  model_dir: models",
                "  birdnet_model: birdnet.onnx",
                "  birdnet_om: birdnet.om",
                "  birdnet_front_onnx: birdnet_front.onnx",
                "  geomodel_model: geo.onnx",
                "  birdnet_backend: cpu",
                "  geomodel_backend: cpu",
                "  sample_rate: 32000",
                "  window_seconds: 3.0",
                "  overlap: 0.0",
                "  top_k: 3",
                "  min_confidence: 0.1",
                "  geomodel_threshold: 0.03",
                "  labels_file: models/labels.csv",
                "  geomodel_labels_file: models/geo.txt",
                "logging:",
                "  level: INFO",
                "  access_log: false",
            ]
        ),
        encoding="utf-8",
    )
    with TestClient(app_module.create_app(str(config_path))) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_quota(client: TestClient) -> None:
    code, payload = client.post("/dongniao", data={"quota": "1"}).json()
    assert code == CODE_OK
    assert payload["image"] > 0


def test_missing_parameters(client: TestClient) -> None:
    code, _ = client.post("/dongniao", data={}).json()
    assert code == CODE_PROCESSING


def test_unknown_result_id_is_processing(client: TestClient) -> None:
    body = client.post("/dongniao", data={"resultid": "nope"}).json()
    assert body == [CODE_PROCESSING, "nope"]


def test_image_upload_returns_result_id(client: TestClient) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(buffer, format="JPEG")
    code, result_id = client.post(
        "/dongniao",
        data={"upload": "1"},
        files={"image": ("bird.jpg", buffer.getvalue(), "image/jpeg")},
    ).json()
    assert code == CODE_PROCESSING
    code, payload = client.post("/dongniao", data={"resultid": result_id}).json()
    assert code == CODE_OK
    assert payload[0]["list"][0][3] == "B"


def test_image_too_small_is_no_animals(client: TestClient) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    result_id = client.post(
        "/dongniao",
        files={"image": ("tiny.png", buffer.getvalue(), "image/png")},
    ).json()[1]
    code, payload = client.post("/dongniao", data={"resultid": result_id}).json()
    assert [code, payload] == [CODE_NO_ANIMALS, "No animals"]


def test_sound_geo_parameters_forwarded(client: TestClient) -> None:
    state = client.app.state.birdid
    buffer = b"RIFF" + b"\x00" * 64
    client.post(
        "/dongniao",
        data={"latitude": "39.9", "longitude": "116.4"},
        files={"sound": ("a.wav", buffer, "audio/wav")},
    )
    assert state.sound_pipeline.calls[0][1] == 39.9
    assert state.sound_pipeline.calls[0][3] == 42


def test_api_key_enforced(monkeypatch, client: TestClient) -> None:
    monkeypatch.setenv("BIRDID_TEST_KEY", "secret")
    code, _ = client.post("/dongniao", data={"quota": "1"}).json()
    assert code == CODE_QUOTA
    code, _ = client.post("/dongniao", data={"quota": "1"}, headers={"api_key": "secret"}).json()
    assert code == CODE_OK


def test_queue_full_returns_busy(monkeypatch, client: TestClient) -> None:
    """队列打满时应立刻返回 1007，而不是让请求排队到超时。"""
    import io

    from PIL import Image

    state = client.app.state.birdid
    monkeypatch.setattr(state.executor, "submit", lambda *a, **k: None)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(buffer, format="JPEG")
    body = client.post(
        "/dongniao",
        files={"image": ("bird.jpg", buffer.getvalue(), "image/jpeg")},
    ).json()
    assert body == [CODE_QUOTA, "server busy"]

def test_download_rejects_oversized_payload() -> None:
    """远端声称或实际超大时都要中止，避免 URL 提交把内存打爆。"""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import pytest

    payload = b"x" * 4096

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:" + str(server.server_port) + "/x.jpg"
        with pytest.raises(ValueError):
            app_module._download(url, 5.0, 1024)
        assert app_module._download(url, 5.0, 8192) == payload
    finally:
        server.shutdown()
        server.server_close()

def test_class_filter_is_opt_in() -> None:
    """不传 class 时保留非鸟结果，显式传 B 时才把非鸟条目裁掉。"""
    entries = [
        {"box": [1, 2, 3, 4], "list": [[99.0, "狗|dog|dog", 16, "M"]]},
        {"box": [5, 6, 7, 8], "list": [[90.0, "鸟|bird|bird", 14, "B"]]},
    ]
    assert app_module._filter_by_class(entries, None) == entries
    kept = app_module._filter_by_class(entries, "B")
    assert [entry["list"][0][3] for entry in kept] == ["B"]
    assert app_module._filter_by_class(entries, "M")[0]["list"][0][2] == 16
    assert app_module._filter_by_class(entries, "A") == []

def test_area_code_becomes_geo_hint(client, monkeypatch) -> None:
    """只给懂鸟的 area 地区码时，也要能换成代表坐标做地理先验。"""
    config = app_module.load_config(None)
    assert config.region_coords["CN-BJ"] == (39.9, 116.4)
    lat, lon, week = app_module._parse_geo({"area": "CN-BJ"}, config)
    assert (lat, lon) == (39.9, 116.4)
    assert 1 <= week <= 48
    # 未知地区码不参与地理先验，行为与不传一致。
    assert app_module._parse_geo({"area": "ZZ-99"}, config) == (None, None, None)
    # 显式经纬度优先于 area。
    lat2, lon2, _ = app_module._parse_geo({"area": "CN-BJ", "latitude": "51.5", "longitude": "-0.1"}, config)
    assert (lat2, lon2) == (51.5, -0.1)
