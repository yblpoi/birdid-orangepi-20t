"""懂鸟兼容的 HTTP 服务入口。

单端点 POST /dongniao，服务端按表单字段分发到不同处理分支，所有业务响应
都是 [code, payload] 二元数组，与懂鸟 v2 协议保持一致。
"""

from __future__ import annotations

import io
import logging
import os
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from . import __version__
from .classifier import SpeciesRegistry
from .config import Config, load_config
from .detector import class_name, class_name_zh
from .executor import TaskExecutor
from .image_pipeline import ImagePipeline
from .protocol import (
    CODE_OK,
    CODE_PROCESSING,
    CODE_QUOTA,
    MSG_BUSY,
    MSG_SOUND_URL_ERROR,
    MSG_URL_FAILED,
    error,
    normalize_class,
    no_animals,
    ok,
    processing,
)
from .sound import WEEKS_IN_YEAR
from .sound_pipeline import SoundPipeline
from .store import build_store

LOGGER = logging.getLogger("birdid")

HTTP_TIMEOUT_IMAGE = 5.0
HTTP_TIMEOUT_SOUND = 10.0
MAX_REDIRECTS = 2


class AppState:
    """进程内持有的模型与存储。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = build_store(
            config.storage.backend,
            config.resolve(config.storage.sqlite_path),
            config.storage.result_ttl_seconds,
        )
        self.executor = TaskExecutor(
            self.store,
            max_workers=config.server.workers,
            ttl_seconds=config.storage.result_ttl_seconds,
            max_queue=config.server.max_queue,
        )
        self.registry = SpeciesRegistry(config.resolve(config.npu.classes_file))
        self.image_pipeline = ImagePipeline(config, self.registry)
        self.sound_pipeline = SoundPipeline(config)

    def close(self) -> None:
        """释放模型与线程池。"""
        self.executor.shutdown()
        self.image_pipeline.close()
        self.sound_pipeline.close()


def create_app(config_path: str | None = None) -> FastAPI:
    """构造 FastAPI 应用。"""
    config = load_config(config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(
            level=getattr(logging, config.logging.level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        state = AppState(config)
        app.state.birdid = state
        LOGGER.info("birdid %s 启动完成，模型目录 %s", __version__, config.npu.model_dir)
        try:
            yield
        finally:
            LOGGER.info("birdid 停止，释放资源")
            state.close()

    app = FastAPI(
        title="birdid-orangepi-20t",
        version=__version__,
        description="昇腾 NPU 加速的鸟类识别服务，兼容懂鸟 v2 协议。",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """健康检查：返回版本与设备信息。"""
        return {"status": "ok", "version": __version__}

    @app.post("/dongniao")
    async def dongniao(request: Request):
        """懂鸟兼容端点。"""
        state: AppState = request.app.state.birdid
        form = await _read_form(request)
        if not _authorized(state, request, form):
            return JSONResponse(status_code=200, content=error(CODE_QUOTA, "invalid api_key"))
        return await _dispatch(state, form)

    return app


async def _read_form(request: Request) -> dict[str, Any]:
    """读取 multipart 或 urlencoded 表单，把文件读成 bytes。"""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        parsed = await request.form()
        result: dict[str, Any] = {}
        for key, value in parsed.multi_items():
            if hasattr(value, "read"):
                result[key] = await value.read()
            else:
                result[key] = value
        return result
    if "application/x-www-form-urlencoded" in content_type:
        parsed = await request.form()
        return dict(parsed)
    return {}


def _authorized(state: AppState, request: Request, form: dict[str, Any]) -> bool:
    """校验 api_key：请求头或同名表单字段二选一。"""
    expected = state.config.auth.api_key
    if not expected:
        return True
    header = request.headers.get(state.config.auth.header_name)
    provided = header or form.get(state.config.auth.form_field)
    if isinstance(provided, bytes):
        provided = provided.decode("utf-8", "ignore")
    return provided == expected


async def _dispatch(state: AppState, form: dict[str, Any]) -> JSONResponse:
    """按字段分派到各业务分支。"""
    if "resultid" in form:
        return JSONResponse(content=_handle_result(state, str(form["resultid"])))
    if "quota" in form:
        return JSONResponse(content=_handle_quota())
    if "animalid" in form:
        return JSONResponse(content=_handle_animal(str(form["animalid"]), str(form.get("class", "B"))))
    if "image" in form or "url" in form:
        return JSONResponse(content=_handle_image(state, form))
    if "sound" in form or "sound_url" in form:
        return JSONResponse(content=_handle_sound(state, form))
    return JSONResponse(content=error(CODE_PROCESSING, "missing parameters"))


def _accept(state: AppState, kind: str, class_code: str, work: Any) -> list[Any]:
    """受理一个后台任务；队列打满时立刻回忙，而不是让请求排队到超时。"""
    result_id = state.executor.submit(kind, class_code, work)
    if result_id is None:
        return error(CODE_QUOTA, MSG_BUSY)
    return processing(result_id)


def _handle_result(state: AppState, result_id: str) -> list[Any]:
    """轮询识别结果。"""
    record = state.executor.get(result_id)
    if record is None:
        return [CODE_PROCESSING, result_id]
    if record.get("state") == "processing":
        return processing(result_id)
    return record.get("payload") or no_animals()


def _handle_quota() -> list[Any]:
    """配额查询：本地部署不限额，返回固定的大额度。"""
    return ok({"image": 100000, "sound": 100000})


def _handle_animal(animal_id: str, class_code: str) -> list[Any]:
    """百科查询：本地暂无物种库，返回空列表。"""
    del class_code
    return ok({"animalid": animal_id, "info": {}})


def _handle_image(state: AppState, form: dict[str, Any]) -> list[Any]:
    """图片识别分支。"""
    class_code = normalize_class(_text(form.get("class")))
    geo = _parse_geo(form)
    if "image" in form:
        raw = form["image"]
        if not isinstance(raw, bytes) or not raw:
            return error(CODE_PROCESSING, "empty image")
        return _accept_image_task(state, raw, class_code, geo)
    url = _text(form.get("url"))
    if not url:
        return error(CODE_PROCESSING, "empty url")

    def work() -> Any:
        try:
            data = _download(url, HTTP_TIMEOUT_IMAGE)
        except Exception:  # noqa: BLE001
            return MSG_URL_FAILED
        return _run_image(state, data, geo)

    return _accept(state, "image", class_code, work)


def _accept_image_task(
    state: AppState, raw: bytes, class_code: str, geo: tuple = (None, None, None)
) -> list[Any]:
    """受理上传图片并异步识别。"""
    if not _valid_image_size(state, raw):
        return error(CODE_PROCESSING, "image too large")
    return _accept(state, "image", class_code, lambda: _run_image(state, raw, geo))


def _handle_sound(state: AppState, form: dict[str, Any]) -> list[Any]:
    """鸟鸣识别分支。"""
    geo = _parse_geo(form)
    if "sound" in form:
        raw = form["sound"]
        if not isinstance(raw, bytes) or not raw:
            return error(CODE_PROCESSING, "empty sound")
        if len(raw) > state.config.server.max_upload_sound_mb * 1024 * 1024:
            return error(CODE_PROCESSING, "sound too large")
        return _accept(state, "sound", "B", lambda: _run_sound(state, raw, geo))
    url = _text(form.get("sound_url"))
    if not url:
        return error(CODE_PROCESSING, MSG_SOUND_URL_ERROR)

    def work() -> Any:
        try:
            data = _download(url, HTTP_TIMEOUT_SOUND)
        except Exception:  # noqa: BLE001
            return MSG_SOUND_URL_ERROR
        return _run_sound(state, data, geo)

    return _accept(state, "sound", "B", work)


def _run_image(state: AppState, raw: bytes, geo: tuple = (None, None, None)) -> Any:
    """解码并执行图片识别，返回懂鸟结果数组或 None。"""
    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    server = state.config.server
    longest = max(width, height)
    shortest = min(width, height)
    if longest > server.max_image_side or shortest < server.min_image_side:
        # 懂鸟规范要求边长 11..8192 像素；越界按无有效动物处理。
        return None
    scale = _draft_decode(image, state.config.npu.decode_max_side)
    image.load()
    result = state.image_pipeline.predict(image, box_scale=scale, geo=geo)
    return result.entries or None


def _draft_decode(image: Image.Image, max_side: int) -> float:
    """JPEG 抽样解码，返回原图与解码图的边长比例。

    PIL 的 draft 只对 JPEG 等少数格式生效，会让解码器直接按 1/2、1/4、1/8
    抽样，省掉大图的整幅解码开销；对 PNG 等格式是空操作，此时比例为 1.0。
    """
    if max_side <= 0 or max(image.size) <= max_side:
        return 1.0
    before = image.size
    image.draft("RGB", (max_side, max_side))
    after = image.size
    if after[0] <= 0 or after[1] <= 0:
        return 1.0
    # draft 按 2 的幂抽样，宽高比例一致；取两者最大值以防边界取整偏差。
    return max(before[0] / after[0], before[1] / after[1])


def _parse_geo(form: dict[str, Any]) -> tuple[float | None, float | None, int | None]:
    """解析可选的地理/季节过滤参数（懂鸟规范之外的扩展字段）。"""
    try:
        lat = float(_text(form.get("latitude"))) if form.get("latitude") is not None else None
        lon = float(_text(form.get("longitude"))) if form.get("longitude") is not None else None
    except (TypeError, ValueError):
        return None, None, None
    if lat is None or lon is None:
        return None, None, None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None, None, None
    week_text = _text(form.get("week"))
    if week_text:
        try:
            week = int(float(week_text))
        except ValueError:
            week = date.today().timetuple().tm_yday * WEEKS_IN_YEAR // 366 + 1
    else:
        week = date.today().timetuple().tm_yday * WEEKS_IN_YEAR // 366 + 1
    return lat, lon, min(WEEKS_IN_YEAR, max(1, week))


def _run_sound(
    state: AppState,
    raw: bytes,
    geo: tuple[float | None, float | None, int | None] = (None, None, None),
) -> Any:
    """执行鸟鸣识别，可选叠加地理/季节过滤。"""
    latitude, longitude, week = geo
    return state.sound_pipeline.predict(raw, latitude, longitude, week)


def _valid_image_size(state: AppState, raw: bytes) -> bool:
    return len(raw) <= state.config.server.max_upload_image_mb * 1024 * 1024


def _text(value: Any) -> str:
    """把表单值统一成去空白的字符串。"""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


class _LimitedRedirect(urllib.request.HTTPRedirectHandler):
    """限制重定向次数，避免恶意跳转。"""

    max_repeats = MAX_REDIRECTS


def _download(url: str, timeout: float) -> bytes:
    """下载远端图片或音频。"""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("仅支持 http/https")
    opener = urllib.request.build_opener(_LimitedRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "birdid/1.0"})
    with opener.open(request, timeout=timeout) as response:
        return response.read()


def main() -> None:
    """命令行入口：uvicorn 启动。"""
    import uvicorn

    config = load_config(os.environ.get("BIRDID_CONFIG"))
    uvicorn.run(
        create_app(os.environ.get("BIRDID_CONFIG")),
        host=config.server.host,
        port=config.server.port,
        log_level=config.logging.level.lower(),
        access_log=config.logging.access_log,
    )


if __name__ == "__main__":
    main()