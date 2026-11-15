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
    async def healthz(request: Request) -> dict[str, Any]:
        """健康检查：版本、排队深度与队列上限，便于外部做负载判断。"""
        state: AppState = request.app.state.birdid
        pending = state.executor.pending
        saturating = state.executor.saturated()
        return {
            "status": "busy" if saturating else "ok",
            "version": __version__,
            "pending": pending,
            "max_queue": config.server.max_queue,
        }

    @app.post("/dongniao")
    async def dongniao(request: Request):
        """懂鸟兼容端点。"""
        state: AppState = request.app.state.birdid
        form = await _read_form(request, _upload_limit(state, "image"), _upload_limit(state, "sound"))
        if not _authorized(state, request, form):
            return JSONResponse(status_code=200, content=error(CODE_QUOTA, "invalid api_key"))
        return await _dispatch(state, form)

    return app


async def _read_form(request: Request, image_limit: int, sound_limit: int) -> dict[str, Any]:
    """读取 multipart 或 urlencoded 表单，把文件读成 bytes。

    上传体积在读取阶段就设上限：不设的话 value.read() 会把任意大的请求体
    整个读进内存，一个畸形请求就能把服务打挂。超限时只读 `limit + 1` 字节，
    后续的体积校验会按正常流程回 1001，不会真的分配整块内存。
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        parsed = await request.form()
        result: dict[str, Any] = {}
        for key, value in parsed.multi_items():
            if hasattr(value, "read"):
                limit = sound_limit if key == "sound" else image_limit
                result[key] = await _read_upload(value, limit)
            else:
                result[key] = value
        return result
    if "application/x-www-form-urlencoded" in content_type:
        parsed = await request.form()
        return dict(parsed)
    return {}


async def _read_upload(value: Any, limit: int) -> bytes:
    """分块读取上传文件，超过 limit 立即停止。

    超限时返回已经读到的部分（长度必定大于 limit），交给调用方按既有的
    体积校验回 1001，这样报错文案与「文件太大」保持一致，内存也止步于
    limit + 一个分块。
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await value.read(256 * 1024)
        if not chunk:
            break
        total += len(chunk)
        chunks.append(chunk)
        if total > limit:
            break
    return b"".join(chunks)


def _upload_limit(state: AppState, kind: str) -> int:
    """按业务种类取上传上限。"""
    if kind == "sound":
        return state.config.server.max_upload_sound_mb * 1024 * 1024
    return state.config.server.max_upload_image_mb * 1024 * 1024


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
    declared = _text(form.get("class"))
    class_code = normalize_class(declared)
    # 只在客户端显式传了 class 时才按类别过滤；默认 B 不应该吃掉非鸟结果。
    class_filter = class_code if declared else None
    geo = _parse_geo(form, state.config)
    geo_weight = _geo_weight(state, form)
    if "image" in form:
        raw = form["image"]
        if not isinstance(raw, bytes) or not raw:
            return error(CODE_PROCESSING, "empty image")
        return _accept_image_task(state, raw, class_code, geo, class_filter, geo_weight)
    url = _text(form.get("url"))
    if not url:
        return error(CODE_PROCESSING, "empty url")

    def work() -> Any:
        try:
            data = _download(url, HTTP_TIMEOUT_IMAGE, _image_limit(state))
        except Exception:  # noqa: BLE001
            return MSG_URL_FAILED
        return _run_image(state, data, geo, class_filter, geo_weight)

    return _accept(state, "image", class_code, work)


def _accept_image_task(
    state: AppState,
    raw: bytes,
    class_code: str,
    geo: tuple = (None, None, None),
    class_filter: "str | None" = None,
    geo_weight: float | None = None,
) -> list[Any]:
    """受理上传图片并异步识别。"""
    if not _valid_image_size(state, raw):
        return error(CODE_PROCESSING, "image too large")
    return _accept(
        state,
        "image",
        class_code,
        lambda: _run_image(state, raw, geo, class_filter, geo_weight),
    )


def _handle_sound(state: AppState, form: dict[str, Any]) -> list[Any]:
    """鸟鸣识别分支。"""
    geo = _parse_geo(form, state.config)
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
            data = _download(url, HTTP_TIMEOUT_SOUND, _sound_limit(state))
        except Exception:  # noqa: BLE001
            return MSG_SOUND_URL_ERROR
        return _run_sound(state, data, geo)

    return _accept(state, "sound", "B", work)


def _run_image(
    state: AppState,
    raw: bytes,
    geo: tuple = (None, None, None),
    class_filter: "str | None" = None,
    geo_weight: float | None = None,
) -> Any:
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
    result = state.image_pipeline.predict(
        image, box_scale=scale, geo=geo, geo_weight=geo_weight
    )
    entries = _filter_by_class(result.entries, class_filter)
    return entries or None


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


def _geo_weight(state: AppState, form: dict[str, Any]) -> float | None:
    """只有 area 地区码时用更保守的地理权重。

    显式经纬度是用户现场的真实位置，可以直接用满权重；area 只能映射到省级
    代表坐标，用它做重排会把「同属近缘种」按省份分布强行拉开。实测在
    北京代表点下满权重会把 Eurasian Blue Tit 压成 Azure Tit（蓝山雀根本不在
    华北分布），把权重降到 0.5 就能保持正确答案且仍有纠偏能力。
    """
    npu = state.config.npu
    if _text(form.get("latitude")) or _text(form.get("longitude")):
        return None
    if _text(form.get("area")):
        return npu.area_geo_weight
    return None


def _parse_area(config: Any, value: Any) -> tuple[float | None, float | None]:
    """把懂鸟的 area 地区码换成该地区的代表坐标。"""
    code = _text(value).strip().upper()
    if not code:
        return None, None
    return config.region_coords.get(code, (None, None))


def _parse_geo(
    form: dict[str, Any],
    config: Any = None,
) -> tuple[float | None, float | None, int | None]:
    """解析可选的地理/季节过滤参数。

    经纬度优先；只给懂鸟的 area 地区码时用该地区的代表坐标；都拿不到就不做
    地理先验（返回 None），行为与不传完全一致。
    """
    try:
        lat = float(_text(form.get("latitude"))) if form.get("latitude") is not None else None
        lon = float(_text(form.get("longitude"))) if form.get("longitude") is not None else None
    except (TypeError, ValueError):
        return None, None, None
    if lat is None or lon is None:
        area_lat, area_lon = _parse_area(config, form.get("area")) if config is not None else (None, None)
        if area_lat is None or area_lon is None:
            return None, None, None
        lat, lon = area_lat, area_lon
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


def _filter_by_class(entries: list[Any], class_filter: "str | None") -> list[Any]:
    """按客户端显式声明的 class 过滤结果条目。

    懂鸟规范里 class 是 B/M/A/R/F 的组合，语义是"只关心这几类"。但默认值
    就是 B，而本项目在整图没有鸟时会如实返回检出的其它类别（例如"狗 99%"），
    所以这里只在客户端显式传了 class 字段时才过滤；不传或传空一律不过滤，
    保持"先找鸟、找不到再说明图里是什么"的行为。
    """
    if not entries or class_filter is None:
        return entries
    allowed = set(class_filter)
    kept = []
    for entry in entries:
        items = [item for item in entry.get("list", []) if item[3] in allowed]
        if items:
            kept.append({"box": entry["box"], "list": items})
    return kept


def _image_limit(state: AppState) -> int:
    """图片字节上限，与懂鸟规范一致的 2MB 默认值。"""
    return state.config.server.max_upload_image_mb * 1024 * 1024


def _sound_limit(state: AppState) -> int:
    """鸟鸣字节上限，默认 5MB。"""
    return state.config.server.max_upload_sound_mb * 1024 * 1024


def _valid_image_size(state: AppState, raw: bytes) -> bool:
    return len(raw) <= _image_limit(state)


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


def _download(url: str, timeout: float, max_bytes: int) -> bytes:
    """下载远端图片或音频，超过 max_bytes 立即中止。

    远端可以声称任意 Content-Length 或干脆不声明，所以除了预检头之外
    还要在读取过程中累计字节数，避免恶意 URL 把进程内存打爆。
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("仅支持 http/https")
    opener = urllib.request.build_opener(_LimitedRedirect)
    request = urllib.request.Request(url, headers={"User-Agent": "birdid/1.0"})
    with opener.open(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and int(declared) > max_bytes:
            raise ValueError("下载内容超过上限")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("下载内容超过上限")
            chunks.append(chunk)
        return b"".join(chunks)


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