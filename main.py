from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import inspect
import json
import mimetypes
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
import astrbot.api.message_components as Comp
from astrbot.api.star import Context, Star


IMAGE_URL_RE = re.compile(r"https?://[^\s<>'\")\]]+", re.IGNORECASE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", re.IGNORECASE)
DATA_URL_RE = re.compile(
    r"data:image/(?P<fmt>png|jpeg|jpg|webp);base64,(?P<data>[A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)
IMAGE_KEYS = {
    "image",
    "images",
    "image_url",
    "image_urls",
    "url",
    "urls",
    "content",
    "b64_json",
    "base64",
    "data",
}


@dataclass
class ImageProviderToolConfig:
    enable: bool = True
    image_provider_id: str = ""
    default_model_name: str = "qwen-image-2.0-pro"
    model_aliases: list[str] = None  # type: ignore[assignment]
    max_prompt_chars: int = 1000
    send_provider_text_fallback: bool = True
    request_timeout_sec: int = 60

    def __post_init__(self) -> None:
        if self.model_aliases is None:
            self.model_aliases = ["qwen-image-2.0-pro", "wan2.7-image-pro"]


@dataclass
class ImageCandidate:
    kind: str
    value: str
    fmt: str = "png"


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "启用", "开启"}
    return bool(value)


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(minimum, parsed)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"[\n,，\s]+", value)
    else:
        items = []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _read_config(config: Any) -> ImageProviderToolConfig:
    raw: dict[str, Any] = {}
    if config is not None:
        try:
            raw = dict(config)
        except Exception:
            raw = getattr(config, "data", {}) or {}

    aliases = _as_str_list(raw.get("model_aliases"))
    default_model = str(raw.get("default_model_name", "") or "qwen-image-2.0-pro").strip()
    if default_model and default_model not in aliases:
        aliases.insert(0, default_model)

    return ImageProviderToolConfig(
        enable=_as_bool(raw.get("enable"), True),
        image_provider_id=str(raw.get("image_provider_id", "") or "").strip(),
        default_model_name=default_model or "qwen-image-2.0-pro",
        model_aliases=aliases or ["qwen-image-2.0-pro"],
        max_prompt_chars=_as_int(raw.get("max_prompt_chars"), 1000),
        send_provider_text_fallback=_as_bool(
            raw.get("send_provider_text_fallback"), True
        ),
        request_timeout_sec=_as_int(raw.get("request_timeout_sec"), 60),
    )


class ImageProviderToolPlugin(Star):
    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.cfg = _read_config(config)
        self.plugin_data_dir = self._get_plugin_data_dir()
        self.generated_dir = self.plugin_data_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _get_plugin_data_dir(self) -> Path:
        base_data = Path("/AstrBot/data")
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path  # type: ignore

            base_data = Path(get_astrbot_data_path())
        except Exception:
            try:
                from astrbot.api.star import get_astrbot_data_path  # type: ignore

                base_data = Path(get_astrbot_data_path())
            except Exception:
                pass

        data_dir = base_data / "plugin_data" / "astrbot_plugin_image_provider_tool"
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "[Image Provider Tool] 无法写入 AstrBot 数据目录，"
                f"改用插件本地 plugin_data: {exc}"
            )
            data_dir = Path(__file__).resolve().parent / "plugin_data"
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir

    @staticmethod
    def _clean_text(text: Any) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _provider_id(provider: Any) -> str:
        if provider is None:
            return ""
        meta = getattr(provider, "meta", None)
        if callable(meta):
            try:
                meta_obj = meta()
                value = getattr(meta_obj, "id", "")
                if value:
                    return str(value)
            except Exception:
                pass
        for attr in ("provider_id", "id", "name"):
            value = getattr(provider, attr, None)
            if value:
                return str(value)
        return ""

    async def _get_provider_by_id(self, provider_id: str) -> Any:
        getter = getattr(self.context, "get_provider_by_id", None)
        if not callable(getter):
            return None
        try:
            return await self._maybe_await(getter(provider_id))
        except Exception as exc:
            logger.warning(
                f"[Image Provider Tool] 获取 Provider 失败: {provider_id}: {exc}"
            )
            return None

    async def _resolve_provider_id(self, event: AstrMessageEvent) -> Optional[str]:
        if self.cfg.image_provider_id:
            provider = await self._get_provider_by_id(self.cfg.image_provider_id)
            if provider:
                return self.cfg.image_provider_id
            logger.warning(
                f"[Image Provider Tool] 配置的 Provider 不可用: {self.cfg.image_provider_id}"
            )
            return None

        current_getter = getattr(self.context, "get_current_chat_provider_id", None)
        if callable(current_getter):
            try:
                provider_id = await self._maybe_await(
                    current_getter(umo=event.unified_msg_origin)
                )
                if provider_id:
                    return str(provider_id)
            except Exception as exc:
                logger.warning(f"[Image Provider Tool] 获取当前会话 Provider 失败: {exc}")

        all_getter = getattr(self.context, "get_all_providers", None)
        if callable(all_getter):
            try:
                providers = await self._maybe_await(all_getter())
                candidates = providers.values() if isinstance(providers, dict) else providers or []
                for provider in candidates:
                    provider_id = self._provider_id(provider)
                    if provider_id:
                        return provider_id
            except Exception as exc:
                logger.warning(f"[Image Provider Tool] 获取 Provider 列表失败: {exc}")

        return None

    def _resolve_model(self, model: Any) -> tuple[Optional[str], Optional[str]]:
        clean = self._clean_text(model)
        if not clean:
            return self.cfg.default_model_name, None
        if clean not in self.cfg.model_aliases:
            aliases = ", ".join(self.cfg.model_aliases)
            return None, f"模型 {clean} 不在允许列表中。可选模型：{aliases}"
        return clean, None

    @staticmethod
    def _build_prompt(prompt: str, model: str, size: str, negative_prompt: str) -> str:
        lines = [
            "请根据用户提示生成一张图片，并在结果中返回图片 URL、Markdown 图片链接、JSON 图片字段或 base64 图片数据。",
            f"目标模型/模型别名：{model}",
            f"图片提示词：{prompt}",
        ]
        if size:
            lines.append(f"期望尺寸：{size}")
        if negative_prompt:
            lines.append(f"负向提示词：{negative_prompt}")
        lines.append("请不要只描述图片；如果生成成功，必须返回可用于下载的图片结果。")
        return "\n".join(lines)

    @staticmethod
    def _structured_prompt_tail() -> str:
        return "请直接生成图片，并只返回可下载图片 URL、Markdown 图片、JSON 图片字段或 base64 图片数据。"

    @staticmethod
    def _make_text_part(text: str) -> Optional[Any]:
        try:
            from astrbot.core.agent.message import TextPart  # type: ignore

            return TextPart(text=text)
        except Exception as exc:
            logger.debug(f"[Image Provider Tool] 当前 AstrBot 不支持 TextPart: {exc}")
            return None

    @staticmethod
    def _supports_extra_user_content_parts(func: Any) -> bool:
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return True
        if "extra_user_content_parts" in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    async def _call_provider_text_chat(self, provider: Any, user_prompt: str) -> Any:
        text_chat = getattr(provider, "text_chat", None)
        if not callable(text_chat):
            return None
        if not self._supports_extra_user_content_parts(text_chat):
            return None

        text_part = self._make_text_part(self._structured_prompt_tail())
        if text_part is None:
            return None

        return await text_chat(
            prompt=user_prompt,
            contexts=[],
            extra_user_content_parts=[text_part],
        )

    async def _call_provider(self, provider_id: str, user_prompt: str) -> Any:
        provider = await self._get_provider_by_id(provider_id)
        if provider is not None:
            llm_resp = await self._call_provider_text_chat(provider, user_prompt)
            if llm_resp is not None:
                return llm_resp

        llm_generate = getattr(self.context, "llm_generate", None)
        if not callable(llm_generate):
            raise RuntimeError("当前 AstrBot Context 不支持 llm_generate")
        return await llm_generate(
            chat_provider_id=provider_id,
            prompt=user_prompt,
        )

    def _extract_image_candidates(self, text: Any, raw: Any = None) -> list[ImageCandidate]:
        candidates: list[ImageCandidate] = []

        def add_url(url: str, force: bool = False) -> None:
            clean = self._strip_url(url)
            if clean and (force or self._looks_like_image_url(clean)):
                candidates.append(ImageCandidate("url", clean))

        def add_base64(data: str, fmt: str = "png") -> None:
            clean = re.sub(r"\s+", "", str(data or ""))
            if self._is_probable_base64_image(clean):
                candidates.append(ImageCandidate("base64", clean, fmt or "png"))

        text_value = str(text or "")
        for match in DATA_URL_RE.finditer(text_value):
            add_base64(match.group("data"), match.group("fmt"))
        for match in MARKDOWN_IMAGE_RE.finditer(text_value):
            add_url(match.group(1), force=True)
        for match in IMAGE_URL_RE.finditer(text_value):
            add_url(match.group(0))

        def walk(value: Any, key_hint: str = "") -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, str(key).lower())
                return
            if isinstance(value, (list, tuple)):
                for child in value:
                    walk(child, key_hint)
                return
            if not isinstance(value, str):
                attrs = ("image", "image_url", "url", "b64_json", "base64", "data")
                for attr in attrs:
                    if hasattr(value, attr):
                        try:
                            walk(getattr(value, attr), attr)
                        except Exception:
                            pass
                return

            value_text = value.strip()
            for match in DATA_URL_RE.finditer(value_text):
                add_base64(match.group("data"), match.group("fmt"))
            for match in MARKDOWN_IMAGE_RE.finditer(value_text):
                add_url(match.group(1), force=True)
            for match in IMAGE_URL_RE.finditer(value_text):
                add_url(match.group(0), force=key_hint in IMAGE_KEYS)
            if key_hint in IMAGE_KEYS:
                add_base64(value_text)

        walk(raw)

        unique: list[ImageCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.kind, candidate.value)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    @staticmethod
    def _strip_url(url: str) -> str:
        return str(url or "").strip().rstrip(".,，。;；")

    @staticmethod
    def _looks_like_image_url(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        lower = url.lower()
        image_tokens = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            "image",
            "aliyuncs",
            "dashscope",
            "oss",
            "base64",
        )
        return any(token in lower for token in image_tokens)

    @staticmethod
    def _is_probable_base64_image(data: str) -> bool:
        if len(data) < 80:
            return False
        try:
            head = base64.b64decode(data[:160] + "==", validate=False)
        except (binascii.Error, ValueError):
            return False
        return (
            head.startswith(b"\x89PNG")
            or head.startswith(b"\xff\xd8\xff")
            or head.startswith(b"RIFF")
        )

    async def _save_candidate(self, candidate: ImageCandidate, index: int) -> Path:
        if candidate.kind == "base64":
            data = base64.b64decode(re.sub(r"\s+", "", candidate.value), validate=False)
            return await asyncio.to_thread(self._write_image_bytes, data, candidate.fmt, index)
        return await asyncio.to_thread(self._download_image, candidate.value, index)

    def _write_image_bytes(self, data: bytes, fmt: str, index: int) -> Path:
        ext = self._normalize_ext(fmt)
        filename = self._filename(index, ext, data)
        path = self.generated_dir / filename
        path.write_bytes(data)
        return path

    def _download_image(self, url: str, index: int) -> Path:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AstrBot Image Provider Tool/0.1",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.request_timeout_sec) as resp:
                data = resp.read()
                content_type = resp.headers.get("content-type", "")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"下载图片失败: {exc}") from exc

        if not data:
            raise RuntimeError("下载图片失败：返回内容为空")

        ext = self._ext_from_url_or_type(url, content_type)
        filename = self._filename(index, ext, data)
        path = self.generated_dir / filename
        path.write_bytes(data)
        return path

    @staticmethod
    def _normalize_ext(fmt: str) -> str:
        clean = str(fmt or "png").lower().lstrip(".")
        if clean == "jpeg":
            clean = "jpg"
        if clean not in {"png", "jpg", "webp"}:
            clean = "png"
        return clean

    def _ext_from_url_or_type(self, url: str, content_type: str) -> str:
        path = urllib.parse.urlparse(url).path
        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix in {"png", "jpg", "jpeg", "webp"}:
            return self._normalize_ext(suffix)
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed:
            return self._normalize_ext(guessed)
        return "png"

    @staticmethod
    def _filename(index: int, ext: str, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()[:12]
        return f"{int(time.time())}_{index}_{digest}.{ext}"

    async def _send_image(self, event: AstrMessageEvent, image_path: Path) -> Optional[Any]:
        component = self._make_image_component(str(image_path))
        if component is not None:
            chain = MessageChain([component])
            sender = getattr(event, "send", None)
            if callable(sender):
                await sender(chain)
                return None

            send_message = getattr(self.context, "send_message", None)
            if callable(send_message):
                await send_message(event.unified_msg_origin, chain)
                return None

        image_result = getattr(event, "image_result", None)
        if callable(image_result):
            return image_result(str(image_path))

        raise RuntimeError("当前 AstrBot 版本不支持发送本地图片")

    @staticmethod
    def _make_image_component(image_path: str) -> Optional[Any]:
        image_cls = getattr(Comp, "Image", None)
        if image_cls is None:
            return None
        for method_name in ("fromFileSystem", "fromPath"):
            method = getattr(image_cls, method_name, None)
            if callable(method):
                return method(image_path)
        try:
            return image_cls(file=image_path)
        except TypeError:
            try:
                return image_cls(image_path)
            except TypeError:
                return None

    @filter.llm_tool(name="generate_image")
    async def generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        model: str = "",
        size: str = "",
        negative_prompt: str = "",
    ):
        """使用 AstrBot 内置 Provider 生成图片。

        当用户要求画图、生成图片、制作插画/海报/头像/场景图时调用。
        只用于文生图，不用于图片编辑、图生图或分析图片。

        Args:
            prompt(string): 图片生成提示词，描述主体、风格、构图、文字等要求。
            model(string): 可选模型名/模型别名，例如 qwen-image-2.0-pro。
            size(string): 可选尺寸要求，例如 1024*1024 或 2048*2048。
            negative_prompt(string): 可选负向提示词，说明不希望出现的内容。
        """
        if not self.cfg.enable:
            yield "文生图工具未启用。"
            return

        clean_prompt = self._clean_text(prompt)
        if not clean_prompt:
            yield "文生图失败：prompt 为空。"
            return
        if len(clean_prompt) > self.cfg.max_prompt_chars:
            yield (
                "文生图失败：提示词过长。"
                f"当前上限为 {self.cfg.max_prompt_chars} 字符，请缩短后再调用。"
            )
            return

        model_name, model_error = self._resolve_model(model)
        if model_error:
            yield f"文生图失败：{model_error}"
            return
        assert model_name is not None

        provider_id = await self._resolve_provider_id(event)
        if not provider_id:
            yield "文生图失败：未找到可用的 AstrBot Provider，请在插件配置中选择 image_provider_id。"
            return

        user_prompt = self._build_prompt(
            clean_prompt,
            model_name,
            self._clean_text(size),
            self._clean_text(negative_prompt),
        )

        try:
            llm_resp = await self._call_provider(provider_id, user_prompt)
        except Exception as exc:
            logger.error(f"[Image Provider Tool] Provider 调用失败: {exc}", exc_info=True)
            yield f"文生图失败：Provider 调用失败：{exc}"
            return

        completion_text = str(getattr(llm_resp, "completion_text", "") or "")
        raw_completion = getattr(llm_resp, "raw_completion", None)
        candidates = self._extract_image_candidates(completion_text, raw_completion)
        if not candidates:
            if self.cfg.send_provider_text_fallback and completion_text.strip():
                yield "文生图失败：Provider 未返回可提取的图片。\n\nProvider 返回：\n" + completion_text.strip()
            else:
                yield "文生图失败：Provider 未返回可提取的图片。"
            return

        saved_paths: list[Path] = []
        for index, candidate in enumerate(candidates, start=1):
            try:
                saved_paths.append(await self._save_candidate(candidate, index))
            except Exception as exc:
                logger.error(f"[Image Provider Tool] 保存图片失败: {exc}", exc_info=True)
                yield f"文生图失败：保存图片失败：{exc}"
                return

        yielded_image_result = False
        for image_path in saved_paths:
            image_result = await self._send_image(event, image_path)
            if image_result is not None:
                yielded_image_result = True
                yield image_result

        file_names = ", ".join(path.name for path in saved_paths)
        status = (
            f"图片已生成并发送。模型：{model_name}；Provider：{provider_id}；"
            f"数量：{len(saved_paths)}；文件：{file_names}"
        )
        if yielded_image_result:
            yield status
        else:
            yield status
