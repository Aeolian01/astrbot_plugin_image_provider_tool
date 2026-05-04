from __future__ import annotations

import asyncio
import base64
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path


def install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    components = types.ModuleType("astrbot.api.message_components")
    star = types.ModuleType("astrbot.api.star")
    core = types.ModuleType("astrbot.core")
    agent = types.ModuleType("astrbot.core.agent")
    agent_message = types.ModuleType("astrbot.core.agent.message")

    class Logger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    class Filter:
        @staticmethod
        def llm_tool(name=None):
            def decorator(func):
                func.__llm_tool_name__ = name
                return func

            return decorator

    class MessageChain(list):
        pass

    class AstrMessageEvent:
        pass

    class Image:
        @classmethod
        def fromFileSystem(cls, path):
            return {"type": "image", "path": path}

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    class TextPart:
        def __init__(self, text):
            self.text = text

    api.logger = Logger()
    event.AstrMessageEvent = AstrMessageEvent
    event.MessageChain = MessageChain
    event.filter = Filter
    components.Image = Image
    star.Context = Context
    star.Star = Star
    agent_message.TextPart = TextPart

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.message_components"] = components
    sys.modules["astrbot.api.star"] = star
    sys.modules["astrbot.core"] = core
    sys.modules["astrbot.core.agent"] = agent
    sys.modules["astrbot.core.agent.message"] = agent_message


install_astrbot_stubs()
main = importlib.import_module("main")


class FakeEvent:
    unified_msg_origin = "umo"

    def __init__(self):
        self.sent = []

    async def send(self, chain):
        self.sent.append(chain)


class FakeResponse:
    def __init__(self, completion_text="", raw_completion=None):
        self.completion_text = completion_text
        self.raw_completion = raw_completion


class FakeProviderMeta:
    id = "fallback-provider"


class FakeProvider:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.calls = []

    def meta(self):
        return FakeProviderMeta()

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class LegacyProvider(FakeProvider):
    async def text_chat(self, prompt=None):
        self.calls.append({"prompt": prompt})
        return self.response


class FakeContext:
    def __init__(self, response=None, provider=None):
        self.response = response or FakeResponse()
        self.provider = provider or FakeProvider(self.response)
        self.llm_generate_calls = []

    async def get_current_chat_provider_id(self, umo=None):
        return ""

    def get_provider_by_id(self, provider_id):
        return self.provider if provider_id == "configured-provider" else None

    def get_all_providers(self):
        return [self.provider]

    async def llm_generate(self, **kwargs):
        self.llm_generate_calls.append(kwargs)
        return self.response


def collect_async(gen):
    async def runner():
        return [item async for item in gen]

    return asyncio.run(runner())


class ImageProviderToolTests(unittest.TestCase):
    def make_plugin(self, config=None, context=None):
        plugin = main.ImageProviderToolPlugin(context or FakeContext(), config or {})
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plugin.generated_dir = Path(tmp.name)
        return plugin

    def test_extracts_markdown_image_url(self):
        plugin = self.make_plugin()
        candidates = plugin._extract_image_candidates(
            "生成完成：![image](https://example.aliyuncs.com/result.png)"
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kind, "url")
        self.assertEqual(candidates[0].value, "https://example.aliyuncs.com/result.png")

    def test_extracts_raw_completion_image_url(self):
        plugin = self.make_plugin()
        raw = {"output": {"images": [{"url": "https://cdn.example.com/download?id=1"}]}}
        candidates = plugin._extract_image_candidates("", raw)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].value, "https://cdn.example.com/download?id=1")

    def test_rejects_overlong_prompt(self):
        plugin = self.make_plugin({"max_prompt_chars": 3})
        result = collect_async(plugin.generate_image(FakeEvent(), "abcd"))
        self.assertEqual(len(result), 1)
        self.assertIn("提示词过长", result[0])

    def test_missing_configured_provider_returns_error(self):
        plugin = self.make_plugin({"image_provider_id": "missing"})
        result = collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(len(result), 1)
        self.assertIn("未找到可用的 AstrBot Provider", result[0])

    def test_provider_text_fallback_when_no_image(self):
        context = FakeContext(FakeResponse("这里只是普通文本"))
        plugin = self.make_plugin({"image_provider_id": "configured-provider"}, context)
        result = collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(len(result), 1)
        self.assertIn("Provider 未返回可提取的图片", result[0])
        self.assertIn("这里只是普通文本", result[0])

    def test_provider_call_uses_structured_user_content(self):
        context = FakeContext(FakeResponse("这里只是普通文本"))
        plugin = self.make_plugin({"image_provider_id": "configured-provider"}, context)
        collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(len(context.provider.calls), 1)
        self.assertEqual(context.llm_generate_calls, [])
        call = context.provider.calls[0]
        self.assertNotIn("system_prompt", call)
        self.assertIn("图片提示词：画一只猫", call["prompt"])
        self.assertEqual(call["contexts"], [])
        self.assertEqual(len(call["extra_user_content_parts"]), 1)
        self.assertEqual(
            call["extra_user_content_parts"][0].text,
            "请直接生成图片，并只返回可下载图片 URL、Markdown 图片、JSON 图片字段或 base64 图片数据。",
        )

    def test_falls_back_to_llm_generate_for_legacy_provider(self):
        provider = LegacyProvider(FakeResponse("这里只是普通文本"))
        context = FakeContext(FakeResponse("这里只是普通文本"), provider)
        plugin = self.make_plugin({"image_provider_id": "configured-provider"}, context)
        collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(provider.calls, [])
        self.assertEqual(len(context.llm_generate_calls), 1)
        self.assertNotIn("system_prompt", context.llm_generate_calls[0])

    def test_base64_image_is_saved_and_sent(self):
        png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n" + b"0" * 128
        ).decode("ascii")
        context = FakeContext(FakeResponse(f"data:image/png;base64,{png}"))
        plugin = self.make_plugin({"image_provider_id": "configured-provider"}, context)
        event = FakeEvent()
        result = collect_async(plugin.generate_image(event, "画一只猫"))
        self.assertEqual(len(event.sent), 1)
        self.assertIn("图片已生成并发送", result[-1])
        self.assertTrue(list(plugin.generated_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
