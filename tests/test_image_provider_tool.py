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

    api.logger = Logger()
    event.AstrMessageEvent = AstrMessageEvent
    event.MessageChain = MessageChain
    event.filter = Filter
    components.Image = Image
    star.Context = Context
    star.Star = Star

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.message_components"] = components
    sys.modules["astrbot.api.star"] = star


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
    def __init__(self, response=None, provider_config=None, current_key=None):
        self.response = response or FakeResponse()
        self.calls = []
        self.provider_config = provider_config or {
            "key": ["sk-test"],
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "timeout": 120,
        }
        self.current_key = current_key
        self.timeout = self.provider_config.get("timeout", 120)

    def meta(self):
        return FakeProviderMeta()

    def get_current_key(self):
        if self.current_key is not None:
            return self.current_key
        keys = self.provider_config.get("key", [""])
        return keys[0] if keys else ""

    def get_keys(self):
        return self.provider_config.get("key", [""])

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


class CapturingImageProviderToolPlugin(main.ImageProviderToolPlugin):
    def __init__(self, context, config=None):
        super().__init__(context, config)
        self.bailian_posts = []
        self.bailian_response = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "image": "https://dashscope-result.aliyuncs.com/result.png"
                                }
                            ]
                        }
                    }
                ]
            },
            "request_id": "rid-ok",
        }

    def _post_bailian_generation(self, endpoint, api_key, payload, timeout, headers):
        self.bailian_posts.append(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": payload,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return self.bailian_response


class ImageProviderToolTests(unittest.TestCase):
    def make_plugin(self, config=None, context=None, plugin_cls=main.ImageProviderToolPlugin):
        plugin = plugin_cls(context or FakeContext(), config or {})
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plugin.generated_dir = Path(tmp.name)
        return plugin

    @staticmethod
    def non_bailian_config(extra=None):
        config = {
            "image_provider_id": "configured-provider",
            "default_model_name": "local-image-model",
            "model_aliases": ["local-image-model"],
        }
        if extra:
            config.update(extra)
        return config

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
        plugin = self.make_plugin(self.non_bailian_config(), context)
        result = collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(len(result), 1)
        self.assertIn("Provider 未返回可提取的图片", result[0])
        self.assertIn("这里只是普通文本", result[0])

    def test_provider_call_uses_structured_user_content(self):
        context = FakeContext(FakeResponse("这里只是普通文本"))
        plugin = self.make_plugin(self.non_bailian_config(), context)
        collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(len(context.provider.calls), 1)
        self.assertEqual(context.llm_generate_calls, [])
        call = context.provider.calls[0]
        self.assertNotIn("system_prompt", call)
        self.assertIsNone(call["prompt"])
        self.assertEqual(len(call["contexts"]), 1)
        message = call["contexts"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual(len(message["content"]), 1)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertIn("图片提示词：画一只猫", message["content"][0]["text"])

    def test_falls_back_to_llm_generate_for_legacy_provider(self):
        provider = LegacyProvider(FakeResponse("这里只是普通文本"))
        context = FakeContext(FakeResponse("这里只是普通文本"), provider)
        plugin = self.make_plugin(self.non_bailian_config(), context)
        collect_async(plugin.generate_image(FakeEvent(), "画一只猫"))
        self.assertEqual(provider.calls, [])
        self.assertEqual(len(context.llm_generate_calls), 1)
        self.assertNotIn("system_prompt", context.llm_generate_calls[0])

    def test_base64_image_is_saved_and_sent(self):
        png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n" + b"0" * 128
        ).decode("ascii")
        context = FakeContext(FakeResponse(f"data:image/png;base64,{png}"))
        plugin = self.make_plugin(self.non_bailian_config(), context)
        event = FakeEvent()
        result = collect_async(plugin.generate_image(event, "画一只猫"))
        self.assertEqual(len(event.sent), 1)
        self.assertIn("图片已生成并发送", result[-1])
        self.assertTrue(list(plugin.generated_dir.iterdir()))

    def test_size_is_limited_to_1080p(self):
        plugin = self.make_plugin()
        cases = {
            "2048*2048": "1080*1080",
            "2560x1440": "1920*1080",
            "1440p": "1920*1080",
            "1024*2048": "960*1920",
            "1024*1024": "1024*1024",
        }
        for requested, expected in cases.items():
            with self.subTest(requested=requested):
                self.assertEqual(plugin._limit_size_to_1080p(requested), expected)

    def test_call_provider_caps_size_before_bailian_payload(self):
        provider = FakeProvider()
        context = FakeContext(provider=provider)
        plugin = self.make_plugin(
            {"image_provider_id": "configured-provider"},
            context,
            CapturingImageProviderToolPlugin,
        )
        asyncio.run(
            plugin._call_provider(
                "configured-provider",
                "a square poster",
                "qwen-image-2.0-pro",
                "2048*2048",
                "",
            )
        )
        payload = plugin.bailian_posts[0]["payload"]
        self.assertEqual(payload["parameters"], {"size": "1080*1080"})

    def test_non_bailian_provider_prompt_uses_limited_size(self):
        context = FakeContext(FakeResponse("plain text"))
        plugin = self.make_plugin(self.non_bailian_config(), context)
        collect_async(
            plugin.generate_image(FakeEvent(), "draw a poster", size="2048*2048")
        )
        text = context.provider.calls[0]["contexts"][0]["content"][0]["text"]
        self.assertIn("1080*1080", text)
        self.assertNotIn("2048*2048", text)

    def test_bailian_payload_uses_single_text_item_and_parameters(self):
        provider = FakeProvider()
        context = FakeContext(provider=provider)
        plugin = self.make_plugin(
            {"image_provider_id": "configured-provider"},
            context,
            CapturingImageProviderToolPlugin,
        )
        response = asyncio.run(
            plugin._call_provider(
                "configured-provider",
                "a cute kitten",
                "qwen-image-2.0-pro",
                "1024*1024",
                "bad hands",
            )
        )
        self.assertIsInstance(response, main.ProviderCallResult)
        self.assertEqual(len(plugin.bailian_posts), 1)
        post = plugin.bailian_posts[0]
        self.assertEqual(
            post["endpoint"],
            main.BAILIAN_BEIJING_GENERATION_ENDPOINT,
        )
        self.assertEqual(post["api_key"], "sk-test")
        self.assertEqual(post["timeout"], 120)
        payload = post["payload"]
        self.assertEqual(payload["model"], "qwen-image-2.0-pro")
        messages = payload["input"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(len(messages[0]["content"]), 1)
        self.assertEqual(messages[0]["content"][0], {"text": "a cute kitten"})
        self.assertEqual(
            payload["parameters"],
            {"size": "1024*1024", "negative_prompt": "bad hands"},
        )

    def test_bailian_endpoint_uses_singapore_api_base(self):
        provider = FakeProvider(
            provider_config={
                "key": ["sk-sg"],
                "api_base": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            }
        )
        context = FakeContext(provider=provider)
        plugin = self.make_plugin(
            {"image_provider_id": "configured-provider"},
            context,
            CapturingImageProviderToolPlugin,
        )
        asyncio.run(
            plugin._call_provider(
                "configured-provider",
                "a mountain",
                "wan2.7-image-pro",
                "",
                "",
            )
        )
        self.assertEqual(
            plugin.bailian_posts[0]["endpoint"],
            main.BAILIAN_SINGAPORE_GENERATION_ENDPOINT,
        )

    def test_extracts_bailian_choice_image_url(self):
        plugin = self.make_plugin()
        raw = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "image": "https://dashscope-result.aliyuncs.com/result.png"
                                }
                            ]
                        }
                    }
                ]
            }
        }
        candidates = plugin._extract_image_candidates("", raw)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].value,
            "https://dashscope-result.aliyuncs.com/result.png",
        )

    def test_bailian_api_error_message_is_clear(self):
        context = FakeContext(provider=FakeProvider())
        plugin = self.make_plugin(
            {"image_provider_id": "configured-provider"},
            context,
            CapturingImageProviderToolPlugin,
        )
        plugin.bailian_response = {
            "request_id": "rid-error",
            "code": "InvalidParameter",
            "message": "bad request",
        }
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(
                plugin._call_provider(
                    "configured-provider",
                    "a kitten",
                    "qwen-image-2.0-pro",
                    "",
                    "",
                )
            )
        message = str(caught.exception)
        self.assertIn("InvalidParameter", message)
        self.assertIn("bad request", message)
        self.assertIn("rid-error", message)
        self.assertNotIn("sk-test", message)

    def test_missing_bailian_key_returns_clear_error(self):
        provider = FakeProvider(provider_config={"key": [], "api_base": ""}, current_key="")
        context = FakeContext(provider=provider)
        plugin = self.make_plugin({"image_provider_id": "configured-provider"}, context)
        result = collect_async(plugin.generate_image(FakeEvent(), "a kitten"))
        self.assertEqual(len(result), 1)
        self.assertIn("Provider 未配置可用 API Key", result[0])
        self.assertNotIn("sk-", result[0])


if __name__ == "__main__":
    unittest.main()
