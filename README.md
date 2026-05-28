# AstrBot Image Provider Tool

让大模型可以主动调用 AstrBot WebUI 中已配置的内置 Provider 生成图片。

这个插件不直接保存百炼、OpenAI 或其他平台的 API Key。请先在 AstrBot WebUI 的 Provider 页面配置好可处理文生图请求的 Provider，例如指向 `qwen-image-2.0-pro` 的百炼/Qwen 图像模型 Provider。

## 功能

- 注册 `generate_image` LLM 工具。
- 默认模型名为 `qwen-image-2.0-pro`。
- 支持通过插件配置选择 AstrBot Provider。
- 支持通过工具参数传入模型别名、尺寸和负向提示词，尺寸会被限制在 1080p 范围内。
- 对 `qwen-image*` 和 `wan2.7-image*` 使用百炼同步图像接口，并复用所选 AstrBot Provider 的 API Key。
- 自动根据 Provider 的 `api_base` 选择北京或新加坡 DashScope 图像接口。
- 其他模型继续使用 AstrBot Provider 的结构化 user content 兜底。
- 从 Provider 返回的 URL、Markdown 图片、JSON 图片字段或 base64 图片中提取图片。
- 将图片保存到 `plugin_data/astrbot_plugin_image_provider_tool/generated/` 后发送本地图片。

## 配置项

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `enable` | `true` | 是否启用插件 |
| `image_provider_id` | 空 | 文生图 Provider ID，使用 AstrBot 内置 Provider |
| `default_model_name` | `qwen-image-2.0-pro` | 默认文生图模型名 |
| `model_aliases` | `qwen-image-2.0-pro`, `wan2.7-image-pro` | `generate_image` 可填写的模型名 |
| `max_prompt_chars` | `1000` | 单次提示词最大字符数 |
| `send_provider_text_fallback` | `true` | 未提取到图片时是否返回 Provider 文本 |
| `request_timeout_sec` | `60` | 下载 Provider 返回图片 URL 的超时时间 |

## 使用方式

安装并启用插件后，可用 AstrBot 工具命令确认：

```text
/tool ls
/tool on generate_image
```

用户可以直接说：

```text
画一张赛博朋克风格的猫咖海报，霓虹灯，雨夜，中文标题“夜猫咖啡”
```

支持函数调用的模型会看到 `generate_image` 工具，并可主动调用：

```python
generate_image(
    prompt="赛博朋克风格的猫咖海报，霓虹灯，雨夜，中文标题“夜猫咖啡”",
    model="qwen-image-2.0-pro",
    size="1080*1080",
    negative_prompt="低清晰度，文字错误"
)
```

## 注意事项

- 本插件只复用 AstrBot 内置 Provider 的配置与 API Key，不在插件配置中保存 API Key。
- `qwen-image*` 和 `wan2.7-image*` 会直接请求百炼 `multimodal-generation/generation` 同步接口，避免 AstrBot 文本聊天解析丢失图片字段。
- 对非百炼图像模型，`model` 参数只会写入 Provider 提示词；底层是否真正切换模型取决于所选 AstrBot Provider 自身能力。
- `size` 参数最大限制为 1080p：横图不超过 `1920*1080`，竖图不超过 `1080*1920`，正方形不超过 `1080*1080`。
- 如果 Provider 只返回普通文本，没有返回图片 URL/base64，本插件会返回明确失败说明。
- 只实现文生图，不实现图生图、局部编辑或多图参考。
