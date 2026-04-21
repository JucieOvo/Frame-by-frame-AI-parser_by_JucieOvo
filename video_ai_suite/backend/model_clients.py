"""
模块名称：model_clients
功能描述：
    统一封装文本模型与视觉模型的调用入口。
    该模块将 OpenAI 兼容端点统一收敛到 LangChain 的 `ChatOpenAI` 客户端，
    同时保留 Ollama 原生接口以处理图片输入与本地模型调用场景。

主要组件：
    - resolve_api_key: 解析端点所需的 API Key。
    - invoke_text_model: 执行文本模型推理。
    - invoke_vision_model: 执行视觉模型推理。

依赖说明：
    - base64: 构造 data URL 图片消息。
    - mimetypes: 推断图片 MIME 类型。
    - langchain_openai: OpenAI 兼容端点的统一聊天客户端。
    - ollama: Ollama 原生聊天客户端。

作者：JucieOvo
创建日期：2026-04-21
修改记录：
    - 2026-04-21 JucieOvo: 创建多端点统一模型调用模块。
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from ollama import Client


def resolve_api_key(
    endpoint: dict[str, Any],
    api_key_overrides: dict[str, str] | None = None,
) -> str | None:
    """
    解析端点使用的 API Key。

    :param endpoint: 端点配置字典。
    :param api_key_overrides: 页面层传入的临时覆盖字典。
    :return: 实际可用的 API Key；无则返回 None。
    """
    endpoint_key = endpoint.get("endpoint_id") or endpoint.get("provider_id") or ""
    if api_key_overrides and endpoint_key in api_key_overrides:
        override_value = str(api_key_overrides[endpoint_key]).strip()
        if override_value:
            return override_value

    env_name = str(endpoint.get("api_key_env_name", "")).strip()
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value

    return None


def _extract_message_text(message_content: Any) -> str:
    """
    将 LangChain 返回消息归一化为字符串。

    :param message_content: 模型返回内容。
    :return: 提取后的纯文本内容。
    """
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_fragments: list[str] = []
        for item in message_content:
            if isinstance(item, str):
                text_fragments.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_fragments.append(str(item["text"]))
                elif item.get("content"):
                    text_fragments.append(str(item["content"]))
        return "\n".join(fragment for fragment in text_fragments if fragment)

    return str(message_content)


def _build_chat_openai_client(
    endpoint: dict[str, Any],
    model_name: str,
    api_key_overrides: dict[str, str] | None = None,
) -> ChatOpenAI:
    """
    构造 OpenAI 兼容端点聊天客户端。

    :param endpoint: 端点配置字典。
    :param model_name: 模型名称。
    :param api_key_overrides: 可选的 API Key 覆盖字典。
    :return: `ChatOpenAI` 客户端实例。
    """
    api_key = resolve_api_key(endpoint, api_key_overrides) or "EMPTY"
    timeout_seconds = float(endpoint.get("timeout_seconds", 60) or 60)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=endpoint.get("base_url", ""),
        timeout=timeout_seconds,
        temperature=0.1,
        max_retries=2,
    )


def _image_to_data_url(image_path: str) -> str:
    """
    将本地图片文件转换为 data URL。

    :param image_path: 图片路径。
    :return: data URL 字符串。
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def invoke_text_model(
    endpoint: dict[str, Any],
    model_name: str,
    prompt: str,
    api_key_overrides: dict[str, str] | None = None,
) -> str:
    """
    执行文本模型推理。

    :param endpoint: 端点配置字典。
    :param model_name: 模型名称。
    :param prompt: 提示词。
    :param api_key_overrides: 可选的 API Key 覆盖字典。
    :return: 模型返回文本。
    """
    provider_type = endpoint.get("provider_type", "openai_compatible")
    if provider_type == "ollama":
        response = Client(host=endpoint.get("base_url", "http://127.0.0.1:11434")).chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_ctx": 4096},
        )
        return response["message"]["content"]

    client = _build_chat_openai_client(endpoint, model_name, api_key_overrides)
    message = client.invoke(prompt)
    return _extract_message_text(message.content)


def invoke_vision_model(
    endpoint: dict[str, Any],
    model_name: str,
    prompt: str,
    image_path: str,
    api_key_overrides: dict[str, str] | None = None,
) -> str:
    """
    执行视觉模型推理。

    :param endpoint: 端点配置字典。
    :param model_name: 模型名称。
    :param prompt: 提示词。
    :param image_path: 图片路径。
    :param api_key_overrides: 可选的 API Key 覆盖字典。
    :return: 模型返回文本。
    """
    provider_type = endpoint.get("provider_type", "openai_compatible")
    if provider_type == "ollama":
        response = Client(host=endpoint.get("base_url", "http://127.0.0.1:11434")).chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
            options={"num_ctx": 4096},
        )
        return response["message"]["content"]

    client = _build_chat_openai_client(endpoint, model_name, api_key_overrides)
    image_url = _image_to_data_url(image_path)
    message = client.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )
        ]
    )
    return _extract_message_text(message.content)
