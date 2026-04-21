"""
模块名称：provider_settings
功能描述：
    从项目根目录 `.env` 读取 VLM 与 LLM 提供商配置。
    该模块用于替代前端动态注册与模型自动发现逻辑，
    让端点来源统一收敛到环境配置文件，由前端仅负责选择提供商并手动填写模型名。

主要组件：
    - load_project_dotenv: 加载项目根目录 `.env`。
    - get_role_providers: 读取指定角色的提供商配置列表。
    - get_provider_by_id: 根据角色与提供商标识读取配置。

依赖说明：
    - os: 读取环境变量与路径处理。
    - dotenv: 从 `.env` 加载环境变量。
    - video_ai_suite.backend.runtime: 获取项目根目录。

作者：JucieOvo
创建日期：2026-04-21
修改记录：
    - 2026-04-21 JucieOvo: 创建 `.env` 提供商配置读取模块。
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from video_ai_suite.backend.runtime import get_program_dir


def load_project_dotenv() -> str:
    """
    加载项目根目录下的 `.env` 文件。

    :return: `.env` 文件绝对路径。
    """
    dotenv_path = os.path.join(get_program_dir(), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path


def _split_provider_ids(raw_value: str) -> list[str]:
    """
    解析角色提供商列表。

    :param raw_value: 环境变量原始值。
    :return: 提供商标识列表。
    """
    separators_normalized = raw_value.replace(";", ",").replace("\n", ",")
    provider_ids = [item.strip() for item in separators_normalized.split(",")]
    return [item for item in provider_ids if item]


def _to_env_suffix(provider_id: str) -> str:
    """
    将提供商标识转换为环境变量后缀。

    :param provider_id: 原始提供商标识。
    :return: 大写下划线格式后缀。
    """
    normalized_chars: list[str] = []
    for char in provider_id:
        if char.isalnum():
            normalized_chars.append(char.upper())
        else:
            normalized_chars.append("_")
    return "".join(normalized_chars)


def _read_provider(role: str, provider_id: str) -> dict[str, Any]:
    """
    读取单个提供商配置。

    :param role: 角色类型，仅支持 `vlm` 或 `llm`。
    :param provider_id: 提供商标识。
    :return: 提供商配置字典。
    """
    role_prefix = role.upper()
    suffix = _to_env_suffix(provider_id)
    env_prefix = f"{role_prefix}_PROVIDER_{suffix}"
    return {
        "provider_id": provider_id,
        "display_name": os.environ.get(f"{env_prefix}_LABEL", provider_id),
        "provider_type": os.environ.get(
            f"{env_prefix}_TYPE", "openai_compatible"
        ).strip(),
        "base_url": os.environ.get(f"{env_prefix}_BASE_URL", "").strip(),
        "api_key_env_name": os.environ.get(
            f"{env_prefix}_API_KEY_ENV_NAME", ""
        ).strip(),
        "enabled": os.environ.get(f"{env_prefix}_ENABLED", "1").strip()
        not in {"0", "false", "False", "FALSE"},
    }


def get_role_providers(role: str) -> list[dict[str, Any]]:
    """
    获取指定角色的全部启用提供商。

    :param role: 角色类型，仅支持 `vlm` 或 `llm`。
    :return: 提供商配置列表。
    """
    load_project_dotenv()
    role_upper = role.upper()
    raw_value = os.environ.get(f"{role_upper}_PROVIDER_IDS", "")
    providers: list[dict[str, Any]] = []
    for provider_id in _split_provider_ids(raw_value):
        provider = _read_provider(role, provider_id)
        if provider["enabled"] and provider["base_url"]:
            providers.append(provider)
    return providers


def get_provider_by_id(role: str, provider_id: str) -> dict[str, Any] | None:
    """
    按角色与标识查找提供商配置。

    :param role: 角色类型，仅支持 `vlm` 或 `llm`。
    :param provider_id: 提供商标识。
    :return: 提供商配置；不存在时返回 None。
    """
    for provider in get_role_providers(role):
        if provider.get("provider_id") == provider_id:
            return provider
    return None
