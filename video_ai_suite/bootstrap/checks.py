"""
模块名称：checks
功能描述：
    提供程序启动前的真实环境检查能力。
    该模块负责检查 API Key、FunASR 模型与 Embedding 模型，并在缺失时触发真实下载流程。

主要组件：
    - check_funasr_models: 检查 FunASR 模型完整性。
    - check_embedding_model: 检查 Embedding 模型完整性。
    - check_and_prepare_environment: 执行启动前完整检查。

依赖说明：
    - os: 路径处理。
    - video_ai_suite.bootstrap.downloads: 缺失模型时执行下载。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始 run.py 中抽离启动检查逻辑。
"""

from __future__ import annotations

import os

from video_ai_suite.backend.provider_settings import get_role_providers, load_project_dotenv
from video_ai_suite.backend.runtime import early_set_cache_env, get_program_cache_dir
from video_ai_suite.bootstrap.downloads import download_embedding_model, download_funasr_models


def check_funasr_models() -> tuple[bool, str]:
    """
    检查 FunASR 模型是否完整存在。

    :return: 是否存在、缓存目录路径。
    """
    cache_dir = get_program_cache_dir()
    modelscope_cache = os.path.join(cache_dir, "modelscope", "hub")

    required_models = [
        "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    ]
    required_files = ["configuration.json", "model.pt"]
    base_candidates = [os.path.join(modelscope_cache, "models"), modelscope_cache]

    for model_name in required_models:
        found = False
        for base_dir in base_candidates:
            model_path = os.path.join(base_dir, *model_name.split("/"))
            if os.path.isdir(model_path) and all(
                os.path.exists(os.path.join(model_path, file_name))
                for file_name in required_files
            ):
                found = True
                break
        if not found:
            return False, modelscope_cache

    return True, modelscope_cache


def check_embedding_model() -> tuple[bool, str | None]:
    """
    检查 Qwen3 Embedding 模型是否已经存在。

    :return: 是否存在、模型路径。
    """
    cache_dir = get_program_cache_dir()
    modelscope_cache = os.path.join(cache_dir, "modelscope", "hub")

    qwen_dir = os.path.join(modelscope_cache, "Qwen")
    if not os.path.isdir(qwen_dir):
        return False, None

    try:
        for name in os.listdir(qwen_dir):
            full_path = os.path.join(qwen_dir, name)
            if not os.path.isdir(full_path):
                continue

            normalized = name.replace("_", "").replace("-", "").replace(".", "").lower()
            if "qwen3embedding" not in normalized:
                continue

            if os.path.exists(os.path.join(full_path, "model.safetensors")) and os.path.exists(
                os.path.join(full_path, "config.json")
            ):
                print(f"找到 Qwen3 Embedding 模型: {full_path}")
                return True, full_path
    except Exception:
        return False, None

    return False, None


def check_provider_api_keys() -> None:
    """
    按 `.env` 中实际启用的提供商动态检查所需 API Key。

    缺失时只输出提示，不阻断整个启动流程。
    """
    load_project_dotenv()
    provider_groups: dict[str, list[str]] = {}

    for role in ("vlm", "llm"):
        for provider in get_role_providers(role):
            api_key_env_name = str(provider.get("api_key_env_name", "")).strip()
            if not api_key_env_name:
                continue

            provider_label = str(
                provider.get("display_name") or provider.get("provider_id") or role
            ).strip()
            provider_groups.setdefault(api_key_env_name, []).append(provider_label)

    if not provider_groups:
        print("当前 .env 未声明任何需要 API Key 的模型提供商")
        return

    for env_name, provider_labels in provider_groups.items():
        providers_text = "、".join(provider_labels)
        if os.environ.get(env_name):
            print(f"{env_name} 已配置，关联提供商: {providers_text}")
        else:
            print(f"未检测到 {env_name} 环境变量，关联提供商: {providers_text}")
            print("缺少该配置将影响对应在线模型功能")


def check_and_prepare_environment() -> bool:
    """
    执行程序启动前的完整环境检查。

    :return: 是否满足启动条件。
    """
    print("=" * 60)
    print("视频智能分析处理套件 - 环境检查")
    print("=" * 60)

    print("\n初始化缓存目录")
    cache_dir = early_set_cache_env()
    print(f"模型缓存目录: {cache_dir}")

    print("\n检查在线模型 API Key")
    check_provider_api_keys()

    print("\n检查 FunASR 语音识别模型")
    funasr_exists, _ = check_funasr_models()
    if not funasr_exists:
        print("未检测到 FunASR 模型，开始下载")
        download_success, modelscope_cache = download_funasr_models()
        if not download_success:
            print("FunASR 模型下载失败，语音识别功能可能不可用")
        else:
            verified, _ = check_funasr_models()
            if verified:
                print(f"FunASR 模型验证成功: {modelscope_cache}")
            else:
                print("FunASR 模型下载完成，但验证未通过")
    else:
        print("FunASR 模型已存在")

    print("\n检查 Qwen3 Embedding 模型")
    embedding_exists, _ = check_embedding_model()
    if not embedding_exists:
        print("未检测到 Qwen3 Embedding 模型，开始下载")
        download_success, model_path = download_embedding_model()
        if not download_success:
            print("Qwen3 Embedding 模型下载失败，RAG 功能将回退到现有逻辑")
        else:
            print(f"Qwen3 Embedding 模型下载成功: {model_path}")
    else:
        print("Qwen3 Embedding 模型已存在")

    print("\n环境检查完成，准备启动应用")
    return True
