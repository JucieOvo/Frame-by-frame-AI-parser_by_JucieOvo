"""
模块名称：runtime
功能描述：
    提供项目运行时的公共基础能力。
    该模块统一负责程序目录识别、缓存目录环境变量初始化、图片目录解析与关键帧目录校验。

主要组件：
    - get_program_dir: 获取程序运行根目录。
    - early_set_cache_env: 在导入模型依赖前设置缓存环境变量。
    - list_image_files: 读取目录中的关键帧图片文件。
    - resolve_keyframe_directory: 规范化并校验关键帧目录。

依赖说明：
    - os: 路径处理与环境变量设置。
    - sys: 判断是否为打包环境。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始单文件应用中抽离运行时公共能力。
"""

from __future__ import annotations

import os
import sys

# 关键帧图片支持的扩展名集合。
# 统一集中定义，避免不同读取入口的过滤规则不一致。
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_program_dir() -> str:
    """
    获取程序运行目录。

    :return: 当前程序的实际运行目录。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_program_cache_dir() -> str:
    """
    获取程序缓存目录。

    :return: 位于程序目录下的 `.cache` 路径。
    """
    return os.path.join(get_program_dir(), ".cache")


DEFAULT_KEYFRAME_DIR = os.path.join(get_program_dir(), "keyframes")


def early_set_cache_env() -> str:
    """
    在模型相关库导入前强制设置缓存环境变量。

    该函数用于统一 ModelScope、HuggingFace 与 Torch 的缓存落盘位置，
    避免用户本机已有环境变量污染项目运行结果。

    :return: 程序缓存目录路径。
    """
    cache_dir = get_program_cache_dir()

    os.environ["MODELSCOPE_CACHE_DIR"] = os.path.join(cache_dir, "modelscope", "hub")
    os.environ["MODELSCOPE_CACHE"] = os.path.join(cache_dir, "modelscope", "hub")
    os.environ["MODELSCOPE_HOME"] = os.path.join(cache_dir, "modelscope")
    os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "huggingface", "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(
        cache_dir,
        "huggingface",
        "transformers",
    )
    os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")
    os.environ["MODELSCOPE_SDK_DEBUG"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"

    for directory in [
        os.environ["MODELSCOPE_CACHE_DIR"],
        os.environ["HF_HOME"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["TORCH_HOME"],
    ]:
        os.makedirs(directory, exist_ok=True)

    return cache_dir


def normalize_user_path(path_value: str | None, base_dir: str | None = None) -> str:
    """
    规范化用户输入路径。

    该函数用于兼容相对路径、环境变量、包裹引号与不同分隔符写法，
    统一返回绝对路径，减少界面层对路径清洗的重复代码。

    :param path_value: 用户输入的原始路径。
    :param base_dir: 相对路径解析基准目录。
    :return: 规范化后的绝对路径；若输入为空则返回空字符串。
    """
    if path_value is None:
        return ""

    normalized_path = str(path_value).strip()
    if not normalized_path:
        return ""

    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    start_char = normalized_path[0]
    end_char = normalized_path[-1]
    if quote_pairs.get(start_char) == end_char:
        normalized_path = normalized_path[1:-1].strip()

    if not normalized_path:
        return ""

    normalized_path = os.path.expandvars(os.path.expanduser(normalized_path))
    normalized_path = normalized_path.replace("/", os.sep).replace("\\", os.sep)

    base_path = base_dir or get_program_dir()
    if not os.path.isabs(normalized_path):
        normalized_path = os.path.join(base_path, normalized_path)

    return os.path.normpath(os.path.abspath(normalized_path))


def list_image_files(directory: str | None) -> list[str]:
    """
    安全读取目录中的图片文件。

    该函数显式使用目录扫描而不是 glob 模式匹配，
    以避免特殊字符路径在 Windows 环境下被误判。

    :param directory: 目标目录。
    :return: 图片文件绝对路径列表。
    :raises OSError: 当目录读取失败时抛出异常。
    """
    normalized_dir = normalize_user_path(directory)
    if not normalized_dir:
        return []

    image_files: list[str] = []
    try:
        with os.scandir(normalized_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                    image_files.append(entry.path)
    except OSError as exc:
        raise OSError(f"读取目录失败: {normalized_dir}，原因: {str(exc)}") from exc

    return image_files


def resolve_keyframe_directory(path_value: str | None) -> tuple[str, list[str], str]:
    """
    规范化并校验关键帧目录。

    :param path_value: 用户输入的关键帧目录路径。
    :return: 目录路径、图片列表、错误信息。
    """
    normalized_dir = normalize_user_path(path_value)
    if not normalized_dir:
        return "", [], "关键帧路径为空"

    if not os.path.exists(normalized_dir):
        return normalized_dir, [], "关键帧路径不存在"

    if not os.path.isdir(normalized_dir):
        return normalized_dir, [], "关键帧路径不是文件夹"

    try:
        image_files = list_image_files(normalized_dir)
    except OSError as exc:
        return normalized_dir, [], str(exc)

    if not image_files:
        return normalized_dir, [], "关键帧文件夹中没有图片文件"

    return normalized_dir, image_files, ""
