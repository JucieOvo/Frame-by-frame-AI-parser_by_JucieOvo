"""
模块名称：env
功能描述：
    提供启动前的环境处理能力。
    该模块负责缓存环境变量设置、历史运行目录归档与 Streamlit 入口定位。

主要组件：
    - force_set_cache_env: 强制设置缓存环境变量。
    - clear_old_cache: 归档旧的运行产物目录。
    - get_streamlit_entry_path: 获取正式 Streamlit 入口脚本路径。

依赖说明：
    - os: 路径处理。
    - sys: 打包环境识别。
    - shutil: 文件夹移动。
    - datetime: 生成归档时间戳。
    - video_ai_suite.backend.runtime: 统一缓存环境设置。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始 run.py 中抽离环境初始化能力。
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime

from dotenv import load_dotenv

from video_ai_suite.backend.runtime import early_set_cache_env, get_program_dir


def load_project_dotenv() -> str:
    """
    加载项目根目录 `.env` 文件。

    :return: `.env` 文件绝对路径。
    """
    dotenv_path = os.path.join(get_program_dir(), ".env")
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path


def force_set_cache_env() -> str:
    """
    强制设置缓存目录环境变量。

    :return: 程序缓存目录路径。
    """
    load_project_dotenv()
    return early_set_cache_env()


def clear_old_cache() -> None:
    """
    清除旧的运行产物目录并移动到归档目录。

    当前仅归档运行时生成的 `cache`、`keyframes` 与 `chroma_db` 目录，
    不对其他业务文件做任何改动。
    """
    try:
        program_dir = get_program_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(program_dir, "过往信息", timestamp)
        folders_to_move = [
            ("cache", "cache"),
            ("keyframes", "keyframes"),
            ("chroma_db", "chroma_db"),
        ]

        moved_count = 0
        for folder_name, display_name in folders_to_move:
            folder_path = os.path.join(program_dir, folder_name)
            if not os.path.exists(folder_path):
                continue

            try:
                os.makedirs(archive_dir, exist_ok=True)
                target_path = os.path.join(archive_dir, folder_name)
                shutil.move(folder_path, target_path)
                print(f"已归档 {display_name} 目录")
                moved_count += 1
            except Exception as exc:
                print(f"归档 {display_name} 目录时出错: {str(exc)}")

        if moved_count > 0:
            print(f"共归档 {moved_count} 个目录，归档路径: {archive_dir}")
        else:
            print("没有需要归档的运行目录")
    except Exception as exc:
        print(f"归档旧目录时出错: {str(exc)}")
        print("继续启动程序")


def get_runtime_root_dir() -> str:
    """
    获取当前运行时可用于定位脚本资源的根目录。

    :return: 源码环境下返回项目目录；打包环境下返回解包目录。
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return get_program_dir()


def get_streamlit_entry_path() -> str:
    """
    获取正式 Streamlit 入口路径。

    :return: `video_ai_suite/streamlit_app/main.py` 的绝对路径。
    """
    return os.path.join(
        get_runtime_root_dir(),
        "video_ai_suite",
        "streamlit_app",
        "main.py",
    )


def print_environment_summary() -> None:
    """
    打印关键环境变量摘要。

    该输出用于帮助用户确认缓存目录是否已经正确切换到项目内。
    """
    print("环境变量列表:")
    print(f"  MODELSCOPE_CACHE_DIR = {os.environ.get('MODELSCOPE_CACHE_DIR')}")
    print(f"  HF_HOME = {os.environ.get('HF_HOME')}")
    print(f"  TORCH_HOME = {os.environ.get('TORCH_HOME')}")
