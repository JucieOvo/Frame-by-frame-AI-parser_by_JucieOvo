"""
模块名称：launcher
功能描述：
    提供项目的正式启动入口。
    该模块负责串联缓存环境初始化、运行目录归档、完整环境检查，以及 Streamlit 应用启动参数组织。

主要组件：
    - main: 启动程序的统一入口。

依赖说明：
    - sys: 组织 Streamlit 启动参数。
    - streamlit.web.cli: 启动 Streamlit 应用。
    - video_ai_suite.bootstrap.env: 启动前环境准备。
    - video_ai_suite.bootstrap.checks: 启动前真实性检查。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始 run.py 中抽离正式启动入口。
"""

from __future__ import annotations

import sys

from video_ai_suite.backend.runtime import early_set_cache_env

# 必须在导入 Streamlit 前先设置缓存目录，避免模型相关依赖沿用用户本机旧环境变量。
early_set_cache_env()

import streamlit.web.cli as stcli

from video_ai_suite.bootstrap.checks import check_and_prepare_environment
from video_ai_suite.bootstrap.env import (
    clear_old_cache,
    force_set_cache_env,
    get_streamlit_entry_path,
    print_environment_summary,
)


def main() -> int:
    """
    启动应用程序。

    :return: Streamlit CLI 返回码。
    """
    print("=" * 60)
    print("清理旧的运行目录")
    print("=" * 60)
    clear_old_cache()

    print("\n初始化缓存环境变量")
    cache_dir = force_set_cache_env()
    print(f"缓存目录已设置: {cache_dir}")

    if not check_and_prepare_environment():
        return 1

    cache_dir = force_set_cache_env()
    print(f"\n最终确认缓存目录: {cache_dir}")
    print_environment_summary()

    file_path = get_streamlit_entry_path()
    sys.argv = [
        "streamlit",
        "run",
        file_path,
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=false",
        "--global.developmentMode=false",
        "--client.toolbarMode=minimal",
        "--server.maxUploadSize=65536",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
