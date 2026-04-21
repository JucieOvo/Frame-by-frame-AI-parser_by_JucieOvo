"""
模块名称：main
功能描述：
    Streamlit 正式入口脚本。
    该入口负责直接加载包内正式页面模块，不再依赖根目录 `app.py`。

主要组件：
    - main: 加载并执行包内正式页面入口。

依赖说明：
    - pathlib: 计算项目根目录。
    - sys: 调整模块搜索路径。
    - video_ai_suite.backend.runtime: 提前设置缓存环境变量。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 创建正式 Streamlit 入口。
"""

from __future__ import annotations

import os

from video_ai_suite.bootstrap.env import load_project_dotenv
from video_ai_suite.backend.runtime import early_set_cache_env, get_program_dir


def main() -> None:
    """
    执行包内正式页面入口。
    """
    # 统一把工作目录切回项目根，保证历史页面中依赖相对路径的读写行为继续落在项目目录。
    os.chdir(get_program_dir())
    load_project_dotenv()
    early_set_cache_env()

    from video_ai_suite.streamlit_app.legacy_app import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()
