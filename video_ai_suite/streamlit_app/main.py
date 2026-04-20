"""
模块名称：main
功能描述：
    Streamlit 正式入口脚本。
    该入口不再直接承担业务逻辑，而是负责加载兼容页面入口，从而把项目的正式启动路径迁移到独立目录。

主要组件：
    - main: 加载并执行兼容页面入口。

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

import sys
from pathlib import Path

from video_ai_suite.backend.runtime import early_set_cache_env


def _ensure_project_root_in_sys_path() -> None:
    """
    将项目根目录加入模块搜索路径。

    该处理用于让新的 Streamlit 入口可以继续加载当前的兼容页面模块，
    在不一次性大搬迁页面代码的前提下，先完成入口结构的正式拆分。
    """
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def main() -> None:
    """
    执行兼容页面入口。
    """
    early_set_cache_env()
    _ensure_project_root_in_sys_path()

    import app as legacy_streamlit_app

    legacy_streamlit_app.main()


if __name__ == "__main__":
    main()
