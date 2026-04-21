"""
模块名称：app
功能描述：
    根目录兼容 Streamlit 入口。
    该文件不再承载页面实现，只在真正执行时延迟加载包内正式页面模块，避免根级导入阶段被重型依赖直接阻断。

主要组件：
    - main: 调用包内正式页面入口。

依赖说明：
    - video_ai_suite.streamlit_app.legacy_app: 正式页面实现。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 根目录文件收敛为兼容入口。
"""


def main() -> None:
    """
    延迟加载包内正式页面入口。
    """
    from video_ai_suite.streamlit_app.legacy_app import main as legacy_main

    legacy_main()


if __name__ == "__main__":
    main()
