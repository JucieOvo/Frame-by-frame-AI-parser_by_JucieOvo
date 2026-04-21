"""
模块名称：setup
功能描述：
    根目录兼容安装入口。
    该文件不再承载安装实现，只负责在执行时转发到包内正式安装器。

主要组件：
    - main: 调用包内正式安装入口。

依赖说明：
    - video_ai_suite.bootstrap.installer: 正式安装实现。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 根目录文件收敛为兼容入口。
"""


def main() -> int:
    """
    延迟加载包内正式安装入口。
    """
    from video_ai_suite.bootstrap.installer import main as installer_main

    return installer_main()


if __name__ == "__main__":
    raise SystemExit(main())
