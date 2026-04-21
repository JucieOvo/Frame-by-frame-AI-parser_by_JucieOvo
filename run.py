"""
模块名称：run
功能描述：
    项目根目录下的正式启动入口。
    该文件只保留最薄的一层转发逻辑，实际启动流程已经迁移到 `video_ai_suite.bootstrap.launcher`。

主要组件：
    - main: 延迟调用正式启动器。

依赖说明：
    - video_ai_suite.bootstrap.launcher: 统一启动入口。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 将原始启动逻辑拆分到 bootstrap 目录，仅保留根入口转发。
"""

def main() -> int:
    """
    延迟加载包内正式启动入口。

    :return: 启动器返回码。
    """
    from video_ai_suite.bootstrap.launcher import main as launcher_main

    return launcher_main()


if __name__ == "__main__":
    raise SystemExit(main())
