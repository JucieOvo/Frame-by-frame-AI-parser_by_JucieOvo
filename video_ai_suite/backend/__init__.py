"""
模块名称：backend
功能描述：
    提供与界面无关的公共后端能力。
    该层负责缓存路径、目录解析、Token 计费等可复用逻辑。

主要组件：
    - runtime: 运行目录、缓存环境变量与路径解析。
    - token_service: Token 统计与费用计算。
    - model_clients: 多端点统一调用入口。
    - provider_settings: `.env` 提供商配置读取能力。
    - job_storage: 批量任务缓存目录与状态持久化。
    - batch_scheduler: 串行与并行批量调度能力。
    - video_pipeline: 批量模式单视频处理流水线。

依赖说明：
    - Python 标准库: 路径处理、系统环境。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 创建后端服务层目录。
"""
