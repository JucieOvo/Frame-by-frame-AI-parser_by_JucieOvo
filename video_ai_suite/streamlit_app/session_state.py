"""
模块名称：session_state
功能描述：
    统一初始化 Streamlit 会话状态。
    该模块负责把原本散落在单文件应用顶部的大量默认状态收拢到页面层，减少页面入口文件的全局噪音。

主要组件：
    - initialize_session_state: 初始化当前会话的默认状态。

依赖说明：
    - asyncio: 初始化并发队列。
    - streamlit: 写入会话状态。
    - video_ai_suite.backend.runtime: 获取默认目录。
    - video_ai_suite.backend.token_service: 构造默认 Token 统计结构。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始单文件应用中抽离会话状态初始化逻辑。
"""

from __future__ import annotations

import asyncio
import os

import streamlit as st

from video_ai_suite.backend.runtime import (
    DEFAULT_KEYFRAME_DIR,
    get_program_cache_dir,
    get_program_dir,
)
from video_ai_suite.backend.token_service import create_empty_token_usage


def initialize_session_state() -> None:
    """
    初始化 Streamlit 会话状态。

    仅在键不存在时写入默认值，避免覆盖用户在当前会话中的真实操作结果。
    """
    program_dir = get_program_dir()
    cache_root_dir = get_program_cache_dir()
    defaults = {
        "selected_model": None,
        "client": None,
        "video_path": None,
        "keyframe_dir": DEFAULT_KEYFRAME_DIR,
        "output_file": os.path.join(program_dir, "草稿.txt"),
        "processing": False,
        "progress_bar": None,
        "api_key": None,
        "use_ollama": False,
        "ollama_models": [],
        "ollama_used": False,
        "show_stop_button": False,
        "user_custom_prompt": "",
        "funasr_model": None,
        "vector_store": None,
        "cache_dir": os.path.join(cache_root_dir, "single", "vlm_results"),
        "current_page": "视频分析",
        "query_history": [],
        "use_existing_keyframes": False,
        "existing_keyframes_path": "",
        "use_existing_vector_db": False,
        "existing_vector_db_path": "",
        "force_reparse_keyframes": False,
        "vector_store_path": os.path.join(cache_root_dir, "single", "chroma_db"),
        "embedding_model": None,
        "vlm_model": "",
        "llm_model": "",
        "llm_use_ollama": False,
        "llm_ollama_model": None,
        "token_usage": create_empty_token_usage(),
        "api_key_overrides": {},
        "selected_vlm_endpoint_id": "",
        "selected_llm_endpoint_id": "",
        "batch_execution_mode": "serial",
        "batch_max_concurrency": 1,
        "batch_submit_interval_seconds": 0.0,
        "batch_retry_interval_seconds": 5.0,
        "batch_post_job_cooldown_seconds": 0.0,
        "batch_max_retries": 0,
        "active_batch_id": "",
        "active_result_job_id": "",
        "active_result_batch_id": "",
        "batch_last_summary": None,
        "result_jobs_cache": [],
        "batch_status_messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "concurrent_queue" not in st.session_state:
        st.session_state.concurrent_queue = asyncio.Queue()
