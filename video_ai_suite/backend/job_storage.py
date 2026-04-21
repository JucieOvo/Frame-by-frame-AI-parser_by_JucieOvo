"""
模块名称：job_storage
功能描述：
    统一管理批量视频任务的硬盘缓存目录、任务清单与状态持久化文件。
    该模块负责把所有运行时中间产物约束在项目根目录 `.cache` 下，避免任务写出到项目外部目录，
    同时通过 batch_id 与 job_id 双层命名空间实现多视频中间产物隔离。

主要组件：
    - create_batch_record: 创建批次目录与批次清单。
    - create_job_record: 创建单视频任务目录与任务清单。
    - write_uploaded_file: 将 Streamlit 上传文件写入任务私有目录。
    - update_job_state: 更新任务状态文件。
    - refresh_batch_runtime: 根据任务状态回写批次运行状态。
    - list_result_jobs: 枚举当前项目下可供复用的处理结果。

依赖说明：
    - hashlib: 生成上传文件摘要与稳定目录键。
    - json: 读写任务状态文件。
    - os: 路径拼接与目录创建。
    - uuid: 生成批次与任务唯一标识。
    - video_ai_suite.backend.runtime: 获取项目缓存根目录。

作者：JucieOvo
创建日期：2026-04-21
修改记录：
    - 2026-04-21 JucieOvo: 创建批量任务硬盘缓存与状态持久化模块。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from typing import Any

from video_ai_suite.backend.runtime import get_program_cache_dir


# 任务元数据会被批量线程并发更新，这里统一使用可重入锁保护读写与统计回写顺序。
_JOB_STORAGE_LOCK = threading.RLock()


def _now_string() -> str:
    """
    获取当前时间字符串。

    :return: 格式化后的本地时间字符串。
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(file_path: str, data: dict[str, Any]) -> None:
    """
    原子化写入 JSON 文件。

    :param file_path: 目标文件路径。
    :param data: 待写入的数据字典。
    """
    with _JOB_STORAGE_LOCK:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        temp_path = f"{file_path}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, file_path)


def _read_json(file_path: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    安全读取 JSON 文件。

    :param file_path: JSON 文件路径。
    :param default: 文件不存在时返回的默认值。
    :return: 读取到的数据字典。
    """
    with _JOB_STORAGE_LOCK:
        if not os.path.exists(file_path):
            return dict(default or {})

        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)


def _merge_job_paths(job_manifest: dict[str, Any], job_state: dict[str, Any]) -> dict[str, str]:
    """
    合并任务清单路径与运行时产物路径。

    运行完成后，`artifacts` 中的路径优先级高于任务清单中的默认路径，
    以保证重用外部关键帧或外部向量数据库时能够正确回绑真实结果目录。

    :param job_manifest: 任务清单字典。
    :param job_state: 任务状态字典。
    :return: 合并后的有效路径字典。
    """
    merged_paths = dict(job_manifest.get("paths", {}))
    artifacts = job_state.get("artifacts", {})
    for key, value in artifacts.items():
        if isinstance(value, str) and value:
            merged_paths[key] = value
    return merged_paths


def _build_timestamp_id(prefix: str) -> str:
    """
    生成带时间戳的唯一标识。

    :param prefix: 标识前缀。
    :return: 带时间戳与短随机串的唯一标识。
    """
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _ensure_within_cache(target_path: str) -> str:
    """
    校验目标路径是否位于项目 `.cache` 内。

    :param target_path: 目标绝对路径。
    :return: 校验通过的绝对路径。
    :raises ValueError: 当路径试图写出到 `.cache` 之外时抛出异常。
    """
    cache_root = os.path.normcase(os.path.abspath(get_program_cache_dir()))
    normalized_target = os.path.normcase(os.path.abspath(target_path))
    common_prefix = os.path.commonpath([cache_root, normalized_target])
    if common_prefix != cache_root:
        raise ValueError(f"目标路径超出项目缓存目录范围: {target_path}")
    return os.path.abspath(target_path)


def get_batches_root_dir() -> str:
    """
    获取批量任务根目录。

    :return: `.cache/batches` 绝对路径。
    """
    root_dir = os.path.join(get_program_cache_dir(), "batches")
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def get_batch_dir(batch_id: str) -> str:
    """
    获取指定批次目录。

    :param batch_id: 批次标识。
    :return: 批次目录绝对路径。
    """
    return _ensure_within_cache(os.path.join(get_batches_root_dir(), batch_id))


def get_job_dir(batch_id: str, job_id: str) -> str:
    """
    获取指定任务目录。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :return: 任务目录绝对路径。
    """
    return _ensure_within_cache(
        os.path.join(get_batch_dir(batch_id), "jobs", job_id)
    )


def get_batch_manifest_path(batch_id: str) -> str:
    """
    获取批次清单文件路径。

    :param batch_id: 批次标识。
    :return: 批次清单文件路径。
    """
    return os.path.join(get_batch_dir(batch_id), "batch_manifest.json")


def get_batch_runtime_path(batch_id: str) -> str:
    """
    获取批次运行时状态文件路径。

    :param batch_id: 批次标识。
    :return: 批次运行时文件路径。
    """
    return os.path.join(get_batch_dir(batch_id), "batch_runtime.json")


def get_job_manifest_path(batch_id: str, job_id: str) -> str:
    """
    获取任务清单文件路径。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :return: 任务清单文件路径。
    """
    return os.path.join(get_job_dir(batch_id, job_id), "job_manifest.json")


def get_job_state_path(batch_id: str, job_id: str) -> str:
    """
    获取任务状态文件路径。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :return: 任务状态文件路径。
    """
    return os.path.join(get_job_dir(batch_id, job_id), "job_state.json")


def get_job_paths(batch_id: str, job_id: str, original_name: str = "") -> dict[str, str]:
    """
    生成单任务的全部核心目录路径。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :param original_name: 原始文件名，用于补充源视频扩展名。
    :return: 任务路径字典。
    """
    job_dir = get_job_dir(batch_id, job_id)
    source_dir = os.path.join(job_dir, "source")
    work_dir = os.path.join(job_dir, "work")
    output_dir = os.path.join(job_dir, "output")
    logs_dir = os.path.join(job_dir, "logs")

    extension = os.path.splitext(original_name)[1] if original_name else ".mp4"
    if not extension:
        extension = ".mp4"

    paths = {
        "job_dir": job_dir,
        "source_dir": source_dir,
        "source_video_path": os.path.join(source_dir, f"original_video{extension}"),
        "work_dir": work_dir,
        "audio_dir": os.path.join(work_dir, "audio"),
        "audio_path": os.path.join(work_dir, "audio", "audio.wav"),
        "keyframe_dir": os.path.join(work_dir, "keyframes"),
        "cache_dir": os.path.join(work_dir, "vlm_results"),
        "llm_dir": os.path.join(work_dir, "llm"),
        "temp_dir": os.path.join(work_dir, "temp"),
        "output_dir": output_dir,
        "output_file": os.path.join(output_dir, "report.md"),
        "summary_file": os.path.join(output_dir, "summary.md"),
        "metadata_file": os.path.join(output_dir, "metadata.json"),
        "vector_store_path": os.path.join(output_dir, "chroma_db"),
        "logs_dir": logs_dir,
        "job_log": os.path.join(logs_dir, "job.log"),
    }

    for path_value in paths.values():
        if os.path.splitext(path_value)[1]:
            os.makedirs(os.path.dirname(path_value), exist_ok=True)
        else:
            os.makedirs(path_value, exist_ok=True)

    return paths


def create_batch_record(
    execution_mode: str,
    max_concurrency: int,
    submit_interval_seconds: float,
    retry_interval_seconds: float,
    post_job_cooldown_seconds: float,
    max_retries: int,
    source_type: str,
) -> dict[str, Any]:
    """
    创建批次目录与批次清单。

    :param execution_mode: 执行模式。
    :param max_concurrency: 最大并发数。
    :param submit_interval_seconds: 启动等待时间。
    :param retry_interval_seconds: 重试等待时间。
    :param post_job_cooldown_seconds: 单任务完成后的冷却时间。
    :param max_retries: 最大重试次数。
    :param source_type: 批次来源描述。
    :return: 新建的批次清单字典。
    """
    batch_id = _build_timestamp_id("batch")
    batch_dir = get_batch_dir(batch_id)
    os.makedirs(os.path.join(batch_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(batch_dir, "logs"), exist_ok=True)

    manifest = {
        "batch_id": batch_id,
        "created_at": _now_string(),
        "execution_mode": execution_mode,
        "max_concurrency": max_concurrency,
        "submit_interval_seconds": submit_interval_seconds,
        "retry_interval_seconds": retry_interval_seconds,
        "post_job_cooldown_seconds": post_job_cooldown_seconds,
        "max_retries": max_retries,
        "source_type": source_type,
        "job_ids": [],
    }
    runtime_state = {
        "queued_count": 0,
        "running_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "cancelled_count": 0,
        "last_updated_at": _now_string(),
    }

    _write_json(get_batch_manifest_path(batch_id), manifest)
    _write_json(get_batch_runtime_path(batch_id), runtime_state)
    return manifest


def create_job_record(batch_id: str, original_name: str) -> dict[str, Any]:
    """
    创建单任务目录与任务清单。

    :param batch_id: 所属批次标识。
    :param original_name: 原始文件名。
    :return: 新建的任务清单字典。
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_paths = get_job_paths(batch_id, job_id, original_name)
    manifest = {
        "job_id": job_id,
        "batch_id": batch_id,
        "original_name": original_name,
        "source_path": job_paths["source_video_path"],
        "created_at": _now_string(),
        "provider_endpoint_id": "",
        "vision_model": "",
        "text_model": "",
        "paths": job_paths,
    }
    state = {
        "status": "queued",
        "stage": "uploaded",
        "retry_count": 0,
        "started_at": "",
        "finished_at": "",
        "error_message": "",
        "artifacts": {
            "source_video_path": job_paths["source_video_path"],
            "keyframe_dir": job_paths["keyframe_dir"],
            "cache_dir": job_paths["cache_dir"],
            "output_file": job_paths["output_file"],
            "vector_store_path": job_paths["vector_store_path"],
        },
    }

    with _JOB_STORAGE_LOCK:
        _write_json(get_job_manifest_path(batch_id, job_id), manifest)
        _write_json(get_job_state_path(batch_id, job_id), state)

        batch_manifest = load_batch_manifest(batch_id)
        batch_manifest.setdefault("job_ids", []).append(job_id)
        _write_json(get_batch_manifest_path(batch_id), batch_manifest)
        refresh_batch_runtime(batch_id)
        return manifest


def write_uploaded_file(batch_id: str, job_id: str, uploaded_file: Any) -> str:
    """
    将 Streamlit 上传文件写入任务私有目录。

    :param batch_id: 所属批次标识。
    :param job_id: 任务标识。
    :param uploaded_file: Streamlit UploadedFile 对象。
    :return: 已落盘的源视频路径。
    """
    job_manifest = load_job_manifest(batch_id, job_id)
    source_path = job_manifest["paths"]["source_video_path"]

    os.makedirs(os.path.dirname(source_path), exist_ok=True)
    uploaded_file.seek(0)
    with open(source_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    update_job_state(
        batch_id,
        job_id,
        stage="uploaded",
        artifacts={"source_video_path": source_path},
    )
    return source_path


def copy_video_file_to_job(batch_id: str, job_id: str, source_file_path: str) -> str:
    """
    将已有视频文件复制到任务私有目录。

    :param batch_id: 所属批次标识。
    :param job_id: 任务标识。
    :param source_file_path: 原始视频路径。
    :return: 复制后的任务私有视频路径。
    """
    job_manifest = load_job_manifest(batch_id, job_id)
    target_path = job_manifest["paths"]["source_video_path"]
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_file_path, target_path)
    update_job_state(
        batch_id,
        job_id,
        stage="uploaded",
        artifacts={"source_video_path": target_path},
    )
    return target_path


def load_batch_manifest(batch_id: str) -> dict[str, Any]:
    """
    读取批次清单。

    :param batch_id: 批次标识。
    :return: 批次清单字典。
    """
    return _read_json(get_batch_manifest_path(batch_id), default={})


def load_batch_runtime(batch_id: str) -> dict[str, Any]:
    """
    读取批次运行时状态。

    :param batch_id: 批次标识。
    :return: 批次运行时状态字典。
    """
    return _read_json(get_batch_runtime_path(batch_id), default={})


def load_job_manifest(batch_id: str, job_id: str) -> dict[str, Any]:
    """
    读取任务清单。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :return: 任务清单字典。
    """
    return _read_json(get_job_manifest_path(batch_id, job_id), default={})


def load_job_state(batch_id: str, job_id: str) -> dict[str, Any]:
    """
    读取任务状态。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :return: 任务状态字典。
    """
    return _read_json(get_job_state_path(batch_id, job_id), default={})


def update_job_manifest(batch_id: str, job_id: str, **updates: Any) -> dict[str, Any]:
    """
    更新任务清单。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :param updates: 待写入字段。
    :return: 更新后的任务清单。
    """
    with _JOB_STORAGE_LOCK:
        manifest = load_job_manifest(batch_id, job_id)
        manifest.update(updates)
        _write_json(get_job_manifest_path(batch_id, job_id), manifest)
        return manifest


def update_job_state(batch_id: str, job_id: str, **updates: Any) -> dict[str, Any]:
    """
    更新任务状态文件。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :param updates: 待写入字段。
    :return: 更新后的任务状态。
    """
    with _JOB_STORAGE_LOCK:
        state = load_job_state(batch_id, job_id)
        artifacts_updates = updates.pop("artifacts", None)
        if artifacts_updates:
            current_artifacts = state.get("artifacts", {})
            current_artifacts.update(artifacts_updates)
            state["artifacts"] = current_artifacts

        state.update(updates)
        state["last_updated_at"] = _now_string()
        _write_json(get_job_state_path(batch_id, job_id), state)
        refresh_batch_runtime(batch_id)
        return state


def refresh_batch_runtime(batch_id: str) -> dict[str, Any]:
    """
    根据当前任务状态回写批次运行统计。

    :param batch_id: 批次标识。
    :return: 更新后的批次运行统计字典。
    """
    with _JOB_STORAGE_LOCK:
        manifest = load_batch_manifest(batch_id)
        job_ids = manifest.get("job_ids", [])
        runtime_state = {
            "queued_count": 0,
            "running_count": 0,
            "success_count": 0,
            "failed_count": 0,
            "cancelled_count": 0,
            "last_updated_at": _now_string(),
        }

        for job_id in job_ids:
            status = load_job_state(batch_id, job_id).get("status", "queued")
            counter_key = f"{status}_count"
            if counter_key in runtime_state:
                runtime_state[counter_key] += 1
            else:
                runtime_state["queued_count"] += 1

        _write_json(get_batch_runtime_path(batch_id), runtime_state)
        return runtime_state


def list_batch_jobs(batch_id: str) -> list[dict[str, Any]]:
    """
    列出指定批次下的全部任务摘要。

    :param batch_id: 批次标识。
    :return: 任务摘要列表。
    """
    manifest = load_batch_manifest(batch_id)
    jobs: list[dict[str, Any]] = []
    for job_id in manifest.get("job_ids", []):
        job_manifest = load_job_manifest(batch_id, job_id)
        job_state = load_job_state(batch_id, job_id)
        jobs.append({**job_manifest, "state": job_state})
    return jobs


def list_result_jobs() -> list[dict[str, Any]]:
    """
    枚举项目 `.cache` 中全部可复用的任务结果。

    :return: 任务结果摘要列表。
    """
    results: list[dict[str, Any]] = []
    batches_root = get_batches_root_dir()
    for batch_id in sorted(os.listdir(batches_root), reverse=True):
        batch_dir = os.path.join(batches_root, batch_id)
        if not os.path.isdir(batch_dir):
            continue

        batch_manifest = _read_json(os.path.join(batch_dir, "batch_manifest.json"), default={})
        for job_id in batch_manifest.get("job_ids", []):
            job_manifest = load_job_manifest(batch_id, job_id)
            job_state = load_job_state(batch_id, job_id)
            paths = _merge_job_paths(job_manifest, job_state)
            results.append(
                {
                    "batch_id": batch_id,
                    "job_id": job_id,
                    "original_name": job_manifest.get("original_name", job_id),
                    "status": job_state.get("status", "queued"),
                    "stage": job_state.get("stage", "uploaded"),
                    "finished_at": job_state.get("finished_at", ""),
                    "cache_dir": paths.get("cache_dir", ""),
                    "keyframe_dir": paths.get("keyframe_dir", ""),
                    "output_file": paths.get("output_file", ""),
                    "vector_store_path": paths.get("vector_store_path", ""),
                }
            )

    return results


def compute_uploaded_file_digest(uploaded_file: Any) -> str:
    """
    计算上传文件的稳定摘要值。

    :param uploaded_file: Streamlit UploadedFile 对象。
    :return: 十六进制摘要字符串。
    """
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getbuffer()).hexdigest()
    uploaded_file.seek(0)
    return digest
