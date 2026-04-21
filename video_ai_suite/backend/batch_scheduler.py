"""
模块名称：batch_scheduler
功能描述：
    提供批量视频任务的串行与可控并行调度能力。
    该模块负责按照用户设定的执行模式、并发数、等待时间与重试策略驱动任务执行，
    并在执行过程中回写任务状态与批次统计。

主要组件：
    - run_batch_jobs: 执行批次任务。

依赖说明：
    - concurrent.futures: 提供线程池并发调度。
    - time: 控制任务启动间隔、重试等待与冷却时间。
    - video_ai_suite.backend.job_storage: 回写任务与批次状态。

作者：JucieOvo
创建日期：2026-04-21
修改记录：
    - 2026-04-21 JucieOvo: 创建批量任务调度模块。
"""

from __future__ import annotations

import concurrent.futures
import time
import traceback
from typing import Any, Callable

from video_ai_suite.backend.job_storage import refresh_batch_runtime, update_job_state


JobRunner = Callable[[str, str], dict[str, Any]]
StatusCallback = Callable[[str], None]


def _run_job_with_retries(
    batch_id: str,
    job_id: str,
    job_runner: JobRunner,
    max_retries: int,
    retry_interval_seconds: float,
    post_job_cooldown_seconds: float,
    status_callback: StatusCallback | None = None,
) -> dict[str, Any]:
    """
    执行单任务并处理失败重试。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :param job_runner: 实际任务执行函数。
    :param max_retries: 最大重试次数。
    :param retry_interval_seconds: 重试间隔。
    :param post_job_cooldown_seconds: 任务完成后的冷却时间。
    :param status_callback: 可选的状态回调。
    :return: 任务执行结果字典。
    """
    last_error_message = ""
    for attempt_index in range(max_retries + 1):
        update_job_state(
            batch_id,
            job_id,
            status="running",
            stage="preparing",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error_message="",
            retry_count=attempt_index,
        )
        if status_callback:
            status_callback(f"任务 {job_id} 开始执行，第 {attempt_index + 1} 次尝试")

        try:
            result = job_runner(batch_id, job_id)
            update_job_state(
                batch_id,
                job_id,
                status="success",
                stage="completed",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                error_message="",
            )
            if status_callback:
                status_callback(f"任务 {job_id} 执行成功")
            if post_job_cooldown_seconds > 0:
                time.sleep(post_job_cooldown_seconds)
            refresh_batch_runtime(batch_id)
            return result
        except Exception as exc:  # noqa: BLE001
            last_error_message = f"{str(exc)}\n{traceback.format_exc()}"
            update_job_state(
                batch_id,
                job_id,
                status="failed",
                stage="preparing",
                finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                error_message=last_error_message,
                retry_count=attempt_index,
            )
            refresh_batch_runtime(batch_id)

            if attempt_index >= max_retries:
                if status_callback:
                    status_callback(f"任务 {job_id} 最终失败: {exc}")
                raise

            if status_callback:
                status_callback(
                    f"任务 {job_id} 失败，将在 {retry_interval_seconds} 秒后重试: {exc}"
                )
            if retry_interval_seconds > 0:
                time.sleep(retry_interval_seconds)

    raise RuntimeError(last_error_message or f"任务 {job_id} 执行失败")


def run_batch_jobs(
    batch_id: str,
    job_ids: list[str],
    job_runner: JobRunner,
    execution_mode: str,
    max_concurrency: int,
    submit_interval_seconds: float,
    retry_interval_seconds: float,
    post_job_cooldown_seconds: float,
    max_retries: int,
    status_callback: StatusCallback | None = None,
) -> dict[str, Any]:
    """
    按指定模式执行批量任务。

    :param batch_id: 批次标识。
    :param job_ids: 任务标识列表。
    :param job_runner: 实际任务执行函数。
    :param execution_mode: 执行模式，支持 `serial` 或 `parallel`。
    :param max_concurrency: 最大并发数。
    :param submit_interval_seconds: 新任务提交间隔。
    :param retry_interval_seconds: 任务重试间隔。
    :param post_job_cooldown_seconds: 任务完成后的冷却时间。
    :param max_retries: 最大重试次数。
    :param status_callback: 可选状态回调。
    :return: 批次执行结果摘要。
    """
    if not job_ids:
        return {"batch_id": batch_id, "success_jobs": [], "failed_jobs": []}

    success_jobs: list[str] = []
    failed_jobs: list[dict[str, str]] = []

    if execution_mode == "serial" or max_concurrency <= 1:
        for index, job_id in enumerate(job_ids, start=1):
            if status_callback:
                status_callback(f"开始处理第 {index}/{len(job_ids)} 个任务: {job_id}")
            try:
                _run_job_with_retries(
                    batch_id,
                    job_id,
                    job_runner,
                    max_retries,
                    retry_interval_seconds,
                    post_job_cooldown_seconds,
                    status_callback,
                )
                success_jobs.append(job_id)
            except Exception as exc:  # noqa: BLE001
                failed_jobs.append({"job_id": job_id, "error": str(exc)})

            if index < len(job_ids) and submit_interval_seconds > 0:
                time.sleep(submit_interval_seconds)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_job_id: dict[concurrent.futures.Future[Any], str] = {}
            for index, job_id in enumerate(job_ids, start=1):
                future = executor.submit(
                    _run_job_with_retries,
                    batch_id,
                    job_id,
                    job_runner,
                    max_retries,
                    retry_interval_seconds,
                    post_job_cooldown_seconds,
                    status_callback,
                )
                future_to_job_id[future] = job_id
                if status_callback:
                    status_callback(f"已提交第 {index}/{len(job_ids)} 个并行任务: {job_id}")
                if index < len(job_ids) and submit_interval_seconds > 0:
                    time.sleep(submit_interval_seconds)

            for future in concurrent.futures.as_completed(future_to_job_id):
                job_id = future_to_job_id[future]
                try:
                    future.result()
                    success_jobs.append(job_id)
                except Exception as exc:  # noqa: BLE001
                    failed_jobs.append({"job_id": job_id, "error": str(exc)})

    refresh_batch_runtime(batch_id)
    return {
        "batch_id": batch_id,
        "success_jobs": success_jobs,
        "failed_jobs": failed_jobs,
    }
