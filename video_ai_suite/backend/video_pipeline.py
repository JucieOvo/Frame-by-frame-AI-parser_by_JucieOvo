"""
模块名称：video_pipeline
功能描述：
    提供脱离 Streamlit 会话状态的单视频任务处理流水线。
    该模块服务于批量模式，统一负责关键帧提取、基于 MoviePy 的音频抽取、语音识别、视觉模型解析与结果落盘，
    确保单任务全部中间产物都落在任务私有目录中。

主要组件：
    - run_video_job_pipeline: 执行单视频任务的完整流水线。

依赖说明：
    - cv2: 关键帧保存。
    - moviepy: 视频音频抽取。
    - scenedetect: 场景检测与关键帧定位。
    - video_ai_suite.backend.model_clients: 多端点模型调用。
    - video_ai_suite.backend.job_storage: 状态回写。

作者：JucieOvo
创建日期：2026-04-21
修改记录：
    - 2026-04-21 JucieOvo: 创建批量模式单视频处理流水线。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import cv2
from moviepy import VideoFileClip
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from video_ai_suite.backend.job_storage import (
    load_job_manifest,
    update_job_manifest,
    update_job_state,
)
from video_ai_suite.backend.model_clients import invoke_vision_model
from video_ai_suite.backend.runtime import list_image_files


def _append_job_log(job_log_path: str, content: str) -> None:
    """
    向任务日志文件追加一行文本。

    :param job_log_path: 日志文件路径。
    :param content: 待追加内容。
    """
    os.makedirs(os.path.dirname(job_log_path), exist_ok=True)
    with open(job_log_path, "a", encoding="utf-8") as file:
        file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}\n")


def _write_text(file_path: str, content: str) -> None:
    """
    将文本内容写入文件。

    :param file_path: 目标文件路径。
    :param content: 文本内容。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def _extract_keyframes(video_path: str, output_dir: str, job_log_path: str) -> int:
    """
    从视频中提取关键帧到任务私有目录。

    :param video_path: 视频路径。
    :param output_dir: 关键帧输出目录。
    :param job_log_path: 任务日志路径。
    :return: 提取到的关键帧数量。
    """
    os.makedirs(output_dir, exist_ok=True)
    _append_job_log(job_log_path, f"开始抽取关键帧: {video_path}")

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    saved_count = 0
    for scene_index, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        target_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        if not success or frame is None:
            _append_job_log(job_log_path, f"关键帧提取失败，场景索引: {scene_index}")
            continue

        output_file = os.path.join(output_dir, f"{scene_index:04d}.png")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if cv2.imwrite(output_file, frame):
            saved_count += 1

    cap.release()
    _append_job_log(job_log_path, f"关键帧提取完成，共提取 {saved_count} 张")
    return saved_count


def _extract_audio(
    video_path: str,
    audio_path: str,
    job_log_path: str,
) -> str | None:
    """
    从视频中抽取 16k WAV 音频。

    :param video_path: 视频路径。
    :param audio_path: 输出音频路径。
    :param job_log_path: 任务日志路径。
    :return: 音频路径；抽取失败时返回 None。
    """
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    video_clip = None
    audio_clip = None
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        if audio_clip is None:
            _append_job_log(job_log_path, "当前视频未检测到可提取音频")
            return None

        audio_clip.write_audiofile(
            audio_path,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )
        _append_job_log(job_log_path, f"音频抽取完成: {audio_path}")
        return audio_path
    except Exception as exc:  # noqa: BLE001
        _append_job_log(job_log_path, f"音频抽取失败: {exc}")
        return None
    finally:
        if audio_clip is not None:
            audio_clip.close()
        if video_clip is not None:
            video_clip.close()


def _run_asr(
    audio_path: str,
    cache_dir: str,
    funasr_model: Any,
    report_file: str,
    job_log_path: str,
) -> str | None:
    """
    对任务音频执行语音识别并持久化结果。

    :param audio_path: 音频路径。
    :param cache_dir: 缓存目录。
    :param funasr_model: FunASR 模型对象。
    :param report_file: 任务报告路径。
    :param job_log_path: 任务日志路径。
    :return: 转录文本；失败时返回 None。
    """
    if funasr_model is None or not audio_path or not os.path.exists(audio_path):
        return None

    result = funasr_model.generate(input=audio_path, cache={}, language="auto", use_itn=True)
    if not result or not isinstance(result, list) or not result[0].get("text"):
        _append_job_log(job_log_path, "语音识别未返回有效文本")
        return None

    transcription = result[0]["text"]
    _write_text(os.path.join(cache_dir, "asr_transcription.txt"), f"语音转录内容：{transcription}")
    with open(report_file, "a", encoding="utf-8") as file:
        file.write("\n语音转录结果\n")
        file.write("=" * 80 + "\n")
        file.write(transcription + "\n")
    _append_job_log(job_log_path, "语音识别完成")
    return transcription


def _append_frame_result(report_file: str, frame_name: str, model_name: str, result_text: str) -> None:
    """
    将单帧分析结果追加到任务报告。

    :param report_file: 任务报告路径。
    :param frame_name: 帧文件名。
    :param model_name: 使用的模型名称。
    :param result_text: 分析结果文本。
    """
    with open(report_file, "a", encoding="utf-8") as file:
        file.write(f"图片: {frame_name}\n")
        file.write(f"使用模型: {model_name}\n")
        file.write("解析结果:\n")
        file.write(result_text.strip() + "\n")
        file.write("-" * 80 + "\n")


def _get_sorted_keyframes(directory: str) -> list[str]:
    """
    获取按帧号排序后的关键帧文件列表。

    :param directory: 关键帧目录。
    :return: 排序后的关键帧路径列表。
    """
    sortable_items: list[tuple[int, str]] = []
    for image_path in list_image_files(directory):
        try:
            frame_number = int(os.path.splitext(os.path.basename(image_path))[0])
        except ValueError:
            continue
        sortable_items.append((frame_number, image_path))
    return [path for _, path in sorted(sortable_items, key=lambda item: item[0])]


def run_video_job_pipeline(
    batch_id: str,
    job_id: str,
    endpoint: dict[str, Any],
    vision_model: str,
    user_prompt: str,
    funasr_model: Any = None,
    api_key_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    执行单视频任务的完整离线流水线。

    :param batch_id: 批次标识。
    :param job_id: 任务标识。
    :param endpoint: 视觉模型端点配置。
    :param vision_model: 视觉模型名称。
    :param user_prompt: 用户自定义提示词。
    :param funasr_model: FunASR 模型对象。
    :param api_key_overrides: API Key 覆盖字典。
    :return: 任务执行结果摘要。
    """
    job_manifest = load_job_manifest(batch_id, job_id)
    if not job_manifest:
        raise ValueError(f"任务不存在: {batch_id}/{job_id}")

    paths = job_manifest["paths"]
    source_video_path = paths["source_video_path"]
    report_file = paths["output_file"]
    cache_dir = paths["cache_dir"]
    keyframe_dir = paths["keyframe_dir"]
    metadata_file = paths["metadata_file"]
    job_log_path = paths["job_log"]

    update_job_manifest(
        batch_id,
        job_id,
        provider_endpoint_id=endpoint.get("provider_id") or endpoint.get("endpoint_id", ""),
        vision_model=vision_model,
    )

    update_job_state(batch_id, job_id, stage="preparing")
    _write_text(
        report_file,
        f"视频分析报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"视频文件: {source_video_path}\n"
        f"视觉端点: {endpoint.get('display_name', endpoint.get('provider_id') or endpoint.get('endpoint_id', '未知端点'))}\n"
        f"视觉模型: {vision_model}\n"
        + "=" * 80
        + "\n\n",
    )

    update_job_state(batch_id, job_id, stage="keyframe_extract")
    keyframe_count = _extract_keyframes(source_video_path, keyframe_dir, job_log_path)
    if keyframe_count <= 0:
        raise RuntimeError(f"任务 {job_id} 未提取到任何关键帧")

    update_job_state(batch_id, job_id, stage="audio_extract")
    audio_path = _extract_audio(source_video_path, paths["audio_path"], job_log_path)

    update_job_state(batch_id, job_id, stage="asr")
    transcription = _run_asr(audio_path, cache_dir, funasr_model, report_file, job_log_path)

    update_job_state(batch_id, job_id, stage="vlm_analysis")
    base_prompt = "请详细解析这张图片的内容，用连贯详细的语言描述其主要内容。如果图片中出现文字，请准确描述它。如果文字模糊不清，请说明并尝试推断其大意。"
    full_prompt = base_prompt + (f" {user_prompt}" if user_prompt else "")

    processed_frames = 0
    for image_path in _get_sorted_keyframes(keyframe_dir):
        frame_name = os.path.basename(image_path)
        frame_output_path = os.path.join(cache_dir, f"{os.path.splitext(frame_name)[0]}.txt")
        result_text = invoke_vision_model(
            endpoint,
            vision_model,
            full_prompt,
            image_path,
            api_key_overrides,
        )
        _write_text(frame_output_path, result_text)
        _append_frame_result(report_file, frame_name, vision_model, result_text)
        processed_frames += 1

    metadata = {
        "batch_id": batch_id,
        "job_id": job_id,
        "original_name": job_manifest.get("original_name", job_id),
        "source_video_path": source_video_path,
        "endpoint_id": endpoint.get("provider_id") or endpoint.get("endpoint_id", ""),
        "vision_model": vision_model,
        "keyframe_count": keyframe_count,
        "processed_frame_count": processed_frames,
        "has_asr_transcription": bool(transcription),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": paths,
    }
    with open(metadata_file, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    update_job_state(
        batch_id,
        job_id,
        stage="completed",
        artifacts={
            "metadata_file": metadata_file,
            "output_file": report_file,
            "keyframe_dir": keyframe_dir,
            "cache_dir": cache_dir,
        },
    )
    _append_job_log(job_log_path, f"任务完成，关键帧 {processed_frames} 张")
    return metadata
