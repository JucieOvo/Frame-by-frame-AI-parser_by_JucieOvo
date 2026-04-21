"""
模块名称：legacy_app
功能描述：
    包内正式保留的 Streamlit 页面主体文件。
    当前版本已经将正式启动项、公共后端能力与会话状态初始化拆分到 `video_ai_suite` 包，
    本文件暂时继续承载历史页面主体逻辑，以降低一次性重构带来的回归风险。

主要组件：
    - main: 正式页面主入口。
    - video_analysis_page: 视频分析页面。
    - multi_analysis_page: 多方式分析页面。

依赖说明：
    - video_ai_suite.backend.runtime: 缓存环境与路径处理。
    - video_ai_suite.backend.token_service: Token 统计计算。
    - video_ai_suite.streamlit_app.session_state: 会话状态初始化。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 接入正式拆分后的启动项、后端能力与 Streamlit 状态初始化。
"""

import asyncio
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from io import BytesIO
from typing import List

from video_ai_suite.backend.runtime import (
    DEFAULT_KEYFRAME_DIR,
    early_set_cache_env,
    get_program_dir,
    get_program_cache_dir,
    list_image_files,
    resolve_keyframe_directory,
)
from video_ai_suite.backend.token_service import (
    accumulate_token_usage,
    calculate_token_cost,
    create_empty_token_usage,
)

# 必须在导入模型相关库之前先设置缓存目录，避免第三方模型库读取到用户本机旧缓存配置。
early_set_cache_env()

import cv2
import numpy as np
import ollama
import streamlit as st
from tqdm import tqdm
from openai import OpenAI
from PIL import Image
import scenedetect
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import aiohttp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import funasr
from funasr import AutoModel
from modelscope.utils.hub import snapshot_download
from modelscope import snapshot_download as modelscope_download
import psutil
import torch
from video_ai_suite.streamlit_app.session_state import initialize_session_state

# ollama客户端
from ollama import Client

client = Client(host="http://localhost:11434")

initialize_session_state()


def reset_token_usage():
    """重置 Token 使用统计。"""
    st.session_state.token_usage = create_empty_token_usage()


def update_token_usage(input_tokens, output_tokens):
    """更新 Token 使用统计。"""
    st.session_state.token_usage = accumulate_token_usage(
        st.session_state.token_usage,
        input_tokens,
        output_tokens,
    )


def display_token_usage(container=None):
    """显示Token使用统计和费用

    Args:
        container: Streamlit容器对象，用于实时更新显示。如果为None，则在当前位置显示
    """
    usage = st.session_state.token_usage

    if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
        return None

    # 如果提供了容器，在容器中显示；否则在当前位置显示
    display_context = container if container else st

    with display_context:
        st.markdown("---")
        st.markdown("### 💰 用量统计与费用")

        # 创建三列显示统计信息
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="输入 Token",
                value=f"{usage['input_tokens']:,}",
                help="包括图片编码和文本输入的Token数",
            )

        with col2:
            st.metric(
                label="输出 Token",
                value=f"{usage['output_tokens']:,}",
                help="模型生成的文本Token数",
            )

        with col3:
            st.metric(
                label="预估总费用",
                value=f"¥{usage['total_cost']:.4f}",
                help="基于阶梯价格计算的预估费用",
            )

    # 详细费用分解
    with st.expander("📊 费用详细分解", expanded=False):
        cost_breakdown = calculate_token_cost(
            usage["input_tokens"], usage["output_tokens"]
        )

        st.markdown(f"""
        **输入Token费用：** ¥{cost_breakdown["input_cost"]:.4f}
        - Token数：{usage["input_tokens"]:,}
        
        **输出Token费用：** ¥{cost_breakdown["output_cost"]:.4f}
        - Token数：{usage["output_tokens"]:,}
        
        **总计：** ¥{cost_breakdown["total_cost"]:.4f}
        
        ---
        
        **计费说明：**
        
        **输入Token阶梯价格（每千Token）：**
        - 0-32K: ¥0.001
        - 32K-128K: ¥0.0015
        - 128K-256K: ¥0.003
        
        **输出Token阶梯价格（每千Token）：**
        - 0-32K: ¥0.01
        - 32K-128K: ¥0.015
        - 128K-256K: ¥0.03
        
        **注意：**
        - 图片会被编码为Token（单图最大16,384 Token）
        - 费用为预估值，实际费用以阿里云账单为准
        - 免费额度：输入和输出各100万Token（90天有效期）
        """)


# --- 免责声明模块 ---
def show_disclaimer():
    """显示程序的免责声明和使用条款"""
    disclaimer = """
**免责声明 下滑查看完整声明 和 操作流程**  
免责声明
本软件（"[视频切分转文字程序]"）按"原样"提供，不提供任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性和不侵权的暗示担保。在任何情况下，作者或版权所有者均不对因软件或软件的使用或其他交易而产生、由软件引起或与之相关的任何索赔、损害或其他责任（无论是合同、侵权还是其他形式的责任）承担任何责任，即使事先被告知此类损害的可能性。
重要提示
本软件可能存在错误、缺陷或不完善之处。
作者不保证软件是：
无错误的。
不间断或可用的。
安全的（不会导致数据丢失、系统损坏或安全漏洞）。
符合您的特定需求或期望。
在法律上、技术上或商业上可行的。
用户自担风险： 您使用、修改、分发本软件或依赖本软件的行为完全由您自己承担风险。您应对使用软件可能导致的任何及所有后果负责，包括但不限于：
数据丢失或损坏。
系统故障或中断。
业务中断。
安全漏洞或数据泄露。
财务损失。
任何其他直接、间接、附带、特殊、后果性或惩罚性损害。
第三方依赖： 本软件可能依赖其他第三方库、服务或组件（统称"依赖项"）。这些依赖项有其自身的许可证和免责声明。本项目的作者不对任何依赖项的功能、安全性、可靠性或合法性负责或提供担保。 您需要自行审查并遵守所有依赖项的条款。
非专业建议： 如果本软件涉及特定领域（如金融、医疗、安全等），其输出或功能不应被视为专业建议。在做出任何依赖软件输出的决策之前，请务必咨询该领域的合格专业人士。
贡献者： 本软件可能包含由社区贡献者提交的代码。项目维护者（作者）会尽力审查贡献，但不保证所有贡献的代码都是安全、无错误或合适的。接受贡献并不意味着维护者对其承担额外的责任。
您的责任
作为软件的用户（或修改者、分发者），您有责任：
在使用前仔细评估软件是否适合您的目的。
在非生产环境中进行充分的测试。
实施适当的安全措施和数据备份。
遵守软件所使用的开源许可证的所有条款。
遵守所有适用的法律和法规。
总结
使用本软件即表示您理解并完全接受本免责声明中的所有条款和风险。如果您不同意这些条款，请不要使用、修改或分发本软件。
本程序无任何政治目的，没有任何政治影射
1. 本程序从环境变量获取 DASHSCOPE_API_KEY (如果使用阿里云)
2. 该密钥仅存储在环境变量中，程序运行时仅在环境变量中临时读取
3. 本程序不会以任何形式:
   - 将API密钥传输到除阿里云以外的外部服务器
   - 将API密钥写入日志/文件
   - 持久化存储API密钥
4. 用户需自行保管好API密钥，本程序开发者不承担因密钥泄露导致的任何责任
使用本程序即表示您同意:
- 您是该API密钥的合法持有者
- 您已了解密钥泄露的风险
- 您自愿承担使用该API密钥的所有责任
"""
    st.warning(disclaimer)


def show_important_reminder():
    with st.title("重要提示"):
        st.info("""
        **重要提示:**
        1. 在完成对视频的分析后，如果您没有再使用的需求，请停止ollama模型的运行
           - 方法一：在完成分析后，在左侧边栏最下方点击停止ollama模型
           - 方法二：在完成分析后，在命令提示符中输入ollama stop 模型名称，如果您忘记了模型名称，输入ollama ps以查询
        2. 在选择ollama模型时请选择vlm[视觉语言模型]如qwen、LLaVa等，作者不保证所有模型都能以预期运行
        3. 如果您运行程序以后没有发现有左侧边栏，请确认您的网页缩放等级正常后，在左上角应有'>'按钮，请单击
        4. 边栏与提示区的空间占比可以调节，将鼠标指针移动至边栏与提示区交界处以调节
        5. 如果您需要更改您的API密钥，请手动更改，如您不知道如何更改，请询问神奇的度娘
        """)


# --- 操作流程说明模块 ---
def show_operation_guide():
    """向用户展示程序的基本操作步骤"""
    with st.title("操作指南"):
        st.success("""
        **操作流程:**
        1. 在左侧边栏配置模型参数
        2. 上传视频文件[视频大小无上限——支持格式: mp4, avi, mov, mkv, flv, wmv]
        3. 点击"开始分析"按钮
        4. 程序将自动:
           - 提取视频关键帧
           - 使用大模型分析关键帧内容
           - 构建向量数据库（使用中文Embedding模型）
        5. 分析结果将显示在页面并保存到[草稿.txt]
        6. 切换到「多方式分析」页面使用RAG智能检索功能
        
        **RAG智能检索功能:**
        - 完整分析报告：生成连贯的视频内容总结
        - 速读摘要：快速了解视频核心内容
        - 智能检索：基于中文Embedding模型的语义搜索，精准回答问题
          * 支持多种检索策略（相似度、MMR去重、阈值过滤）
          * 可视化展示相关关键帧
          * 自动引用来源
          * 查询历史记录
        """)


# --- FFmpeg检查模块 ---
def check_ffmpeg():
    """
    检查FFmpeg是否可用，优先使用项目内置的FFmpeg

    返回:
        bool: FFmpeg是否可用
        str: 可用的FFmpeg路径
    """
    # 项目内置FFmpeg路径
    builtin_ffmpeg_path = os.path.join(
        get_program_dir(), "ffmpeg_downlaod", "bin", "ffmpeg.exe"
    )

    # 首先检查项目内置的FFmpeg
    if os.path.exists(builtin_ffmpeg_path):
        try:
            subprocess.run(
                [builtin_ffmpeg_path, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            st.info(f"✅ 使用项目内置FFmpeg: {builtin_ffmpeg_path}")
            return True, builtin_ffmpeg_path
        except:
            st.warning(f"⚠️ 项目内置FFmpeg不可用: {builtin_ffmpeg_path}")

    # 如果项目内置FFmpeg不可用，检查系统PATH中的FFmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        st.info("✅ 使用系统PATH中的FFmpeg")
        return True, "ffmpeg"
    except:
        st.error("❌ 未找到可用的FFmpeg，请确保FFmpeg已安装并添加到系统PATH")
        return False, None


# --- 获取Ollama模型列表 ---
def get_ollama_models():
    """
    获取可用的Ollama模型列表

    注意：新版本的Ollama API必须使用Client实例，不能直接调用ollama.list()
    """
    try:
        # 新版本必须使用Client实例
        client = Client(host="http://localhost:11434")
        response = client.list()

        # 安全地获取模型列表，优先使用model属性，如果不存在则使用name属性
        models_list = []
        for model in response.get("models", []):
            # 新版本使用model属性，旧版本使用name属性
            model_name = model.get("model") or model.get("name")
            if model_name:
                models_list.append(model_name)
        return models_list
    except Exception as e:
        st.error(f"""无法获取Ollama模型列表: {e}""")
        # 尝试直接调用命令行作为备用方案
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip().split("\n")
            models = []
            if lines and lines[0].startswith("NAME"):
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception as e2:
            st.error(f"调用 'ollama list' 命令也失败了: {e2}")
            return []


# --- 模型选择和初始化模块 ---
def init_model():
    """初始化选择的模型"""
    try:
        if st.session_state.use_ollama:
            # 检查是否已选择模型
            if not st.session_state.selected_model:
                st.error("请先选择一个Ollama模型")
                return False
            st.success(f"已选择 Ollama 模型: {st.session_state.selected_model}")
            return True
        else:
            # 从环境变量获取API密钥
            st.session_state.api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not st.session_state.api_key:
                st.error("请先在环境变量中设置 DASHSCOPE_API_KEY")
                return False

            # 初始化阿里云客户端
            st.session_state.client = OpenAI(
                api_key=st.session_state.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            # 图片分析使用VLM模型
            st.session_state.selected_model = st.session_state.vlm_model
            st.success(f"已选择阿里云 DashScope")
            st.info(f"视觉模型: {st.session_state.vlm_model}")
            return True
    except Exception as e:
        st.error(f"模型初始化失败: {str(e)}")
        return False


# --- 智能抽帧模块 (pyscenedetect) ---
def extract_keyframes_pyscenedetect(video_path, output_dir):
    """
    使用pyscenedetect进行智能场景检测和关键帧提取

    参数:
        video_path (str): 视频文件路径
        output_dir (str): 关键帧输出目录

    返回:
        int: 成功提取的关键帧数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"[抽帧] 处理视频: {os.path.basename(video_path)}")

    try:
        # 使用新的API打开视频
        video = open_video(video_path)
        scene_manager = SceneManager()

        # 添加内容检测器，设置阈值检测场景变化
        scene_manager.add_detector(ContentDetector(threshold=30.0))

        # 开始场景检测
        scene_manager.detect_scenes(video=video)

        # 获取场景列表
        scene_list = scene_manager.get_scene_list()

        # 获取视频信息
        fps = video.frame_rate

        print(f"[抽帧] 检测到 {len(scene_list)} 个场景")

        # 创建场景信息列表，保持序列化
        # 关键点：预先确定每个场景的输出文件名，确保序列化不混乱
        scene_info_list = []
        for i, scene in enumerate(scene_list):
            # 获取场景的开始和结束帧
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()

            # 选择场景中间帧作为关键帧
            middle_frame = (start_frame + end_frame) // 2

            # 预先确定输出文件名，使用场景索引确保序列化
            scene_info_list.append(
                {
                    "scene_index": i,  # 场景索引，用于保持序列
                    "start_frame": start_frame,  # 场景开始帧
                    "end_frame": end_frame,  # 场景结束帧
                    "middle_frame": middle_frame,  # 场景中间帧（关键帧位置）
                    "output_file": os.path.join(
                        output_dir, f"{i:04d}.png"
                    ),  # 固定文件名，避免并行冲突
                }
            )

        # 使用批量处理提取关键帧（优化版，速度更快）
        try:
            keyframe_count = extract_keyframes_batch(video_path, scene_info_list)

            # 检查是否成功提取了足够的关键帧
            if keyframe_count > 0:
                print(f"[抽帧] ✅ 关键帧提取完成! 共提取 {keyframe_count} 个关键帧")
                return keyframe_count
            else:
                print("[抽帧] ⚠️ 批量提取未成功，尝试使用备用方法...")
                raise Exception("批量提取失败，关键帧数为0")
        except Exception as e:
            print(f"[抽帧] 批量提取方法出错: {str(e)}")
            print("[抽帧] 🔄 正在使用并行提取方法（备用方案）...")
            keyframe_count = extract_keyframes_parallel(video_path, scene_info_list)
            print(f"[抽帧] ✅ 关键帧提取完成! 共提取 {keyframe_count} 个关键帧")
            return keyframe_count

    except Exception as e:
        print(f"[抽帧] pyscenedetect场景检测失败: {e}")
        # 回退到原来的方法
        print("[抽帧] 回退到传统关键帧提取方法...")
        return extract_keyframes_traditional(video_path, output_dir)


# --- 批量关键帧提取模块 (优化版) ---
def extract_keyframes_batch(video_path, scene_info_list):
    """
    批量提取关键帧 - 优化版本

    核心优化思路：
    1. 先完成场景检测，确定所有需要提取的帧号
    2. 使用OpenCV一次性遍历视频，在对应帧号处保存
    3. 避免多次调用FFmpeg，大幅提升速度

    参数:
        video_path (str): 视频文件路径
        scene_info_list (list): 场景信息列表，包含每个场景的帧信息和输出路径

    返回:
        int: 成功提取的关键帧数量
    """
    if not scene_info_list:
        return 0

    print(f"[抽帧] 🚀 准备批量提取 {len(scene_info_list)} 个关键帧...")

    try:
        # 构建帧号到输出文件的映射（使用字典快速查找）
        frame_to_output = {}
        for scene_info in scene_info_list:
            frame_num = scene_info["middle_frame"]
            output_file = scene_info["output_file"]
            frame_to_output[frame_num] = output_file

        # 获取所有需要提取的帧号（排序后可以顺序读取）
        target_frames = sorted(frame_to_output.keys())

        print(f"[抽帧] 📋 场景检测完成，确定了 {len(target_frames)} 个关键帧位置")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[抽帧] ❌ 无法打开视频文件: {video_path}")
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"[抽帧] 📹 视频信息: 总帧数 {total_frames}, FPS {fps:.2f}")
        print(
            f"[抽帧] 🎯 需要提取的帧号: {target_frames[:5]}{'...' if len(target_frames) > 5 else ''}"
        )
        print(f"[抽帧] ⏳ 正在批量提取关键帧...")

        # 提取关键帧
        successful_count = 0

        # 使用更简单可靠的方法：直接对每个目标帧进行跳转和读取
        for idx, target_frame in enumerate(target_frames):
            try:
                # 直接跳转到目标帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                # 验证跳转是否成功（有些视频格式seek不准确）
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if abs(actual_pos - target_frame) > 5:
                    # 跳转不准确，使用逐帧读取的方式
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 10))
                    for _ in range(min(15, target_frame + 1)):
                        ret = cap.grab()
                        if not ret:
                            break

                # 读取该帧
                ret, frame = cap.read()

                # 调试信息
                if idx == 0:
                    print(
                        f"[抽帧-调试] 第1帧 - ret={ret}, frame is None={frame is None}"
                    )
                    if frame is not None:
                        print(f"[抽帧-调试] 第1帧 - frame.shape={frame.shape}")

                if ret and frame is not None:
                    output_file = frame_to_output[target_frame]

                    # 调试输出路径
                    if idx == 0:
                        print(f"[抽帧-调试] 输出路径: {output_file}")
                        print(f"[抽帧-调试] 输出目录: {os.path.dirname(output_file)}")

                    # 确保输出目录存在
                    output_dir = os.path.dirname(output_file)
                    if output_dir:  # 如果有目录部分
                        os.makedirs(output_dir, exist_ok=True)

                    # 使用 cv2.imencode + 手动写入来支持中文路径
                    try:
                        # 将图像编码为PNG格式
                        success, encoded_img = cv2.imencode(".png", frame)
                        if success:
                            # 手动写入文件（支持中文路径）
                            with open(output_file, "wb") as f:
                                f.write(encoded_img.tobytes())
                            success = True
                        else:
                            success = False
                    except Exception as e:
                        print(f"[抽帧] 保存失败: {str(e)}")
                        success = False

                    if success and os.path.exists(output_file):
                        # 验证文件确实被创建且有内容
                        file_size = os.path.getsize(output_file)
                        if file_size > 0:
                            successful_count += 1
                            # 第一帧和每10帧显示一次进度
                            if successful_count == 1:
                                print(
                                    f"[抽帧] ✅ 第一帧提取成功！文件: {output_file}, 大小: {file_size} bytes"
                                )
                            elif successful_count % 10 == 0:
                                print(
                                    f"[抽帧] ⏳ 已提取: {successful_count}/{len(target_frames)}"
                                )
                        else:
                            print(f"[抽帧] ⚠️ 保存的文件 {output_file} 大小为0")
                    else:
                        print(
                            f"[抽帧] ⚠️ cv2.imwrite返回: {success}, 文件存在: {os.path.exists(output_file)}"
                        )
                else:
                    print(f"[抽帧] ⚠️ 读取帧 {target_frame} 失败")

            except Exception as e:
                print(f"[抽帧] ⚠️ 处理帧 {target_frame} 时出错: {str(e)}")

        cap.release()

        print(
            f"[抽帧] ✅ 批量提取完成！成功提取 {successful_count}/{len(target_frames)} 个关键帧"
        )
        return successful_count

    except Exception as e:
        print(f"[抽帧] ❌ 批量提取关键帧时出错: {str(e)}")
        import traceback

        print(f"[抽帧] 详细错误: {traceback.format_exc()}")

        return 0


# --- 并行关键帧提取模块 (保留作为备用) ---
def extract_keyframes_parallel(video_path, scene_info_list):
    """
    并行提取关键帧，保持序列化（备用方案）

    核心设计原则：
    1. 预先确定文件名，避免并行写入冲突
    2. 使用线程安全计数器更新进度
    3. 限制并发数，避免资源竞争
    4. 保持帧的序列化顺序

    参数:
        video_path (str): 视频文件路径
        scene_info_list (list): 场景信息列表，包含每个场景的帧信息和输出路径

    返回:
        int: 成功提取的关键帧数量
    """
    import concurrent.futures
    import threading

    # 线程安全的计数器，用于进度更新
    completed_count = 0
    lock = threading.Lock()

    def update_progress():
        """线程安全的进度更新函数"""
        nonlocal completed_count
        with lock:
            completed_count += 1
            if completed_count % 10 == 0:
                print(
                    f"[抽帧-并行] ⏳ 已完成: {completed_count}/{len(scene_info_list)}"
                )

    def extract_single_frame(scene_info):
        """
        提取单个场景的关键帧（使用OpenCV）

        参数:
            scene_info (dict): 场景信息字典

        返回:
            bool: 提取是否成功
        """
        scene_index = scene_info["scene_index"]
        middle_frame = scene_info["middle_frame"]
        output_file = scene_info["output_file"]

        # 使用OpenCV提取关键帧（不再使用FFmpeg）
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                update_progress()
                return False

            # 跳转到目标帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

            # 读取帧
            ret, frame = cap.read()

            if ret and frame is not None:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # 使用 cv2.imencode + 手动写入来支持中文路径
                try:
                    success, encoded_img = cv2.imencode(".png", frame)
                    if success:
                        with open(output_file, "wb") as f:
                            f.write(encoded_img.tobytes())
                        success = True
                    else:
                        success = False
                except:
                    success = False

                # 验证文件
                if (
                    success
                    and os.path.exists(output_file)
                    and os.path.getsize(output_file) > 0
                ):
                    cap.release()
                    update_progress()
                    return True

            cap.release()
            update_progress()
            return False

        except Exception as e:
            # 任何异常都返回False
            update_progress()
            return False

    # 使用线程池并行处理（使用OpenCV）
    # 限制最大并发数为4，避免过多并发导致的资源竞争
    max_workers = min(4, len(scene_info_list))
    successful_count = 0

    # 显示初始状态
    print(
        f"[抽帧-并行] 准备并行提取 {len(scene_info_list)} 个场景的关键帧（使用OpenCV）..."
    )

    # 使用线程池执行器进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有提取任务到线程池
        future_to_scene = {
            executor.submit(extract_single_frame, scene_info): scene_info
            for scene_info in scene_info_list
        }

        # 收集处理结果，确保所有任务完成
        for future in concurrent.futures.as_completed(future_to_scene):
            scene_info = future_to_scene[future]
            try:
                success = future.result()
                if success:
                    successful_count += 1
            except Exception as e:
                # 记录失败信息
                print(
                    f"[抽帧-并行] ⚠️ 提取场景 {scene_info['scene_index'] + 1} 关键帧失败: {str(e)}"
                )

    print(
        f"[抽帧-并行] ✅ 并行提取完成！成功提取 {successful_count}/{len(scene_info_list)} 个关键帧"
    )
    return successful_count


# --- 传统关键帧提取方法 (备用) ---
def extract_keyframes_traditional(video_path, output_dir):
    """传统关键帧提取方法，作为备用方案"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[抽帧-传统] 错误: 无法打开视频文件 {video_path}")
        return 0

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"[抽帧-传统] 视频信息: {duration:.1f}秒, {total_frames}帧, {fps:.2f}FPS")

    # 设置关键帧间隔（每5秒提取一帧）
    frame_interval = int(fps * 5)
    keyframe_count = 0

    for frame_index in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            output_file = os.path.join(output_dir, f"{keyframe_count:04d}.png")
            # 使用 cv2.imencode + 手动写入来支持中文路径
            try:
                success, encoded_img = cv2.imencode(".png", frame)
                if success:
                    with open(output_file, "wb") as f:
                        f.write(encoded_img.tobytes())
                    keyframe_count += 1
            except Exception as e:
                print(f"[抽帧-传统] 保存失败: {str(e)}")

            if keyframe_count % 10 == 0:
                print(f"[抽帧-传统] ⏳ 已提取: {keyframe_count}")

    cap.release()

    print(f"[抽帧-传统] ✅ 传统方法提取完成! 共提取 {keyframe_count} 个关键帧")
    return keyframe_count


def extract_keyframes(video_path, output_dir):
    """关键帧提取主函数 - 优先使用pyscenedetect

    参数:
        video_path: 视频文件路径
        output_dir: 关键帧输出目录
    """
    return extract_keyframes_pyscenedetect(video_path, output_dir)


def extract_keyframes_with_heartbeat(video_path, output_dir, heartbeat_result):
    """带心跳监控的关键帧提取函数

    参数:
        video_path: 视频文件路径
        output_dir: 关键帧输出目录
        heartbeat_result: 心跳监控结果字典，用于更新心跳状态

    返回:
        int: 成功提取的关键帧数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"[抽帧-心跳] 处理视频: {os.path.basename(video_path)}")

    try:
        # 发送初始心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] = 0

        # 使用新的API打开视频
        video = open_video(video_path)
        scene_manager = SceneManager()

        # 添加内容检测器，设置阈值检测场景变化
        scene_manager.add_detector(ContentDetector(threshold=30.0))

        # 发送场景检测开始心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] += 1
        print(
            f"[抽帧-心跳] 场景检测开始，心跳计数: {heartbeat_result['heartbeat_count']}"
        )

        # 开始场景检测
        scene_manager.detect_scenes(video=video)

        # 获取场景列表
        scene_list = scene_manager.get_scene_list()

        # 获取视频信息
        fps = video.frame_rate

        print(f"[抽帧-心跳] 检测到 {len(scene_list)} 个场景")

        # 发送场景检测完成心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] += 1
        print(
            f"[抽帧-心跳] 场景检测完成，心跳计数: {heartbeat_result['heartbeat_count']}"
        )

        # 创建场景信息列表，保持序列化
        scene_info_list = []
        for i, scene in enumerate(scene_list):
            # 获取场景的开始和结束帧
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()

            # 选择场景中间帧作为关键帧
            middle_frame = (start_frame + end_frame) // 2

            # 预先确定输出文件名，使用场景索引确保序列化
            scene_info_list.append(
                {
                    "scene_index": i,  # 场景索引，用于保持序列
                    "start_frame": start_frame,  # 场景开始帧
                    "end_frame": end_frame,  # 场景结束帧
                    "middle_frame": middle_frame,  # 场景中间帧（关键帧位置）
                    "output_file": os.path.join(
                        output_dir, f"{i:04d}.png"
                    ),  # 固定文件名，避免并行冲突
                }
            )

        # 使用批量处理提取关键帧（优化版，速度更快）
        try:
            # 发送批量提取开始心跳
            heartbeat_result["last_heartbeat"] = time.time()
            heartbeat_result["heartbeat_count"] += 1
            print(
                f"[抽帧-心跳] 批量提取开始，心跳计数: {heartbeat_result['heartbeat_count']}"
            )

            keyframe_count = extract_keyframes_batch_with_heartbeat(
                video_path, scene_info_list, heartbeat_result
            )

            # 检查是否成功提取了足够的关键帧
            if keyframe_count > 0:
                # 发送完成心跳
                heartbeat_result["last_heartbeat"] = time.time()
                heartbeat_result["heartbeat_count"] += 1
                print(
                    f"[抽帧-心跳] ✅ 关键帧提取完成! 共提取 {keyframe_count} 个关键帧，总心跳: {heartbeat_result['heartbeat_count']}"
                )
                return keyframe_count
            else:
                print("[抽帧-心跳] ⚠️ 批量提取未成功，尝试使用备用方法...")
                raise Exception("批量提取失败，关键帧数为0")
        except Exception as e:
            print(f"[抽帧-心跳] 批量提取方法出错: {str(e)}")
            print("[抽帧-心跳] 🔄 正在使用并行提取方法（备用方案）...")

            # 发送备用方法开始心跳
            heartbeat_result["last_heartbeat"] = time.time()
            heartbeat_result["heartbeat_count"] += 1

            keyframe_count = extract_keyframes_parallel_with_heartbeat(
                video_path, scene_info_list, heartbeat_result
            )

            # 发送备用方法完成心跳
            heartbeat_result["last_heartbeat"] = time.time()
            heartbeat_result["heartbeat_count"] += 1

            print(
                f"[抽帧-心跳] ✅ 关键帧提取完成! 共提取 {keyframe_count} 个关键帧，总心跳: {heartbeat_result['heartbeat_count']}"
            )
            return keyframe_count

    except Exception as e:
        print(f"[抽帧-心跳] pyscenedetect场景检测失败: {e}")
        # 回退到原来的方法
        print("[抽帧-心跳] 回退到传统关键帧提取方法...")

        # 发送传统方法开始心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] += 1

        keyframe_count = extract_keyframes_traditional_with_heartbeat(
            video_path, output_dir, heartbeat_result
        )

        # 发送传统方法完成心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] += 1

        return keyframe_count


def extract_keyframes_batch_with_heartbeat(
    video_path, scene_info_list, heartbeat_result
):
    """带心跳监控的批量关键帧提取函数

    参数:
        video_path (str): 视频文件路径
        scene_info_list (list): 场景信息列表
        heartbeat_result: 心跳监控结果字典

    返回:
        int: 成功提取的关键帧数量
    """
    if not scene_info_list:
        return 0

    print(f"[抽帧-心跳-批量] 准备批量提取 {len(scene_info_list)} 个关键帧...")

    try:
        # 构建帧号到输出文件的映射
        frame_to_output = {}
        for scene_info in scene_info_list:
            frame_num = scene_info["middle_frame"]
            output_file = scene_info["output_file"]
            frame_to_output[frame_num] = output_file

        # 获取所有需要提取的帧号（排序后可以顺序读取）
        target_frames = sorted(frame_to_output.keys())

        print(
            f"[抽帧-心跳-批量] 场景检测完成，确定了 {len(target_frames)} 个关键帧位置"
        )

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[抽帧-心跳-批量] ❌ 无法打开视频文件: {video_path}")
            return 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"[抽帧-心跳-批量] 📹 视频信息: 总帧数 {total_frames}, FPS {fps:.2f}")
        print(f"[抽帧-心跳-批量] ⏳ 正在批量提取关键帧...")

        # 提取关键帧
        successful_count = 0

        for idx, target_frame in enumerate(target_frames):
            try:
                # 每处理5帧发送一次心跳
                if idx % 5 == 0:
                    heartbeat_result["last_heartbeat"] = time.time()
                    heartbeat_result["heartbeat_count"] += 1
                    print(
                        f"[抽帧-心跳-批量] 处理进度: {idx + 1}/{len(target_frames)}，心跳计数: {heartbeat_result['heartbeat_count']}"
                    )

                # 直接跳转到目标帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

                # 验证跳转是否成功
                actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if abs(actual_pos - target_frame) > 5:
                    # 跳转不准确，使用逐帧读取的方式
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 10))
                    for _ in range(min(15, target_frame + 1)):
                        ret = cap.grab()
                        if not ret:
                            break

                # 读取该帧
                ret, frame = cap.read()

                if ret and frame is not None:
                    output_file = frame_to_output[target_frame]

                    # 确保输出目录存在
                    output_dir = os.path.dirname(output_file)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)

                    # 使用 cv2.imencode + 手动写入来支持中文路径
                    try:
                        success, encoded_img = cv2.imencode(".png", frame)
                        if success:
                            with open(output_file, "wb") as f:
                                f.write(encoded_img.tobytes())
                            success = True
                        else:
                            success = False
                    except Exception as e:
                        print(f"[抽帧-心跳-批量] 保存失败: {str(e)}")
                        success = False

                    if success and os.path.exists(output_file):
                        # 验证文件确实被创建且有内容
                        file_size = os.path.getsize(output_file)
                        if file_size > 0:
                            successful_count += 1
                            # 第一帧和每10帧显示一次进度
                            if successful_count == 1:
                                print(
                                    f"[抽帧-心跳-批量] ✅ 第一帧提取成功！文件: {output_file}, 大小: {file_size} bytes"
                                )
                            elif successful_count % 10 == 0:
                                print(
                                    f"[抽帧-心跳-批量] ⏳ 已提取: {successful_count}/{len(target_frames)}"
                                )
                        else:
                            print(
                                f"[抽帧-心跳-批量] ⚠️ 保存的文件 {output_file} 大小为0"
                            )
                    else:
                        print(
                            f"[抽帧-心跳-批量] ⚠️ cv2.imwrite返回: {success}, 文件存在: {os.path.exists(output_file)}"
                        )
                else:
                    print(f"[抽帧-心跳-批量] ⚠️ 读取帧 {target_frame} 失败")

            except Exception as e:
                print(f"[抽帧-心跳-批量] ⚠️ 处理帧 {target_frame} 时出错: {str(e)}")

        cap.release()

        # 发送完成心跳
        heartbeat_result["last_heartbeat"] = time.time()
        heartbeat_result["heartbeat_count"] += 1

        print(
            f"[抽帧-心跳-批量] ✅ 批量提取完成！成功提取 {successful_count}/{len(target_frames)} 个关键帧"
        )
        return successful_count

    except Exception as e:
        print(f"[抽帧-心跳-批量] ❌ 批量提取关键帧时出错: {str(e)}")
        import traceback

        print(f"[抽帧-心跳-批量] 详细错误: {traceback.format_exc()}")

        return 0


def extract_keyframes_parallel_with_heartbeat(
    video_path, scene_info_list, heartbeat_result
):
    """带心跳监控的并行关键帧提取函数（备用方案）

    参数:
        video_path (str): 视频文件路径
        scene_info_list (list): 场景信息列表
        heartbeat_result: 心跳监控结果字典

    返回:
        int: 成功提取的关键帧数量
    """
    import concurrent.futures
    import threading

    # 线程安全的计数器，用于进度更新
    completed_count = 0
    lock = threading.Lock()

    def update_progress():
        """线程安全的进度更新函数"""
        nonlocal completed_count
        with lock:
            completed_count += 1
            # 每完成5个任务发送一次心跳
            if completed_count % 5 == 0:
                heartbeat_result["last_heartbeat"] = time.time()
                heartbeat_result["heartbeat_count"] += 1
                print(
                    f"[抽帧-心跳-并行] ⏳ 已完成: {completed_count}/{len(scene_info_list)}，心跳计数: {heartbeat_result['heartbeat_count']}"
                )

    def extract_single_frame(scene_info):
        """提取单个场景的关键帧（使用OpenCV）"""
        scene_index = scene_info["scene_index"]
        middle_frame = scene_info["middle_frame"]
        output_file = scene_info["output_file"]

        # 使用OpenCV提取关键帧
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                update_progress()
                return False

            # 跳转到目标帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

            # 读取帧
            ret, frame = cap.read()

            if ret and frame is not None:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # 使用 cv2.imencode + 手动写入来支持中文路径
                try:
                    success, encoded_img = cv2.imencode(".png", frame)
                    if success:
                        with open(output_file, "wb") as f:
                            f.write(encoded_img.tobytes())
                        success = True
                    else:
                        success = False
                except:
                    success = False

                cap.release()

                if success:
                    update_progress()
                    return True
                else:
                    update_progress()
                    return False
            else:
                cap.release()
                update_progress()
                return False

        except Exception as e:
            print(f"[抽帧-心跳-并行] ⚠️ 提取场景 {scene_index + 1} 关键帧失败: {str(e)}")
            update_progress()
            return False

    # 使用线程池并行处理
    max_workers = min(4, len(scene_info_list))
    successful_count = 0

    # 显示初始状态
    print(f"[抽帧-心跳-并行] 准备并行提取 {len(scene_info_list)} 个场景的关键帧...")

    # 发送开始心跳
    heartbeat_result["last_heartbeat"] = time.time()
    heartbeat_result["heartbeat_count"] += 1

    # 使用线程池执行器进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有提取任务到线程池
        future_to_scene = {
            executor.submit(extract_single_frame, scene_info): scene_info
            for scene_info in scene_info_list
        }

        # 收集处理结果，确保所有任务完成
        for future in concurrent.futures.as_completed(future_to_scene):
            scene_info = future_to_scene[future]
            try:
                success = future.result()
                if success:
                    successful_count += 1
            except Exception as e:
                # 记录失败信息
                print(
                    f"[抽帧-心跳-并行] ⚠️ 提取场景 {scene_info['scene_index'] + 1} 关键帧失败: {str(e)}"
                )

    # 发送完成心跳
    heartbeat_result["last_heartbeat"] = time.time()
    heartbeat_result["heartbeat_count"] += 1

    print(
        f"[抽帧-心跳-并行] ✅ 并行提取完成！成功提取 {successful_count}/{len(scene_info_list)} 个关键帧"
    )
    return successful_count


def extract_keyframes_traditional_with_heartbeat(
    video_path, output_dir, heartbeat_result
):
    """带心跳监控的传统关键帧提取方法（备用方案）

    参数:
        video_path (str): 视频文件路径
        output_dir (str): 关键帧输出目录
        heartbeat_result: 心跳监控结果字典

    返回:
        int: 成功提取的关键帧数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[抽帧-心跳-传统] 错误: 无法打开视频文件 {video_path}")
        return 0

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(
        f"[抽帧-心跳-传统] 视频信息: {duration:.1f}秒, {total_frames}帧, {fps:.2f}FPS"
    )

    # 设置关键帧间隔（每5秒提取一帧）
    frame_interval = int(fps * 5)
    keyframe_count = 0

    # 发送开始心跳
    heartbeat_result["last_heartbeat"] = time.time()
    heartbeat_result["heartbeat_count"] += 1

    for frame_index in range(0, total_frames, frame_interval):
        # 每处理10帧发送一次心跳
        if frame_index % (frame_interval * 10) == 0:
            heartbeat_result["last_heartbeat"] = time.time()
            heartbeat_result["heartbeat_count"] += 1
            print(
                f"[抽帧-心跳-传统] 处理进度: {frame_index}/{total_frames}，心跳计数: {heartbeat_result['heartbeat_count']}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if ret:
            output_file = os.path.join(output_dir, f"{keyframe_count:04d}.png")
            # 使用 cv2.imencode + 手动写入来支持中文路径
            try:
                success, encoded_img = cv2.imencode(".png", frame)
                if success:
                    with open(output_file, "wb") as f:
                        f.write(encoded_img.tobytes())
                    keyframe_count += 1
            except Exception as e:
                print(f"[抽帧-心跳-传统] 保存失败: {str(e)}")

            if keyframe_count % 10 == 0:
                print(f"[抽帧-心跳-传统] ⏳ 已提取: {keyframe_count}")

    cap.release()

    # 发送完成心跳
    heartbeat_result["last_heartbeat"] = time.time()
    heartbeat_result["heartbeat_count"] += 1

    print(f"[抽帧-心跳-传统] ✅ 传统方法提取完成! 共提取 {keyframe_count} 个关键帧")
    return keyframe_count


# --- FunASR自动语音识别模块 ---
def setup_funasr_model():
    """初始化FunASR模型（模型应该已经由run.py下载）"""
    try:
        # 如果模型已经初始化，直接返回
        if st.session_state.funasr_model is not None:
            return True

        # 使用环境变量中的缓存目录（由run.py设置）
        if "MODELSCOPE_CACHE_DIR" not in os.environ:
            st.error("缓存目录环境变量未设置，请通过 run.py 启动程序")
            return False

        # 初始化模型（不指定版本，使用已下载的版本）
        st.info("正在初始化FunASR模型...")
        with st.spinner("初始化FunASR模型中..."):
            model = AutoModel(
                model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                disable_update=True,  # 禁止更新
                disable_log=False,
            )

        st.session_state.funasr_model = model
        st.success("✅ FunASR模型初始化完成!")
        return True

    except Exception as e:
        st.error(f"❌ FunASR模型初始化失败: {e}")
        import traceback

        st.error(f"详细错误: {traceback.format_exc()}")
        return False


def extract_audio_from_video(video_path):
    """从视频中提取音频"""
    audio_path = video_path.replace(os.path.splitext(video_path)[1], ".wav")

    try:
        # 使用session_state中保存的FFmpeg路径，优先使用项目内置的FFmpeg
        ffmpeg_cmd = st.session_state.get("ffmpeg_path", "ffmpeg")
        command = [
            ffmpeg_cmd,
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
            "-y",  # 覆盖已存在的文件
        ]

        subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return audio_path

    except Exception as e:
        st.error(f"音频提取失败: {e}")
        return None


def transcribe_audio(audio_path):
    """使用FunASR进行语音转录"""
    if not st.session_state.funasr_model:
        st.error("FunASR模型未初始化")
        return None

    try:
        st.info("正在进行语音转录...")
        with st.spinner("语音转录中..."):
            # 使用FunASR进行语音识别
            result = st.session_state.funasr_model.generate(
                input=audio_path, cache={}, language="auto", use_itn=True
            )

        # 提取转录文本
        if result and len(result) > 0:
            transcription = result[0]["text"]
            st.success("语音转录完成!")
            return transcription
        else:
            st.warning("未检测到语音内容")
            return None

    except Exception as e:
        st.error(f"语音转录失败: {e}")
        return None


def run_asr_analysis(video_path):
    """运行完整的ASR分析流程"""
    try:
        # 检查FunASR模型是否已初始化（应该在主线程中初始化）
        if not st.session_state.funasr_model:
            # 如果未初始化，尝试初始化（但这不应该在线程中发生）
            if not setup_funasr_model():
                return None

        # 提取音频
        audio_path = extract_audio_from_video(video_path)
        if not audio_path:
            return None

        # 转录音频
        transcription = transcribe_audio(audio_path)

        # 清理临时音频文件
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as cleanup_error:
            print(f"清理音频文件失败: {cleanup_error}")

        return transcription

    except Exception as e:
        print(f"ASR分析失败: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- 图片处理模块 ---
def read_image_as_base64(image_path):
    """读取图片文件并转换为Base64编码字符串"""
    try:
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
            base64_bytes = base64.b64encode(image_data)
            return base64_bytes.decode("utf-8").strip()
    except Exception as e:
        st.error(f"读取或编码图片 {image_path} 时出错: {e}")
        return None


def get_sorted_image_files(directory=DEFAULT_KEYFRAME_DIR):
    """获取目录中按数字序号排序的图片文件列表"""
    image_files = list_image_files(directory)

    filtered_files = []
    for file in image_files:
        try:
            filename = os.path.basename(file)
            num = int(os.path.splitext(filename)[0])
            filtered_files.append((num, file))
        except ValueError:
            continue

    sorted_files = sorted(filtered_files, key=lambda x: x[0])
    return [file for _, file in sorted_files]


def get_mime_type(filename):
    """根据文件扩展名返回对应的MIME类型"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        return "application/octet-stream"


# --- 多模态分析与并发控制模块 ---
class RateLimiter:
    """速率限制器，确保不超过60 QPM"""

    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def acquire(self):
        """获取请求许可"""
        now = time.time()
        # 清理超过1分钟的请求记录
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        # 检查是否超过限制
        if len(self.requests) >= self.max_requests:
            # 等待直到有可用的请求配额
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # 重新清理并检查
                self.requests = [
                    req_time
                    for req_time in self.requests
                    if now + wait_time - req_time < 60
                ]

        # 记录当前请求
        self.requests.append(time.time())


async def process_single_image(
    image_file,
    output_file,
    cache_dir,
    max_retries=3,
    user_custom_prompt="",
    use_ollama=False,
    selected_model="",
    client=None,
    vlm_model="",
    rate_limiter=None,
    skip_cache=False,
):
    """异步处理单张图片

    Args:
        image_file: 图片文件路径
        output_file: 输出文件路径
        cache_dir: 缓存目录
        max_retries: 最大重试次数
        user_custom_prompt: 用户自定义提示词
        use_ollama: 是否使用Ollama
        selected_model: 选择的模型名称
        client: API客户端
        vlm_model: 视觉语言模型名称
        rate_limiter: 共享的速率限制器实例
        skip_cache: 是否跳过缓存，强制重新解析
    """
    filename = os.path.basename(image_file)
    cache_file = os.path.join(cache_dir, f"{os.path.splitext(filename)[0]}.txt")

    # 检查缓存（如果不跳过缓存）
    if not skip_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_result = f.read()
            if (
                cached_result
                and cached_result.strip()
                and not cached_result.startswith("处理失败")
            ):
                return {
                    "filename": filename,
                    "result": cached_result,
                    "cached": True,
                    "success": True,
                    "processing_time": 0,
                }
        except:
            pass

    # 构建提示词
    base_prompt = "请详细解析这张图片的内容，用连贯详细的语言描述其主要内容。如果图片中出现文字，请准确描述它。如果文字模糊不清，请说明并尝试推断其大意。"
    full_prompt = base_prompt + (f" {user_custom_prompt}" if user_custom_prompt else "")

    success = False
    result = ""
    processing_time = 0

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # 应用速率限制（使用共享的rate_limiter实例）
            if rate_limiter:
                await rate_limiter.acquire()

            if use_ollama:
                # 使用 Ollama 处理（注意：ollama是同步的，在异步中调用）
                # ⚠️ 重要：每次创建全新的messages列表，确保不使用历史上下文
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: Client(host="http://localhost:11434").chat(
                        model=selected_model,
                        messages=[
                            {
                                "role": "user",
                                "content": full_prompt,
                                "images": [image_file],
                            }
                        ],
                        options={
                            "num_ctx": 4096,  # 明确设置上下文窗口大小
                        },
                    ),
                )
                result = response["message"]["content"]
            else:
                # 使用阿里云 DashScope 处理
                mime_type = get_mime_type(image_file)
                base64_image = read_image_as_base64(image_file)
                if base64_image is None:
                    raise Exception(f"无法读取或编码图片 {image_file}")

                image_url = f"data:{mime_type};base64,{base64_image}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": full_prompt},
                        ],
                    }
                ]

                # 使用 aiohttp 进行异步HTTP请求（如果可能），否则用executor
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=vlm_model, messages=messages, stream=False
                    ),
                )
                result = completion.choices[0].message.content

                # 注意：不在此处更新Token统计，因为异步线程无法访问 st.session_state
                # Token用量仅使用公式预估，不实时修改

            processing_time = time.time() - start_time
            success = True
            break

        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2**attempt, 30))
            else:
                result = f"处理失败: {str(e)}"
                processing_time = time.time() - start_time

    # 保存到缓存（确保目录存在）
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(result)

    # 保存到结果文件
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"图片: {filename}\n")
        if success:
            model_name = selected_model if use_ollama else vlm_model
            f.write(f"使用模型: {model_name}\n")
            f.write(f"处理时间: {processing_time:.1f}秒\n")
            f.write("解析结果:\n")
            f.write(result.strip() + "\n")
        else:
            model_name = selected_model if use_ollama else vlm_model
            f.write(f"❌ 图片 {filename} 处理失败 (使用模型: {model_name})\n")
        f.write("-" * 80 + "\n")

    return {
        "filename": filename,
        "result": result,
        "cached": False,
        "success": success,
        "processing_time": processing_time,
    }


async def process_images_concurrently(image_files, output_file, cache_dir):
    """并发处理多张图片"""
    tasks = []
    for image_file in image_files:
        task = process_single_image(image_file, output_file, cache_dir)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def process_single_image_with_display(
    image_file, output_file, cache_dir, container
):
    """处理单张图片并在指定容器中实时展示结果"""
    result = await process_single_image(image_file, output_file, cache_dir)

    # 在容器中实时展示结果
    with container:
        # 显示图片
        try:
            img = Image.open(image_file)
            st.image(
                img, caption=f"关键帧: {result['filename']}", use_container_width=True
            )
        except:
            st.warning(f"无法显示图片: {result['filename']}")

        # 显示分析结果
        if result["success"]:
            st.success(
                f"✅ {result['filename']} 分析完成 ({result['processing_time']:.1f}秒)"
            )
            st.markdown(f"**分析结果:**\n{result['result']}")
        else:
            st.error(f"❌ {result['filename']} 分析失败")

        st.divider()

    return result


def process_image_group(group, output_file, max_retries=3):
    """处理单张/多张图片并保存分析结果"""
    image_file = group[0]
    filename = os.path.basename(image_file)

    # 显示图片
    try:
        img = Image.open(image_file)
        st.image(img, caption=f"关键帧: {filename}", use_container_width=True)
    except:
        st.warning(f"无法显示图片: {filename}")

    # 构建提示词
    base_prompt = "请详细解析这张图片的内容，用连贯详细的语言描述其主要内容。如果图片中出现文字，请准确描述它。如果文字模糊不清，请说明并尝试推断其大意。"
    user_prompt = st.session_state.user_custom_prompt
    full_prompt = base_prompt + (f" {user_prompt}" if user_prompt else "")

    status = st.empty()
    status.info(f"分析中: {filename}...")

    success = False
    result = ""
    processing_time = 0

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            if st.session_state.use_ollama:
                # 使用 Ollama 处理
                # ⚠️ 重要：每次创建全新的messages列表，确保不使用历史上下文
                client = Client(host="http://localhost:11434")
                response = client.chat(
                    model=st.session_state.selected_model,
                    messages=[
                        {"role": "user", "content": full_prompt, "images": [image_file]}
                    ],
                    options={
                        "num_ctx": 4096,  # 明确设置上下文窗口大小，每次独立
                    },
                )
                result = response["message"]["content"]
            else:
                # 使用阿里云 DashScope 处理
                mime_type = get_mime_type(image_file)
                base64_image = read_image_as_base64(image_file)
                if base64_image is None:
                    raise Exception(f"无法读取或编码图片 {image_file}")

                image_url = f"data:{mime_type};base64,{base64_image}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": full_prompt},
                        ],
                    }
                ]
                completion = st.session_state.client.chat.completions.create(
                    model=st.session_state.vlm_model,  # 使用视觉模型
                    messages=messages,
                    stream=False,
                )
                result = completion.choices[0].message.content

                # Token用量仅使用公式预估，不实时修改

            processing_time = time.time() - start_time
            success = True
            break

        except Exception as e:
            status.warning(f"处理错误 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = min(2**attempt, 30)
                time.sleep(sleep_time)
            else:
                status.error(f"处理失败，已达到最大重试次数 ({max_retries})")

    # 保存到缓存目录（确保目录存在）
    os.makedirs(st.session_state.cache_dir, exist_ok=True)
    cache_file = os.path.join(
        st.session_state.cache_dir, f"{os.path.splitext(filename)[0]}.txt"
    )
    with open(cache_file, "w", encoding="utf-8") as f:
        if success:
            f.write(result.strip())
        else:
            f.write("处理失败")

    # 保存结果到文件
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"图片: {filename}\n")
        if success:
            f.write(f"使用模型: {st.session_state.selected_model}\n")
            f.write(f"处理时间: {processing_time:.1f}秒\n")
            f.write("解析结果:\n")
            f.write(result.strip() + "\n")
        else:
            f.write(
                f"❌ 图片 {filename} 处理失败 (使用模型: {st.session_state.selected_model})\n"
            )
        f.write("-" * 80 + "\n")

    if success:
        status.success(f"分析完成 ({processing_time:.1f}秒)")
        st.markdown(f"**分析结果:**\n{result}")
        return True
    else:
        status.error(f"最终处理失败: {filename}")
        return False


# --- Embedding模型管理模块 ---
def find_embedding_model():
    """查找已下载的Qwen3 Embedding模型"""
    cache_dir = get_program_cache_dir()
    modelscope_cache = os.path.join(cache_dir, "modelscope", "hub")

    # 检查Qwen3-Embedding-0.6B模型
    # Windows系统会将点号替换为下划线，需要兼容两种命名方式
    model_name_original = "Qwen/Qwen3-Embedding-0.6B"
    model_name_windows = "Qwen/Qwen3-Embedding-0___6B"  # Windows兼容版本

    # 优先检查Windows兼容版本
    model_path = os.path.join(modelscope_cache, model_name_windows)
    if os.path.exists(model_path):
        st.info(f"✅ 找到Qwen3 Embedding模型: {model_path}")
        return model_path

    # 回退检查原始版本
    model_path = os.path.join(modelscope_cache, model_name_original)
    if os.path.exists(model_path):
        st.info(f"✅ 找到Qwen3 Embedding模型: {model_path}")
        return model_path

    st.warning("⚠️ 未找到Qwen3 Embedding模型")
    st.info("将使用备用的英文模型（效果可能略差）")
    return None


def load_embedding_model():
    """加载Embedding模型（优先中文，回退英文）"""
    try:
        # 查找已下载的中文模型
        model_path = find_embedding_model()

        if model_path:
            # 使用ModelScope下载的中文模型
            st.info("正在加载中文Embedding模型...")
            with st.spinner("加载中..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": "cpu"},  # 使用CPU，避免GPU冲突
                )
            st.success("✅ 中文Embedding模型加载成功！")
            return embeddings
        else:
            # 回退到英文模型
            st.warning("使用备用英文Embedding模型")
            with st.spinner("加载备用模型..."):
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )
            st.info("✅ 备用Embedding模型加载成功")
            return embeddings

    except Exception as e:
        st.error(f"❌ Embedding模型加载失败: {e}")
        import traceback

        st.error(f"详细错误: {traceback.format_exc()}")

        # 最后的回退方案
        try:
            st.warning("尝试使用最简单的备用模型...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            st.info("✅ 最终备用模型加载成功")
            return embeddings
        except Exception as e2:
            st.error(f"❌ 所有Embedding模型都加载失败: {e2}")
            return None


# --- RAG检索增强生成模块 ---
def force_close_chroma_db(vector_store):
    """强制关闭ChromaDB连接并释放文件句柄

    Args:
        vector_store: ChromaDB向量存储对象
    """
    if not vector_store:
        return

    try:
        # 1. 尝试关闭集合连接
        if hasattr(vector_store, "_collection"):
            try:
                # ChromaDB的collection可能有close方法
                if hasattr(vector_store._collection, "close"):
                    vector_store._collection.close()
            except:
                pass

        # 2. 尝试关闭客户端连接
        if hasattr(vector_store, "_client"):
            try:
                # ChromaDB的client可能有close或reset方法
                if hasattr(vector_store._client, "close"):
                    vector_store._client.close()
                elif hasattr(vector_store._client, "reset"):
                    vector_store._client.reset()
            except:
                pass

        # 3. 尝试删除持久化客户端的引用
        if hasattr(vector_store, "_persist_directory"):
            try:
                delattr(vector_store, "_persist_directory")
            except:
                pass
    except Exception as e:
        print(f"关闭ChromaDB时出错: {e}")


def safe_remove_chroma_db(chroma_db_path, max_retries=5):
    """安全删除ChromaDB文件夹

    Args:
        chroma_db_path: ChromaDB路径
        max_retries: 最大重试次数

    Returns:
        bool: 是否成功删除
    """
    import gc
    import time

    if not os.path.exists(chroma_db_path):
        return True

    st.info(f"正在清理旧的向量数据库: {chroma_db_path}")

    # 先尝试关闭当前的vector_store
    if st.session_state.vector_store:
        try:
            force_close_chroma_db(st.session_state.vector_store)
            st.session_state.vector_store = None
        except Exception as e:
            print(f"关闭vector_store时出错: {e}")

    # 强制垃圾回收多次
    for _ in range(3):
        gc.collect()
        time.sleep(0.3)

    # 尝试删除
    for attempt in range(max_retries):
        try:
            # Windows特定：尝试重命名文件夹（有时比直接删除更可靠）
            if sys.platform == "win32" and attempt > 0:
                try:
                    temp_path = chroma_db_path + f"_deleting_{int(time.time())}"
                    os.rename(chroma_db_path, temp_path)
                    chroma_db_path = temp_path
                    st.info(f"已重命名数据库文件夹（尝试 {attempt + 1}/{max_retries}）")
                except:
                    pass

            # 尝试删除
            shutil.rmtree(chroma_db_path, ignore_errors=False)
            st.success("✅ 已清除旧的向量数据库")
            return True

        except PermissionError as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间：2秒, 4秒, 6秒...
                st.warning(
                    f"⚠️ 数据库文件被占用，等待{wait_time}秒后重试... ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)

                # 每次重试前再次强制垃圾回收
                for _ in range(3):
                    gc.collect()
                    time.sleep(0.2)
            else:
                st.error(f"❌ 无法删除旧数据库（已重试{max_retries}次）")
                st.error(f"文件夹路径: {chroma_db_path}")
                st.error("**可能的解决方案：**")
                st.markdown("""
                1. 关闭所有可能占用数据库的程序（包括之前的程序实例）
                2. 重启程序后再试
                3. 重启计算机
                4. 手动删除该文件夹
                5. 使用任务管理器结束相关Python进程
                """)
                return False

        except Exception as e:
            st.error(f"❌ 删除数据库时出错: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                st.warning(f"等待后重试... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                return False

    return False


def build_vector_store(cache_dir, force_rebuild=False):
    """构建向量数据库（支持持久化和增量更新）"""
    try:
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 向量数据库路径
        chroma_db_path = st.session_state.vector_store_path

        # 加载或创建嵌入模型
        if st.session_state.embedding_model is None:
            st.info("首次使用，正在初始化Embedding模型...")
            st.session_state.embedding_model = load_embedding_model()

            if st.session_state.embedding_model is None:
                st.error("Embedding模型加载失败，无法构建向量数据库")
                return None

        embeddings = st.session_state.embedding_model

        # 检查是否已存在向量数据库且不强制重建
        if os.path.exists(chroma_db_path) and not force_rebuild:
            try:
                st.info("加载已存在的向量数据库...")
                vector_store = Chroma(
                    persist_directory=chroma_db_path, embedding_function=embeddings
                )

                # 验证数据库是否有效
                collection = vector_store._collection
                if collection.count() > 0:
                    st.success(
                        f"成功加载向量数据库，包含 {collection.count()} 个文档片段"
                    )
                    return vector_store
                else:
                    st.warning("向量数据库为空，将重新构建")
            except Exception as e:
                st.warning(f"加载向量数据库失败: {e}，将重新构建")

        # 获取所有缓存文件
        cache_files = glob.glob(os.path.join(cache_dir, "*.txt"))
        if not cache_files:
            st.warning("没有找到缓存文件，请先完成视频分析")
            return None

        # 读取所有文档并添加丰富的元数据
        documents = []
        for cache_file in sorted(cache_files):
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if (
                    content
                    and content != "处理失败"
                    and not content.startswith("处理失败:")
                ):
                    # 从文件名提取信息
                    filename = os.path.basename(cache_file)
                    frame_num = os.path.splitext(filename)[0]

                    # 判断是ASR内容还是关键帧内容
                    if filename == "asr_transcription.txt":
                        # ASR语音转录内容
                        metadata = {
                            "frame": "ASR",
                            "frame_number": -1,  # 特殊标记
                            "source": cache_file,
                            "content_type": "audio",  # 音频内容
                            "char_count": len(content),
                            "timestamp": os.path.getmtime(cache_file),
                        }
                    else:
                        # 关键帧视觉内容
                        metadata = {
                            "frame": frame_num,
                            "frame_number": int(frame_num),  # 用于排序
                            "source": cache_file,
                            "content_type": "visual",  # 视觉内容
                            "char_count": len(content),
                            "timestamp": os.path.getmtime(cache_file),
                        }

                    documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            st.warning("没有有效的文档内容")
            return None

        st.info(f"准备处理 {len(documents)} 个文档...")

        # 优化的文本分割策略
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 增加块大小以保留更多上下文
            chunk_overlap=100,  # 增加重叠以保持连贯性
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        split_documents = text_splitter.split_documents(documents)

        st.info(f"文档分块完成，共 {len(split_documents)} 个片段")

        # 删除旧的数据库（如果强制重建）
        if force_rebuild and os.path.exists(chroma_db_path):
            # 使用增强的安全删除函数
            if not safe_remove_chroma_db(chroma_db_path, max_retries=5):
                st.error("⚠️ 无法删除旧数据库，将尝试继续...")
                st.warning("如果后续构建失败，请手动删除数据库文件夹或重启程序")
                # 不直接返回None，尝试继续（可能会失败，但给用户一个机会）

        # 确保向量数据库目录的父目录存在
        parent_dir = os.path.dirname(chroma_db_path)
        if parent_dir:  # 如果有父目录
            os.makedirs(parent_dir, exist_ok=True)

        # 创建向量数据库（持久化）
        with st.spinner("正在构建向量索引..."):
            vector_store = Chroma.from_documents(
                documents=split_documents,
                embedding=embeddings,
                persist_directory=chroma_db_path,
            )

        st.success(f"✅ 向量数据库构建完成，包含 {len(split_documents)} 个文档片段")
        st.info(f"📁 数据库已保存至: {chroma_db_path}")

        return vector_store

    except Exception as e:
        st.error(f"构建向量数据库失败: {e}")
        import traceback

        st.error(f"详细错误信息: {traceback.format_exc()}")
        return None


def search_vector_store(
    query, vector_store, k=5, search_type="similarity", score_threshold=0.5, fetch_k=20
):
    """在向量数据库中搜索相关内容（支持多种检索策略）

    参数:
        query: 查询字符串
        vector_store: 向量数据库
        k: 返回结果数量
        search_type: 检索类型 ("similarity" | "mmr" | "similarity_score")
        score_threshold: 相似度阈值（仅用于similarity_score）
        fetch_k: MMR检索时的候选数量
    """
    if not vector_store:
        st.error("向量数据库未初始化")
        return []

    try:
        if search_type == "mmr":
            # 最大边际相关性搜索（减少冗余）
            results = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
        elif search_type == "similarity_score":
            # 带相似度分数的搜索
            results_with_scores = vector_store.similarity_search_with_score(query, k=k)
            # 过滤低于阈值的结果
            results = [
                doc for doc, score in results_with_scores if score <= score_threshold
            ]
            if not results:
                st.warning(f"未找到相似度高于阈值 {score_threshold} 的结果")
        else:
            # 默认相似度搜索
            results = vector_store.similarity_search(query, k=k)

        # 按帧号排序结果（如果有frame_number元数据）
        try:
            results = sorted(results, key=lambda x: x.metadata.get("frame_number", 0))
        except:
            pass

        return results
    except Exception as e:
        st.error(f"搜索失败: {e}")
        import traceback

        st.error(f"详细错误: {traceback.format_exc()}")
        return []


def generate_rag_response(query, search_results, include_citations=True):
    """基于检索结果生成回答（使用多方式分析页面的独立模型配置）"""
    if not search_results:
        return "未找到相关信息", []

    # 构建带编号的上下文（用于引用）
    context_parts = []
    frame_references = []

    for i, doc in enumerate(search_results, 1):
        frame_num = doc.metadata.get("frame", "未知")
        frame_references.append(frame_num)
        context_parts.append(f"[片段{i}] (关键帧 {frame_num}):\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    # 优化的提示词
    prompt = f"""你是一个专业的视频内容分析助手。请基于以下视频关键帧的分析内容，准确、详细地回答用户的问题。

用户问题：
{query}

相关视频内容：
{context}

回答要求：
1. 仅基于提供的视频内容回答，不要添加未提及的信息
2. 如果信息不足，请明确说明
3. 用清晰、连贯的语言组织答案
4. 如果引用特定片段，请注明片段编号（如 [片段1]）
5. 尽可能提供具体细节

请给出您的回答："""

    try:
        response_text = ""

        # 使用多方式分析页面的独立配置
        if st.session_state.llm_use_ollama:
            # 使用Ollama
            if not st.session_state.llm_ollama_model:
                return "错误：未选择Ollama模型", []

            # ⚠️ 重要：每次创建全新的messages列表，确保不使用历史上下文
            client = Client(host="http://localhost:11434")
            response = client.chat(
                model=st.session_state.llm_ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 4096,  # 明确设置上下文窗口，每次独立
                },
            )
            response_text = response["message"]["content"]
        else:
            # 使用阿里云DashScope的LLM模型
            if not st.session_state.client:
                # 尝试初始化
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    st.session_state.client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                else:
                    return "错误：阿里云客户端未初始化，请检查API密钥", []

            messages = [{"role": "user", "content": prompt}]
            completion = st.session_state.client.chat.completions.create(
                model=st.session_state.llm_model,  # 使用LLM模型
                messages=messages,
                stream=False,
            )
            response_text = completion.choices[0].message.content

            # Token用量仅使用公式预估，不实时修改

        # 添加引用信息
        if include_citations:
            citations = "\n\n---\n**引用的关键帧:**\n"
            for i, frame_num in enumerate(frame_references, 1):
                citations += f"- [片段{i}] 关键帧 {frame_num}\n"
            response_text += citations

        return response_text, frame_references

    except Exception as e:
        error_msg = f"生成回答失败: {str(e)}"
        import traceback

        error_detail = traceback.format_exc()
        st.error(f"详细错误: {error_detail}")
        return f"{error_msg}\n\n详细错误:\n{error_detail}", []


# --- 成文与速读功能模块 ---
def generate_comprehensive_report(cache_dir):
    """生成完整的视频分析报告（使用多方式分析页面的独立模型配置）"""
    try:
        # 获取所有缓存文件
        cache_files = glob.glob(os.path.join(cache_dir, "*.txt"))
        if not cache_files:
            return "没有找到分析结果"

        # 分别读取视觉内容和语音内容
        visual_texts = []
        audio_text = None

        for cache_file in sorted(cache_files):
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if (
                    content
                    and content != "处理失败"
                    and not content.startswith("处理失败:")
                ):
                    filename = os.path.basename(cache_file)

                    if filename == "asr_transcription.txt":
                        # 语音转录内容
                        audio_text = content
                    else:
                        # 关键帧内容
                        frame_num = os.path.splitext(filename)[0]
                        visual_texts.append(f"帧 {frame_num}: {content}")

        if not visual_texts and not audio_text:
            return "没有有效的分析内容"

        # 构建组合内容
        combined_parts = []

        if visual_texts:
            combined_parts.append("【视觉内容分析】\n" + "\n\n".join(visual_texts))

        if audio_text:
            combined_parts.append(f"【语音转录内容】\n{audio_text}")

        combined_text = "\n\n" + "=" * 50 + "\n\n".join(combined_parts)

        # 构建成文提示词
        prompt = f"""你是一个专业的视频内容分析助手。请将以下关于一个视频的分析内容（包括关键帧视觉分析和语音转录）整合成一份连贯、详细、口语化的中文总结报告。

要求：
1. 综合视觉内容和语音内容，形成完整的视频理解
2. 内容流畅，逻辑清晰
3. 涵盖所有重要场景、物体、动作、文字和语音信息
4. 如果视觉和语音内容相互补充，请自然地融合它们

视频分析内容如下：

{combined_text}"""

        # 使用多方式分析页面的独立配置
        if st.session_state.llm_use_ollama:
            # 使用Ollama
            if not st.session_state.llm_ollama_model:
                return "错误：未选择Ollama模型，请在左侧边栏配置"

            # ⚠️ 重要：每次创建全新的messages列表，确保不使用历史上下文
            client = Client(host="http://localhost:11434")
            response = client.chat(
                model=st.session_state.llm_ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 4096,  # 明确设置上下文窗口，每次独立
                },
            )
            return response["message"]["content"]
        else:
            # 使用阿里云DashScope的LLM模型
            if not st.session_state.client:
                # 尝试初始化
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    st.session_state.client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                else:
                    return "错误：阿里云客户端未初始化，请检查API密钥或在左侧边栏配置"

            messages = [{"role": "user", "content": prompt}]
            completion = st.session_state.client.chat.completions.create(
                model=st.session_state.llm_model,  # 使用LLM模型
                messages=messages,
                stream=False,
            )

            # Token用量仅使用公式预估，不实时修改

            return completion.choices[0].message.content

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        return f"生成报告失败: {str(e)}\n\n详细错误:\n{error_detail}"


def generate_quick_summary(cache_dir):
    """生成快速摘要（使用多方式分析页面的独立模型配置）"""
    try:
        # 获取所有缓存文件
        cache_files = glob.glob(os.path.join(cache_dir, "*.txt"))
        if not cache_files:
            return "没有找到分析结果"

        # 分别读取视觉内容和语音内容
        visual_texts = []
        audio_text = None

        for cache_file in sorted(cache_files):
            with open(cache_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if (
                    content
                    and content != "处理失败"
                    and not content.startswith("处理失败:")
                ):
                    filename = os.path.basename(cache_file)

                    if filename == "asr_transcription.txt":
                        # 语音转录内容
                        audio_text = content
                    else:
                        # 关键帧内容
                        visual_texts.append(content)

        if not visual_texts and not audio_text:
            return "没有有效的分析内容"

        # 构建组合内容
        combined_parts = []

        if visual_texts:
            combined_parts.append("视觉内容：" + "\n".join(visual_texts))

        if audio_text:
            combined_parts.append(f"语音内容：{audio_text}")

        combined_text = "\n\n".join(combined_parts)

        # 构建速读提示词
        prompt = f"""你是一个专业的摘要助手。请基于以下视频内容分析（包括视觉和语音），提取最核心的信息，用极其精简的几句话（不超过150字）概括整个视频的主要内容。

要求：综合视觉和语音信息，直接给出摘要，不需要开场白。

视频分析内容：

{combined_text}"""

        # 使用多方式分析页面的独立配置
        if st.session_state.llm_use_ollama:
            # 使用Ollama
            if not st.session_state.llm_ollama_model:
                return "错误：未选择Ollama模型，请在左侧边栏配置"

            # ⚠️ 重要：每次创建全新的messages列表，确保不使用历史上下文
            client = Client(host="http://localhost:11434")
            response = client.chat(
                model=st.session_state.llm_ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_ctx": 8192,  # 明确设置上下文窗口，每次独立
                },
            )
            return response["message"]["content"]
        else:
            # 使用阿里云DashScope的LLM模型
            if not st.session_state.client:
                # 尝试初始化
                api_key = os.environ.get("DASHSCOPE_API_KEY")
                if api_key:
                    st.session_state.client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                else:
                    return "错误：阿里云客户端未初始化，请检查API密钥或在左侧边栏配置"

            messages = [{"role": "user", "content": prompt}]
            completion = st.session_state.client.chat.completions.create(
                model=st.session_state.llm_model,  # 使用LLM模型
                messages=messages,
                stream=False,
            )

            # Token用量仅使用公式预估，不实时修改

            return completion.choices[0].message.content

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        return f"生成摘要失败: {str(e)}\n\n详细错误:\n{error_detail}"


# --- 停止Ollama模型函数 ---
def stop_ollama_model():
    """停止Ollama模型以释放资源"""
    if not st.session_state.selected_model:
        st.error("未选择任何Ollama模型")
        return False

    try:
        # 构造停止命令
        command = ["ollama", "stop", st.session_state.selected_model]

        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 检查执行结果
        if result.returncode == 0:
            st.success(f"已停止模型: {st.session_state.selected_model}")
            st.session_state.show_stop_button = False  # 隐藏停止按钮
            return True
        else:
            st.error(f"停止模型失败: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        st.error(f"停止模型时出错: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"发生未知错误: {str(e)}")
        return False


# --- 主分析函数 ---
def run_video_analysis():
    """视频分析主程序"""
    st.session_state.processing = True

    try:
        # 重置Token使用统计（每次分析重新开始计算）
        if not st.session_state.use_ollama:
            reset_token_usage()
            st.info("💰 用量统计已启用，将在分析过程中实时显示Token使用情况")

        # 检查FFmpeg
        ffmpeg_available, ffmpeg_path = check_ffmpeg()
        if not ffmpeg_available:
            st.error("FFmpeg未安装或未找到，请确保FFmpeg已安装并添加到系统PATH")
            return

        # 保存FFmpeg路径到session_state，供后续使用
        st.session_state.ffmpeg_path = ffmpeg_path

        # 初始化模型
        if not init_model():
            st.error("模型初始化失败，请检查配置")
            return

        # 先重置为默认关键帧目录，避免上一次外部目录残留到当前任务。
        st.session_state.keyframe_dir = DEFAULT_KEYFRAME_DIR

        # 如果使用已有关键帧，则统一走规范化和目录校验逻辑。
        if (
            st.session_state.use_existing_keyframes
            and st.session_state.existing_keyframes_path
        ):
            normalized_keyframe_dir, image_files, keyframe_error = (
                resolve_keyframe_directory(st.session_state.existing_keyframes_path)
            )
            if keyframe_error:
                st.error(f"关键帧路径无效，无法继续分析: {keyframe_error}")
                st.session_state.processing = False
                return

            st.session_state.existing_keyframes_path = normalized_keyframe_dir
            st.session_state.keyframe_dir = normalized_keyframe_dir
            st.info(
                f"📁 使用已有关键帧文件夹: {normalized_keyframe_dir} ({len(image_files)} 个文件)"
            )

            if st.session_state.force_reparse_keyframes:
                st.warning(
                    "🔄 已启用强制重新解析，将忽略所有缓存结果，重新分析所有关键帧"
                )

        # 在关键帧目录确定后再创建本次运行所需目录。
        os.makedirs(st.session_state.keyframe_dir, exist_ok=True)
        os.makedirs(st.session_state.cache_dir, exist_ok=True)

        # 如果使用已有向量数据库，更新向量数据库路径
        if (
            st.session_state.use_existing_vector_db
            and st.session_state.existing_vector_db_path
        ):
            if os.path.exists(
                st.session_state.existing_vector_db_path
            ) and os.path.isdir(st.session_state.existing_vector_db_path):
                st.session_state.vector_store_path = (
                    st.session_state.existing_vector_db_path
                )
                st.info(
                    f"📚 使用已有向量数据库: {st.session_state.existing_vector_db_path}"
                )
            else:
                st.warning(
                    f"⚠️ 向量数据库路径无效: {st.session_state.existing_vector_db_path}，将尝试重新构建"
                )

        # 初始化结果文件
        with open(st.session_state.output_file, "w", encoding="utf-8") as f:
            f.write(f"视频分析报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if st.session_state.video_path:
                f.write(f"视频文件: {st.session_state.video_path}\n")
            else:
                f.write(f"使用已有关键帧: {st.session_state.existing_keyframes_path}\n")
            f.write(
                f"使用的模型后端: {'Ollama' if st.session_state.use_ollama else '阿里云 DashScope'}\n"
            )
            if st.session_state.use_ollama:
                f.write(f"使用的模型: {st.session_state.selected_model}\n")
            else:
                f.write(f"图片分析模型 (VLM): {st.session_state.vlm_model}\n")
                f.write(f"文本生成模型 (LLM): {st.session_state.llm_model}\n")
            f.write("=" * 80 + "\n\n")

        import asyncio
        import threading

        pipeline_start_time = time.time()

        skip_extraction = (
            st.session_state.use_existing_keyframes
            and st.session_state.existing_keyframes_path
        )
        has_video_for_asr = st.session_state.video_path and os.path.exists(
            st.session_state.video_path
        )

        asr_result = {"transcription": None, "error": None}
        extraction_result = {"count": 0, "finished": False, "error": None}
        vlm_state = {
            "results": {},
            "processed": 0,
            "discovered": 0,
            "finished": False,
            "error": None,
        }

        def asr_worker(video_path, model_obj):
            """语音识别工作线程函数"""
            if not model_obj:
                asr_result["error"] = "FunASR模型未初始化"
                return

            try:
                print("[ASR] 🎤 语音识别线程已启动...")
                # 提取音频
                audio_path = extract_audio_from_video(video_path)
                if not audio_path:
                    asr_result["error"] = "音频提取失败"
                    return

                # 转录音频（直接使用传入的模型对象，不访问session_state）
                result = model_obj.generate(
                    input=audio_path, cache={}, language="auto", use_itn=True
                )

                if result and len(result) > 0:
                    asr_result["transcription"] = result[0]["text"]
                    print("[ASR] ✅ 语音识别完成！")
                else:
                    asr_result["error"] = "未检测到语音内容"

                # 清理临时音频文件
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                except:
                    pass

            except Exception as e:
                asr_result["error"] = f"语音识别错误: {str(e)}"
                import traceback

                print(traceback.format_exc())

        def extraction_worker(video_path, output_dir):
            """关键帧提取工作线程函数 - 带心跳监控的版本"""
            try:
                print("[抽帧] 🎬 关键帧提取线程已启动...")

                # 初始化心跳监控
                extraction_result["last_heartbeat"] = time.time()
                extraction_result["heartbeat_count"] = 0

                # 使用带心跳监控的关键帧提取函数
                count = extract_keyframes_with_heartbeat(
                    video_path, output_dir, extraction_result
                )

                extraction_result["count"] = count
                extraction_result["finished"] = True
                extraction_result["heartbeat_count"] = extraction_result.get(
                    "heartbeat_count", 0
                )

                print(
                    f"[抽帧] ✅ 关键帧提取完成！共提取 {count} 个关键帧，心跳次数: {extraction_result['heartbeat_count']}"
                )

            except Exception as e:
                print(f"[抽帧] ❌ 关键帧提取失败: {str(e)}")
                extraction_result["count"] = 0
                extraction_result["finished"] = True
                extraction_result["error"] = str(e)

        # 启动并行线程
        st.markdown("---")
        st.markdown("### 🚀 并行处理优化")

        if skip_extraction and not has_video_for_asr:
            st.info("使用已有关键帧，无需视频文件")
        elif skip_extraction:
            st.info("同时启动：语音识别（可选） + VLM分析（使用已有关键帧）")
        else:
            st.info("同时启动：关键帧提取 + 语音识别（可选） + VLM分析")

        # 1) 先启动抽帧线程，避免ASR模型初始化阻塞整体流水线
        extraction_thread = None
        if not skip_extraction:
            extraction_thread = threading.Thread(
                target=extraction_worker,
                args=(st.session_state.video_path, st.session_state.keyframe_dir),
                daemon=True,
                name="Extraction-Worker",
            )
            extraction_thread.start()
            st.info("🎬 关键帧提取线程已启动")
        else:
            extraction_result["finished"] = True
            existing_frames = list_image_files(st.session_state.keyframe_dir)
            extraction_result["count"] = len(existing_frames)
            st.success(f"✅ 使用已有关键帧：共 {extraction_result['count']} 个")
        # 2) 再初始化ASR模型并启动ASR线程（仅在有视频文件时）
        funasr_model_ref = None
        if has_video_for_asr:
            st.info("正在初始化语音识别模型...")
            funasr_initialized = setup_funasr_model()
            funasr_model_ref = (
                st.session_state.funasr_model if funasr_initialized else None
            )
        else:
            st.info("ℹ️ 未提供视频文件，跳过语音识别")

        asr_thread = None
        if has_video_for_asr:
            asr_thread = threading.Thread(
                target=asr_worker,
                args=(st.session_state.video_path, funasr_model_ref),
                daemon=True,
                name="ASR-Worker",
            )
            asr_thread.start()
            st.info("🎤 语音识别线程已启动")

        st.subheader("实时分析结果")

        output_file_path = st.session_state.output_file
        cache_dir_path = st.session_state.cache_dir
        keyframe_dir_path = st.session_state.keyframe_dir
        user_custom_prompt = st.session_state.user_custom_prompt
        use_ollama = st.session_state.use_ollama
        selected_model = st.session_state.selected_model if use_ollama else ""
        client = st.session_state.client if not use_ollama else None
        vlm_model = st.session_state.vlm_model if not use_ollama else ""
        skip_cache = st.session_state.force_reparse_keyframes

        shared_rate_limiter = RateLimiter(max_requests_per_minute=60)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def run_vlm_pipeline():
            try:

                async def vlm_pipeline_async():
                    max_in_flight = 1 if use_ollama else 6
                    pending = {}
                    processed_filenames = set()
                    stable_rounds = 0
                    last_discovered_count = -1
                    scan_interval = 0.8

                    heartbeat_timeout = 900
                    max_total_time = 7200

                    while True:
                        discovered_images = get_sorted_image_files(keyframe_dir_path)
                        discovered_filenames = [
                            os.path.basename(p) for p in discovered_images
                        ]

                        vlm_state["discovered"] = len(discovered_images)

                        for image_path in discovered_images:
                            filename = os.path.basename(image_path)
                            if filename in processed_filenames:
                                continue
                            if any(
                                pending_filename == filename
                                for pending_filename in pending.values()
                            ):
                                continue
                            if len(pending) >= max_in_flight:
                                break

                            task = asyncio.create_task(
                                process_single_image(
                                    image_path,
                                    output_file_path,
                                    cache_dir_path,
                                    max_retries=3,
                                    user_custom_prompt=user_custom_prompt,
                                    use_ollama=use_ollama,
                                    selected_model=selected_model,
                                    client=client,
                                    vlm_model=vlm_model,
                                    rate_limiter=shared_rate_limiter,
                                    skip_cache=skip_cache,
                                )
                            )
                            pending[task] = filename

                        if pending:
                            done, _ = await asyncio.wait(
                                pending.keys(),
                                timeout=scan_interval,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            for task in done:
                                filename = pending.pop(task, None)
                                if not filename:
                                    continue
                                try:
                                    result = task.result()
                                except Exception as e:
                                    result = {
                                        "filename": filename,
                                        "result": f"处理失败: {str(e)}",
                                        "cached": False,
                                        "success": False,
                                    }
                                vlm_state["results"][filename] = result
                                processed_filenames.add(filename)
                                vlm_state["processed"] = len(processed_filenames)
                        else:
                            await asyncio.sleep(scan_interval)

                        extraction_done = skip_extraction or (
                            extraction_result.get("finished", False)
                            and (
                                not extraction_thread
                                or not extraction_thread.is_alive()
                            )
                        )

                        if (
                            not extraction_done
                            and extraction_thread
                            and extraction_thread.is_alive()
                        ):
                            last_heartbeat = extraction_result.get(
                                "last_heartbeat", pipeline_start_time
                            )
                            if (
                                time.time() - last_heartbeat > heartbeat_timeout
                                and len(discovered_images) == 0
                            ):
                                raise TimeoutError(
                                    f"关键帧提取长时间无进展（超过 {heartbeat_timeout} 秒）"
                                )

                        if time.time() - pipeline_start_time > max_total_time:
                            raise TimeoutError("整体处理超出最大允许时长")

                        if extraction_done and len(discovered_images) == 0:
                            if extraction_result.get("error"):
                                raise RuntimeError(
                                    f"关键帧提取失败: {extraction_result['error']}"
                                )
                            raise RuntimeError(
                                f"在目录 {keyframe_dir_path} 中未找到任何图片"
                            )

                        if extraction_done and not pending:
                            if len(discovered_images) == last_discovered_count and len(
                                processed_filenames
                            ) >= len(discovered_filenames):
                                stable_rounds += 1
                            else:
                                stable_rounds = 0
                            last_discovered_count = len(discovered_images)

                            if stable_rounds >= 3:
                                break
                        else:
                            stable_rounds = 0
                            last_discovered_count = len(discovered_images)

                    vlm_state["finished"] = True

                asyncio.run(vlm_pipeline_async())
            except Exception as e:
                vlm_state["error"] = str(e)
                vlm_state["finished"] = True

        vlm_thread = threading.Thread(
            target=run_vlm_pipeline, daemon=True, name="VLM-Pipeline"
        )
        vlm_thread.start()

        while vlm_thread.is_alive():
            vlm_thread.join(timeout=1.0)
            discovered = int(vlm_state.get("discovered", 0) or 0)
            processed = int(vlm_state.get("processed", 0) or 0)

            expected_total = discovered
            if extraction_result.get("finished") and extraction_result.get("count"):
                expected_total = max(expected_total, int(extraction_result["count"]))

            if expected_total <= 0:
                progress_bar.progress(0.0)
                status_text.text("等待关键帧生成中...")
            else:
                progress = processed / max(1, expected_total)
                progress_bar.progress(min(progress, 0.99))

                extraction_status = (
                    "已完成"
                    if (skip_extraction or extraction_result.get("finished"))
                    else "进行中"
                )
                status_text.text(
                    f"VLM分析中... 已完成: {processed}/{expected_total}，关键帧提取: {extraction_status}"
                )

        if vlm_state.get("error"):
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ 图片分析失败: {vlm_state['error']}")
            return

        progress_bar.progress(1.0)
        status_text.text("分析完成！")

        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        results = list(vlm_state["results"].values())

        # 按帧号排序结果并显示（图+文）
        st.subheader("📊 分析结果汇总（按帧号排序）")

        # 将结果转换为字典，按帧号排序
        result_dict = {}
        success_count = 0
        for result in results:
            # 跳过异常结果
            if isinstance(result, Exception):
                st.warning(f"⚠️ 处理过程中遇到异常: {str(result)}")
                continue

            if isinstance(result, dict):
                filename = result.get("filename", "")
                result_dict[filename] = result
                if result.get("success", False):
                    success_count += 1

        total_images = len(result_dict)

        # 按文件名（帧号）排序显示（图+文排布）
        sorted_filenames = sorted(result_dict.keys())
        for filename in sorted_filenames:
            result = result_dict[filename]

            # 显示图片
            frame_num = os.path.splitext(filename)[0]

            keyframe_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = os.path.join(
                    st.session_state.keyframe_dir, f"{frame_num}{ext}"
                )
                if os.path.exists(candidate):
                    keyframe_path = candidate
                    break

            try:
                if keyframe_path:
                    img = Image.open(keyframe_path)
                    st.image(
                        img, caption=f"关键帧: {filename}", use_container_width=True
                    )
                else:
                    st.warning(f"无法找到图片文件: {filename}")
            except:
                st.warning(f"无法显示图片: {filename}")

            # 显示分析结果
            if result.get("success"):
                status_indicator = "✅"
                if result.get("cached"):
                    st.success(f"{status_indicator} {filename} 分析完成 (使用缓存)")
                elif result.get("processing_time"):
                    st.success(
                        f"{status_indicator} {filename} 分析完成 ({result['processing_time']:.1f}秒)"
                    )
                else:
                    st.success(f"{status_indicator} {filename} 分析完成")

                # 从缓存文件读取完整的分析结果
                cache_file = os.path.join(cache_dir_path, f"{frame_num}.txt")
                analysis_text = result.get("result", "")

                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            analysis_text = f.read().strip()
                    except:
                        pass

                st.markdown(
                    f"**分析结果:**\n{analysis_text if analysis_text else '无结果'}"
                )
            else:
                st.error(f"❌ {filename} 分析失败")
                st.markdown(f"**错误信息:**\n{result.get('result', '无结果')}")

            st.divider()

        st.success(f"✅ 图片分析完成！成功: {success_count}/{total_images}")

        # 等待ASR线程完成（在显示结果前）
        st.markdown("---")
        st.subheader("🎤 语音识别结果")

        if asr_thread:
            st.info("⏳ 等待语音识别完成...")
            asr_thread.join(timeout=300)  # 最多等待5分钟

            if asr_thread.is_alive():
                st.warning("⚠️ 语音识别超时，将跳过语音转录功能")
                asr_result["transcription"] = None
        else:
            st.info("ℹ️ 未执行语音识别（无视频文件）")

        if asr_result.get("transcription"):
            # 显示语音识别成功
            st.success("✅ 语音转录完成！")

            # 显示转录文本
            st.markdown("**语音转录内容:**")
            st.info(asr_result["transcription"])

            # 保存到文件
            with open(st.session_state.output_file, "a", encoding="utf-8") as f:
                f.write("\n语音转录结果:\n")
                f.write("=" * 80 + "\n")
                f.write(asr_result["transcription"] + "\n")
                f.write("=" * 80 + "\n\n")

            # 💡 将ASR结果保存到cache目录，使其可以被向量化和用于分析
            asr_cache_file = os.path.join(
                st.session_state.cache_dir, "asr_transcription.txt"
            )
            with open(asr_cache_file, "w", encoding="utf-8") as f:
                f.write(f"语音转录内容：{asr_result['transcription']}")

        elif asr_result.get("error"):
            st.warning(f"⚠️ 语音转录失败: {asr_result['error']}")
        else:
            st.info("ℹ️ 视频无语音内容或语音识别被跳过")

        st.divider()

        # 构建或加载向量数据库
        if (
            st.session_state.use_existing_vector_db
            and st.session_state.existing_vector_db_path
        ):
            # 使用已有向量数据库
            st.info("📚 正在加载已有向量数据库...")
            try:
                # 加载Embedding模型
                if st.session_state.embedding_model is None:
                    st.info("正在初始化Embedding模型...")
                    st.session_state.embedding_model = load_embedding_model()

                if st.session_state.embedding_model:
                    # 加载已有向量数据库
                    from langchain_community.vectorstores import Chroma

                    st.session_state.vector_store = Chroma(
                        persist_directory=st.session_state.vector_store_path,
                        embedding_function=st.session_state.embedding_model,
                    )

                    # 验证数据库
                    try:
                        collection = st.session_state.vector_store._collection
                        count = collection.count()
                        st.success(f"✅ 成功加载向量数据库，包含 {count} 个文档片段")
                    except:
                        st.warning("向量数据库加载成功，但无法获取文档数量")
                else:
                    st.error("Embedding模型加载失败")
                    st.session_state.vector_store = None

            except Exception as e:
                st.error(f"加载向量数据库失败: {str(e)}")
                st.warning("将尝试重新构建向量数据库...")
                st.session_state.vector_store = build_vector_store(
                    st.session_state.cache_dir, force_rebuild=True
                )
        else:
            # 重新构建向量数据库
            st.info("正在构建向量数据库...")
            st.session_state.vector_store = build_vector_store(
                st.session_state.cache_dir, force_rebuild=True
            )

        st.success("🎉 视频分析完成！")
        st.info("💡 请切换到'多方式分析'页面查看完整报告、速读摘要和智能检索功能")

        # Token用量已在分析前使用公式预估显示，此处不再重复

        # 如果使用了Ollama模型，显示停止按钮
        if st.session_state.use_ollama:
            st.session_state.ollama_used = True
            st.session_state.show_stop_button = True
            st.warning(
                f"Ollama模型 {st.session_state.selected_model} 仍在运行中，可以点击下方按钮停止以释放资源"
            )

    except Exception as e:
        st.error(f"处理过程中发生错误: {str(e)}")
    finally:
        st.session_state.processing = False


# --- Streamlit UI ---
def video_analysis_page():
    """视频分析页面"""
    st.title("🎬 视频智能分析处理套件 v3.0")
    st.caption("by CN_榨汁Ovo - 愿世界和平🕊️")

    # 添加作者联系方式
    st.markdown("### 作者联系方式")
    st.markdown("""
    - **QQ**: 153115068(请备注来意)
    - **QQ邮箱**: 153115068@QQ.COM(请备注来意)
    - **Bug反馈**: 请通过QQ或邮箱反馈问题，说明bug情况，需 报错截图 或bug截图
    -  您的反馈将帮助作者更好更快的抓虫🐛🐛🐛
    """)

    # 显示免责声明和操作指南
    show_disclaimer()
    show_important_reminder()
    show_operation_guide()

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置参数")

        # 工作模式选择
        st.subheader("🎯 工作模式")
        work_mode = st.radio(
            "选择您的工作场景：",
            ["🆕 新视频分析", "🔄 关键帧再解析"],
            help="新视频分析：上传视频进行完整分析\n关键帧再解析：使用已有关键帧重新分析",
        )

        is_reparse_mode = work_mode == "🔄 关键帧再解析"

        if is_reparse_mode:
            st.info(
                "💡 **关键帧再解析模式**\n\n适用场景：\n- 更换模型重新分析\n- 优化提示词\n- 对之前结果不满意"
            )

        # 提示词自定义
        st.divider()
        st.subheader("📝 提示词自定义")
        st.session_state.user_custom_prompt = st.text_area(
            "自定义提示词（可选）",
            value=st.session_state.user_custom_prompt,
            placeholder="在此添加您想要模型额外关注的内容或以特别格式输出结果的提示词...",
            help="此内容将附加到基础提示词后面",
        )

        # 模型选择
        st.divider()
        st.subheader("🤖 模型选择")
        model_option = st.radio(
            "选择模型后端:",
            ["阿里云 DashScope", "Ollama (本地部署)"],
            index=0,
            help="阿里云需要环境变量设置API密钥，Ollama需要本地运行服务",
        )

        st.session_state.use_ollama = model_option == "Ollama (本地部署)"

        if not st.session_state.use_ollama:
            st.divider()
            st.subheader("🖼️ 视觉模型选择（VLM）")
            st.caption("用于分析图片内容")

            vlm_options = {
                "qwen3-vl-plus": "Qwen3-VL-Plus（新一代，更强大，略慢）",
                "qwen-vl-plus": "Qwen-VL-Plus（推荐，速度快）",
                "qwen-vl-max-latest": "Qwen-VL-Max Latest（最强，较慢）",
                "qwen-vl-plus-latest": "Qwen-VL-Plus Latest（最新版）",
            }

            st.session_state.vlm_model = st.selectbox(
                "选择视觉语言模型:",
                options=list(vlm_options.keys()),
                format_func=lambda x: vlm_options[x],
                index=0,
                help="用于分析视频关键帧图片",
            )

            # 从环境变量获取API密钥
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if api_key:
                st.success("✅ API密钥已配置")
            else:
                st.error("❌ 请先在环境变量中设置 DASHSCOPE_API_KEY")
        else:
            # 获取Ollama模型列表
            if st.button("刷新模型列表", key="refresh_models"):
                st.session_state.ollama_models = get_ollama_models()

            if not st.session_state.ollama_models:
                st.session_state.ollama_models = get_ollama_models()

            if st.session_state.ollama_models:
                st.selectbox(
                    "选择Ollama模型:",
                    st.session_state.ollama_models,
                    key="selected_model",
                    help="选择要使用的Ollama视觉模型",
                )
            else:
                st.warning("未找到任何Ollama模型，请确保Ollama服务正在运行")

        # 关键帧来源配置
        st.divider()
        if is_reparse_mode:
            st.subheader("🔄 选择要再解析的关键帧")
            st.caption("请选择之前已经提取的关键帧文件夹")
            # 再解析模式下，自动启用使用已有关键帧
            st.session_state.use_existing_keyframes = True
            use_existing_keyframes = True
        else:
            st.subheader("🔄 重用已有数据（可选）")
            st.caption(
                "如果之前已经处理过视频，可以选择重用已有的关键帧和向量数据库，避免重复处理"
            )

            # 使用已有关键帧
            use_existing_keyframes = st.checkbox(
                "使用已有关键帧文件夹",
                value=st.session_state.use_existing_keyframes,
                help="选择之前已经提取过的关键帧文件夹，跳过抽帧步骤",
            )
            st.session_state.use_existing_keyframes = use_existing_keyframes

        if use_existing_keyframes:
            # 检查"过往信息"文件夹
            history_folder = os.path.join(get_program_dir(), "过往信息")

            # 选择输入方式
            input_method = st.radio(
                "选择输入方式：",
                ["从过往信息文件夹选择", "手动输入路径"],
                key="keyframe_input_method",
                horizontal=True,
            )

            if input_method == "从过往信息文件夹选择":
                # 从过往信息文件夹选择
                if os.path.exists(history_folder):
                    # 获取所有子文件夹
                    history_items = []
                    try:
                        for item in os.listdir(history_folder):
                            item_path = os.path.join(history_folder, item)
                            if os.path.isdir(item_path):
                                # 检查是否包含关键帧
                                keyframe_path = os.path.join(item_path, "keyframes")
                                (
                                    normalized_keyframe_path,
                                    image_files,
                                    keyframe_error,
                                ) = resolve_keyframe_directory(keyframe_path)
                                if not keyframe_error:
                                    history_items.append(
                                        {
                                            "name": item,
                                            "path": normalized_keyframe_path,
                                            "count": len(image_files),
                                        }
                                    )
                    except Exception as e:
                        st.warning(f"读取过往信息文件夹时出错: {str(e)}")

                    if history_items:
                        # 创建选择框
                        selected_item = st.selectbox(
                            "选择历史处理记录：",
                            options=history_items,
                            format_func=lambda x: (
                                f"{x['name']} ({x['count']} 个关键帧)"
                            ),
                            key="history_keyframe_selector",
                        )

                        if selected_item:
                            st.session_state.existing_keyframes_path = selected_item[
                                "path"
                            ]
                            st.success(f"✅ 已选择: {selected_item['name']}")
                            st.info(f"📁 路径: {selected_item['path']}")
                    else:
                        st.warning("⚠️ 过往信息文件夹中没有找到包含关键帧的记录")
                        st.caption("提示：处理过的视频会自动保存在'过往信息'文件夹中")
                else:
                    st.warning("⚠️ 过往信息文件夹不存在")
                    st.caption(f"将在首次处理视频时自动创建: {history_folder}")
            else:
                # 手动输入路径
                existing_keyframes_path = st.text_input(
                    "关键帧文件夹路径",
                    value=st.session_state.existing_keyframes_path,
                    placeholder="例如: keyframes 或 C:/path/to/keyframes",
                    help="输入已有的关键帧文件夹路径（绝对路径或相对路径）",
                )
                st.session_state.existing_keyframes_path = existing_keyframes_path

                # 验证手动输入的路径
                if existing_keyframes_path:
                    normalized_keyframe_path, image_files, keyframe_error = (
                        resolve_keyframe_directory(existing_keyframes_path)
                    )
                    if keyframe_error:
                        st.error(f"❌ {keyframe_error}")
                    else:
                        st.success(f"✅ 找到 {len(image_files)} 个关键帧文件")
                        if normalized_keyframe_path != existing_keyframes_path:
                            st.caption(f"实际读取路径: {normalized_keyframe_path}")

            # 强制重新解析选项
            st.divider()
            if is_reparse_mode:
                # 再解析模式下，自动启用强制重新解析
                st.warning("🔄 **再解析模式已启用**")
                st.caption("将忽略所有缓存结果，重新分析所有关键帧")
                st.session_state.force_reparse_keyframes = True

                # 显示再解析说明
                with st.expander("ℹ️ 关于再解析", expanded=False):
                    st.markdown("""
                    **再解析会做什么：**
                    - ✅ 忽略所有已有的分析缓存
                    - ✅ 使用当前选择的模型重新分析
                    - ✅ 应用当前的提示词设置
                    - ✅ 生成新的分析结果
                    
                    **注意事项：**
                    - ⚠️ 使用云端API会产生新的费用
                    - ⚠️ 本地模型免费但耗时较长
                    - 💡 旧的分析缓存会被覆盖
                    """)
            else:
                st.caption("💡 提示：如果已有解析结果，可以选择重新解析")
                force_reparse = st.checkbox(
                    "强制重新解析关键帧",
                    value=st.session_state.force_reparse_keyframes,
                    help="即使已有缓存结果，也强制重新分析所有关键帧。适用于更换模型、修改提示词或优化分析结果的场景。",
                )
                st.session_state.force_reparse_keyframes = force_reparse

                if force_reparse:
                    st.info("🔄 将忽略已有的分析缓存，重新解析所有关键帧")

        # 使用已有向量数据库
        if not is_reparse_mode:
            # 只在非再解析模式下显示向量数据库选项
            use_existing_vector_db = st.checkbox(
                "使用已有向量数据库",
                value=st.session_state.use_existing_vector_db,
                help="选择之前已经构建的向量数据库文件夹，跳过向量构建步骤",
            )
            st.session_state.use_existing_vector_db = use_existing_vector_db
        else:
            # 再解析模式下，不使用已有向量数据库（因为分析结果变了）
            st.session_state.use_existing_vector_db = False
            use_existing_vector_db = False
            st.caption("💡 再解析模式下，将自动重新构建向量数据库以匹配新的分析结果")

        if use_existing_vector_db:
            # 检查"过往信息"文件夹
            history_folder = os.path.join(get_program_dir(), "过往信息")

            # 选择输入方式
            input_method_db = st.radio(
                "选择输入方式：",
                ["从过往信息文件夹选择", "手动输入路径"],
                key="vector_db_input_method",
                horizontal=True,
            )

            if input_method_db == "从过往信息文件夹选择":
                # 从过往信息文件夹选择
                if os.path.exists(history_folder):
                    # 获取所有子文件夹
                    history_db_items = []
                    try:
                        for item in os.listdir(history_folder):
                            item_path = os.path.join(history_folder, item)
                            if os.path.isdir(item_path):
                                # 检查是否包含向量数据库
                                db_path = os.path.join(item_path, "chroma_db")
                                if os.path.exists(db_path) and os.path.isdir(db_path):
                                    # 检查是否是有效的向量数据库
                                    if os.path.exists(
                                        os.path.join(db_path, "chroma.sqlite3")
                                    ):
                                        history_db_items.append(
                                            {"name": item, "path": db_path}
                                        )
                    except Exception as e:
                        st.warning(f"读取过往信息文件夹时出错: {str(e)}")

                    if history_db_items:
                        # 创建选择框
                        selected_db_item = st.selectbox(
                            "选择历史向量数据库：",
                            options=history_db_items,
                            format_func=lambda x: x["name"],
                            key="history_vector_db_selector",
                        )

                        if selected_db_item:
                            st.session_state.existing_vector_db_path = selected_db_item[
                                "path"
                            ]
                            st.success(f"✅ 已选择: {selected_db_item['name']}")
                            st.info(f"📚 路径: {selected_db_item['path']}")
                    else:
                        st.warning("⚠️ 过往信息文件夹中没有找到向量数据库")
                        st.caption("提示：处理过的视频会自动保存在'过往信息'文件夹中")
                else:
                    st.warning("⚠️ 过往信息文件夹不存在")
                    st.caption(f"将在首次处理视频时自动创建: {history_folder}")
            else:
                # 手动输入路径
                existing_vector_db_path = st.text_input(
                    "向量数据库文件夹路径",
                    value=st.session_state.existing_vector_db_path,
                    placeholder="例如: chroma_db 或 C:/path/to/chroma_db",
                    help="输入已有的向量数据库文件夹路径（绝对路径或相对路径）",
                )
                st.session_state.existing_vector_db_path = existing_vector_db_path

                # 验证手动输入的路径
                if existing_vector_db_path:
                    if os.path.exists(existing_vector_db_path) and os.path.isdir(
                        existing_vector_db_path
                    ):
                        st.success(f"✅ 向量数据库路径有效")
                    else:
                        st.error("❌ 路径不存在或不是文件夹")

        # 视频上传（仅在新视频分析模式下显示）
        if not is_reparse_mode:
            st.divider()
            uploaded_file = st.file_uploader(
                "上传视频文件",
                type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
                help="""视频大小无上限——支持格式: mp4, avi, mov, mkv, flv, wmv""",
            )

            if uploaded_file:
                # 保存上传的文件
                st.session_state.video_path = uploaded_file.name
                with open(st.session_state.video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"已上传: {uploaded_file.name}")
        else:
            # 再解析模式下不需要上传视频
            uploaded_file = None
            st.divider()
            st.info(
                "📌 **再解析模式下无需上传视频**\n\n系统将直接使用已选择的关键帧进行分析"
            )

        # 开始分析按钮
        # 判断是否可以开始分析
        can_start = False
        disable_reason = ""

        if st.session_state.processing:
            disable_reason = "分析进行中..."
        elif st.session_state.use_ollama and not st.session_state.selected_model:
            disable_reason = "请先选择Ollama模型"
        elif is_reparse_mode:
            # 再解析模式下，检查是否选择了关键帧
            if not st.session_state.existing_keyframes_path:
                disable_reason = "请选择要再解析的关键帧文件夹"
            else:
                _, _, keyframe_error = resolve_keyframe_directory(
                    st.session_state.existing_keyframes_path
                )
                if keyframe_error:
                    disable_reason = keyframe_error
                else:
                    can_start = True
        elif st.session_state.use_existing_keyframes:
            # 勾选了使用已有关键帧
            if not st.session_state.existing_keyframes_path:
                disable_reason = "请输入关键帧文件夹路径"
            else:
                _, _, keyframe_error = resolve_keyframe_directory(
                    st.session_state.existing_keyframes_path
                )
                if keyframe_error:
                    disable_reason = keyframe_error
                else:
                    can_start = True
        elif uploaded_file:
            # 如果上传了视频，可以开始
            can_start = True
        else:
            disable_reason = "请上传视频或选择已有关键帧"

        # 根据模式显示不同的按钮文本
        button_text = "🔄 开始再解析" if is_reparse_mode else "开始分析"
        button_help = (
            "使用当前配置重新分析所选关键帧" if is_reparse_mode else "开始视频分析处理"
        )

        if st.button(
            button_text,
            disabled=not can_start,
            type="primary",
            use_container_width=True,
            help=disable_reason if not can_start else button_help,
        ):
            run_video_analysis()

        # 显示状态
        if st.session_state.processing:
            st.warning("分析进行中，请勿关闭页面...")

        # 停止Ollama模型按钮
        if st.session_state.show_stop_button:
            if st.button(
                "停止Ollama模型",
                key="stop_ollama",
                help="停止当前运行的Ollama模型以释放系统资源",
                use_container_width=True,
                type="secondary",
            ):
                stop_ollama_model()

        st.divider()
        st.caption("注意: 处理时间取决于视频长度和模型性能")

    # 主内容区
    if st.session_state.processing:
        st.info("视频分析中，请稍候...")


def multi_analysis_page():
    """多方式分析页面"""
    st.title("📊 多方式分析")

    # 侧边栏配置
    with st.sidebar:
        st.header("模型配置")

        # 模型后端选择（独立配置）
        model_option = st.radio(
            "选择模型后端:",
            ["阿里云 DashScope", "Ollama (本地部署)"],
            index=0,
            key="multi_analysis_model_backend",
            help="选择用于生成报告和问答的模型后端",
        )

        st.session_state.llm_use_ollama = model_option == "Ollama (本地部署)"

        if not st.session_state.llm_use_ollama:
            # 阿里云配置
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if api_key:
                st.success("✅ API密钥已配置")

                # 确保client已初始化（但不影响视频分析页面的配置）
                if not st.session_state.client:
                    st.session_state.client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )

                st.subheader("📝 文本模型选择（LLM）")
                st.caption("用于生成报告、摘要和问答")

                llm_options = {
                    "qwen-plus": "Qwen-Plus（推荐，性价比高）",
                    "qwen-max": "Qwen-Max（最强）",
                    "qwen-plus-latest": "Qwen-Plus Latest（最新版）",
                    "qwen-turbo": "Qwen-Turbo（最快，便宜）",
                    "qwen-turbo-latest": "Qwen-Turbo Latest（最新快速版）",
                    "qwen-long": "Qwen-Long（长文本，最大1M tokens）",
                }

                st.session_state.llm_model = st.selectbox(
                    "选择文本生成模型:",
                    options=list(llm_options.keys()),
                    format_func=lambda x: llm_options[x],
                    index=0,
                    key="llm_model_selector_multi",
                    help="用于生成报告、摘要和智能问答",
                )
            else:
                st.error("❌ 请先在环境变量中设置 DASHSCOPE_API_KEY")
        else:
            # Ollama配置
            st.subheader("Ollama 模型选择")

            # 获取Ollama模型列表
            if st.button("刷新模型列表", key="refresh_models_multi"):
                st.session_state.ollama_models = get_ollama_models()

            if not st.session_state.ollama_models:
                st.session_state.ollama_models = get_ollama_models()

            if st.session_state.ollama_models:
                st.session_state.llm_ollama_model = st.selectbox(
                    "选择Ollama模型:",
                    st.session_state.ollama_models,
                    key="llm_ollama_selector",
                    help="选择要使用的Ollama模型（用于文本生成）",
                )
                st.success(f"✅ 已选择: {st.session_state.llm_ollama_model}")
            else:
                st.warning("未找到任何Ollama模型，请确保Ollama服务正在运行")

        st.divider()
        st.header("向量数据库管理")

        # Embedding模型状态
        st.subheader("🧠 Embedding模型")
        if st.session_state.embedding_model:
            st.success("✅ 已加载")
        else:
            st.warning("❌ 未加载")
            st.caption("模型将在首次构建向量数据库时自动加载")

        st.divider()

        # 显示数据库状态
        st.subheader("📚 向量数据库")
        if st.session_state.vector_store:
            try:
                count = st.session_state.vector_store._collection.count()
                st.success(f"✅ 数据库已加载\n\n文档片段: {count}")
            except:
                st.info("数据库已初始化")
        else:
            st.warning("❌ 数据库未初始化")

        # 加载/重建数据库按钮
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔄 加载数据库",
                help="从磁盘加载已有的向量数据库",
                use_container_width=True,
            ):
                with st.spinner("正在加载..."):
                    st.session_state.vector_store = build_vector_store(
                        st.session_state.cache_dir, force_rebuild=False
                    )
                if st.session_state.vector_store:
                    st.success("加载成功！")
                    st.rerun()
                else:
                    st.error("加载失败，请检查是否已完成视频分析")

        with col2:
            if st.button(
                "🔨 重建数据库",
                help="重新构建向量数据库（适用于更新了分析结果）",
                use_container_width=True,
            ):
                with st.spinner("正在重建..."):
                    st.session_state.vector_store = build_vector_store(
                        st.session_state.cache_dir, force_rebuild=True
                    )
                if st.session_state.vector_store:
                    st.success("重建成功！")
                    st.rerun()
                else:
                    st.error("重建失败，请检查缓存文件")

    if not st.session_state.vector_store:
        st.warning("⚠️ 请先完成视频分析或点击左侧边栏的「加载数据库」按钮")
        st.info("""
        **提示：**
        1. 如果是首次使用，请先到「视频分析」页面完成视频分析
        2. 如果之前已经分析过视频，点击左侧边栏的「🔄 加载数据库」按钮
        3. 如果更新了分析结果，点击「🔨 重建数据库」按钮
        """)
        return

    # 使用选项卡组织不同分析方式
    tab1, tab2, tab3 = st.tabs(["📝 完整分析报告", "⚡ 速读摘要", "🔍 智能检索"])

    with tab1:
        st.subheader("完整分析报告")
        st.info("生成包含所有关键帧分析和语音转录的完整报告")

        if st.button("生成完整报告", key="generate_full_report"):
            with st.spinner("正在生成完整分析报告..."):
                comprehensive_report = generate_comprehensive_report(
                    st.session_state.cache_dir
                )

                # 保存完整报告
                report_file = "完整分析报告.txt"
                with open(report_file, "w", encoding="utf-8") as f:
                    f.write("视频智能分析完整报告\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(comprehensive_report)
                    f.write("\n\n")
                    f.write("=" * 50 + "\n")
                    f.write("注：本报告已整合视觉分析和语音转录内容\n")

                # 显示完整报告
                st.markdown("### 完整分析报告内容")
                st.markdown(comprehensive_report)

                # 提供结果下载
                with open(report_file, "rb") as f:
                    st.download_button(
                        label="📥 下载完整分析报告",
                        data=f,
                        file_name=report_file,
                        mime="text/plain",
                        use_container_width=True,
                    )

                # Token用量仅在视频分析时使用公式预估，不在此处显示

    with tab2:
        st.subheader("速读摘要")
        st.info("快速生成视频内容的精简摘要，便于快速了解核心信息")

        if st.button("生成速读摘要", key="generate_quick_summary"):
            with st.spinner("正在生成速读摘要..."):
                summary = generate_quick_summary(st.session_state.cache_dir)
                st.markdown("### 视频核心摘要")
                st.markdown(summary)

                # 提供摘要下载
                summary_file = "视频核心摘要.txt"
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("视频核心摘要\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(summary)

                with open(summary_file, "rb") as f:
                    st.download_button(
                        label="📥 下载速读摘要",
                        data=f,
                        file_name=summary_file,
                        mime="text/plain",
                        use_container_width=True,
                    )

                # Token用量仅在视频分析时使用公式预估，不在此处显示

    with tab3:
        st.subheader("🔍 智能检索")

        # 添加功能介绍
        with st.expander("📖 什么是智能检索？", expanded=False):
            st.markdown("""
            ### 功能概述
            智能检索使用先进的**向量语义搜索**技术，能够理解您问题的语义含义，而不仅仅是关键词匹配。
            
            ### 核心优势
            - 🧠 **语义理解**：理解问题的真实含义，而非简单的关键词匹配
            - 🎯 **精准定位**：快速找到视频中与问题最相关的片段
            - 📊 **多策略支持**：提供三种检索策略，适应不同场景
            - 🖼️ **可视化**：显示相关关键帧，直观展示视频内容
            
            ### 使用方法
            1. 在下方输入您的问题
            2. （可选）调整高级检索选项
            3. 点击"🔍 检索"按钮
            4. 查看相关片段和智能回答
            """)

        # 添加检索策略说明
        with st.expander("🎯 检索策略说明", expanded=False):
            st.markdown("""
            ### 三种检索策略详解
            
            #### 1️⃣ 相似度搜索（默认推荐）
            **原理**：计算问题与视频内容的语义相似度，返回最相关的片段。
            
            **适用场景**：
            - ✅ 日常使用
            - ✅ 查找特定内容（如"视频中出现了什么人物？"）
            - ✅ 概括性问题（如"视频主要讲了什么？"）
            
            **优点**：速度快，结果相关度高  
            **缺点**：可能包含一些重复内容
            
            ---
            
            #### 2️⃣ MMR搜索（减少冗余）
            **原理**：Maximum Marginal Relevance（最大边际相关性）算法，在保证相关性的同时，减少结果间的重复。
            
            **适用场景**：
            - ✅ 希望看到**不同角度**的内容
            - ✅ 视频内容重复度高（如同一场景多次出现）
            - ✅ 需要**全面了解**视频各个方面
            
            **优点**：结果多样化，信息覆盖面广  
            **缺点**：可能牺牲部分相关度
            
            **使用建议**：
            - 当默认搜索返回太多相似内容时使用
            - 想要获得更全面的视频理解时使用
            
            ---
            
            #### 3️⃣ 阈值搜索（过滤低相关度）
            **原理**：只返回相似度分数高于设定阈值的结果，过滤掉不太相关的内容。
            
            **适用场景**：
            - ✅ 查找**非常具体**的内容
            - ✅ 确保结果**高度相关**
            - ✅ 宁缺毋滥的场景
            
            **优点**：结果精准度极高  
            **缺点**：可能找不到结果（如果阈值设置太高）
            
            **使用建议**：
            - 从默认阈值(0.5)开始
            - 如果结果太少，降低阈值
            - 如果结果不够精准，提高阈值
            
            ---
            
            ### 💡 选择建议
            
            | 需求 | 推荐策略 |
            |------|---------|
            | 首次使用 | 相似度搜索 |
            | 结果太相似 | MMR搜索 |
            | 查找特定内容 | 阈值搜索 |
            | 快速查询 | 相似度搜索 |
            | 深度分析 | MMR搜索 |
            """)

        st.info(
            "💡 提示：输入问题后点击检索，系统会自动找到最相关的视频片段并生成智能回答"
        )

        # 高级检索选项（可折叠）
        with st.expander("⚙️ 高级检索选项", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                search_type = st.selectbox(
                    "检索策略",
                    ["similarity", "mmr", "similarity_score"],
                    format_func=lambda x: {
                        "similarity": "相似度搜索（默认）",
                        "mmr": "MMR搜索（减少冗余）",
                        "similarity_score": "阈值搜索（过滤低相关度）",
                    }[x],
                    help="选择不同的检索策略以优化结果",
                )

                k_results = st.slider(
                    "返回结果数量",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="返回的相关片段数量",
                )

            with col2:
                show_keyframes = st.checkbox(
                    "显示关键帧图片", value=True, help="在结果中显示对应的关键帧图片"
                )

                include_citations = st.checkbox(
                    "包含引用信息", value=True, help="在回答中包含引用的关键帧编号"
                )

        # 查询输入
        query = st.text_input(
            "输入您的问题：",
            placeholder="例如：视频中出现了哪些人物？主要讲了什么内容？",
            key="rag_query",
        )

        # 检索按钮
        col_search, col_clear = st.columns([3, 1])
        with col_search:
            search_clicked = st.button(
                "🔍 检索", key="rag_search", type="primary", use_container_width=True
            )
        with col_clear:
            if st.button("🗑️ 清空历史", key="clear_history", use_container_width=True):
                st.session_state.query_history = []
                st.success("查询历史已清空")
                st.rerun()

        # 执行检索
        if search_clicked:
            if query:
                with st.spinner("正在检索相关信息..."):
                    search_results = search_vector_store(
                        query,
                        st.session_state.vector_store,
                        k=k_results,
                        search_type=search_type,
                    )

                    if search_results:
                        st.success(f"✅ 找到 {len(search_results)} 条相关信息")

                        # 显示检索结果
                        st.markdown("### 📋 检索到的相关片段")
                        for i, doc in enumerate(search_results):
                            frame_num = doc.metadata.get("frame", "未知")
                            content_type = doc.metadata.get("content_type", "visual")

                            # 根据内容类型设置标题
                            if content_type == "audio":
                                title = f"片段 {i + 1} - 🎤 语音转录"
                            else:
                                title = f"片段 {i + 1} - 🖼️ 关键帧 {frame_num}"

                            with st.expander(title, expanded=False):
                                # 显示关键帧图片（仅对视觉内容）
                                if show_keyframes and content_type == "visual":
                                    keyframe_path = os.path.join(
                                        st.session_state.keyframe_dir,
                                        f"{frame_num}.png",
                                    )
                                    if os.path.exists(keyframe_path):
                                        try:
                                            img = Image.open(keyframe_path)
                                            st.image(
                                                img,
                                                caption=f"关键帧 {frame_num}",
                                                use_container_width=True,
                                            )
                                        except:
                                            st.warning("无法加载关键帧图片")

                                # 显示文本内容
                                if content_type == "audio":
                                    st.markdown("**语音内容:**")
                                else:
                                    st.markdown("**内容描述:**")
                                st.write(doc.page_content)

                                # 显示元数据
                                metadata_info = []
                                if doc.metadata.get("char_count"):
                                    metadata_info.append(
                                        f"字符数: {doc.metadata['char_count']}"
                                    )
                                if content_type == "audio":
                                    metadata_info.append("类型: 语音转录")
                                elif content_type == "visual":
                                    metadata_info.append("类型: 视觉内容")

                                if metadata_info:
                                    st.caption(" | ".join(metadata_info))

                        # 生成智能回答
                        st.markdown("### 🤖 智能回答")
                        with st.spinner("正在生成回答..."):
                            response, frame_refs = generate_rag_response(
                                query,
                                search_results,
                                include_citations=include_citations,
                            )
                            st.markdown(response)

                            # 保存到查询历史
                            st.session_state.query_history.append(
                                {
                                    "query": query,
                                    "response": response,
                                    "frame_refs": frame_refs,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }
                            )

                            # Token用量仅在视频分析时使用公式预估，不在此处显示
                    else:
                        st.warning("未找到相关信息，请尝试调整查询内容或检索参数")
            else:
                st.warning("请输入问题后再检索")

        # 显示查询历史
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📝 查询历史")

            for i, history_item in enumerate(
                reversed(st.session_state.query_history[-5:])
            ):
                with st.expander(
                    f"问题 {len(st.session_state.query_history) - i}: {history_item['query']}",
                    expanded=False,
                ):
                    st.caption(f"时间: {history_item['timestamp']}")
                    st.markdown("**回答:**")
                    st.markdown(history_item["response"])


def main():
    st.set_page_config(
        page_title="视频智能分析处理套件 v3.0", page_icon="🎬", layout="wide"
    )

    # 页面导航
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择页面:", ["视频分析", "多方式分析"])

    # 根据选择显示页面
    if page == "视频分析":
        video_analysis_page()
    elif page == "多方式分析":
        multi_analysis_page()


if __name__ == "__main__":
    main()
