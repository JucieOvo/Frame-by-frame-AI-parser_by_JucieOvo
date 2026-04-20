"""
模块名称：downloads
功能描述：
    提供启动阶段所需模型的真实下载逻辑。
    该模块负责下载 FunASR 模型与 Qwen3 Embedding 模型，并统一写入项目缓存目录。

主要组件：
    - download_funasr_models: 下载 FunASR 语音识别模型。
    - download_embedding_model: 下载 Qwen3 Embedding 模型。

依赖说明：
    - os: 路径处理。
    - video_ai_suite.backend.runtime: 缓存环境初始化。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从原始 run.py 中抽离模型下载逻辑。
"""

from __future__ import annotations

import os

from video_ai_suite.backend.runtime import early_set_cache_env


def download_funasr_models() -> tuple[bool, str]:
    """
    下载 FunASR 模型到项目缓存目录。

    :return: 下载是否成功，以及实际缓存目录。
    """
    print("正在下载 FunASR 语音识别模型")
    print("该过程可能持续几分钟，请耐心等待")

    cache_dir = early_set_cache_env()
    modelscope_cache = os.path.join(cache_dir, "modelscope", "hub")

    try:
        print("正在导入 FunASR 库")
        from funasr import AutoModel

        print(f"模型下载目录: {modelscope_cache}")
        AutoModel(
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            model_revision="v2.0.4",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_model_revision="v2.0.4",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            punc_model_revision="v2.0.4",
        )
        print("FunASR 模型下载完成")
        return True, modelscope_cache
    except ImportError:
        print("未找到 FunASR 库，请先安装 funasr")
        return False, modelscope_cache
    except Exception as exc:
        print(f"FunASR 模型下载失败: {str(exc)}")
        print("请检查网络连接与可用磁盘空间")
        return False, modelscope_cache


def download_embedding_model() -> tuple[bool, str | None]:
    """
    下载 Qwen3 Embedding 模型到项目缓存目录。

    :return: 下载是否成功，以及模型路径。
    """
    print("正在下载 Qwen3 Embedding 向量模型")
    print("该过程可能持续几分钟，请耐心等待")

    cache_dir = early_set_cache_env()
    modelscope_cache = os.path.join(cache_dir, "modelscope", "hub")

    try:
        print("正在导入 ModelScope 库")
        from modelscope import snapshot_download

        model_name = "Qwen/Qwen3-Embedding-0.6B"
        print(f"模型下载目录: {modelscope_cache}")
        print(f"正在下载 {model_name}")
        model_path = snapshot_download(model_name, cache_dir=modelscope_cache)
        print(f"Qwen3 Embedding 模型下载完成: {model_path}")
        return True, model_path
    except ImportError:
        print("未找到 ModelScope 库，请先安装 modelscope")
        return False, None
    except Exception as exc:
        print(f"Qwen3 Embedding 模型下载失败: {str(exc)}")
        print("请检查网络连接与可用磁盘空间")
        return False, None
