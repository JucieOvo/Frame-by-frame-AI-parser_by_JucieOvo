import os
import sys

# --- 关键：在导入任何模型库之前，先强制设置环境变量 ---
def _early_set_cache_env():
    """在导入模型库之前设置环境变量"""
    if getattr(sys, 'frozen', False):
        program_dir = os.path.dirname(sys.executable)
    else:
        program_dir = os.path.dirname(os.path.abspath(__file__))
    
    cache_dir = os.path.join(program_dir, '.cache')
    
    # 强制设置所有可能的环境变量
    os.environ['MODELSCOPE_CACHE_DIR'] = os.path.join(cache_dir, 'modelscope', 'hub')
    os.environ['MODELSCOPE_CACHE'] = os.path.join(cache_dir, 'modelscope', 'hub')
    os.environ['MODELSCOPE_HOME'] = os.path.join(cache_dir, 'modelscope')
    os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'huggingface', 'transformers')
    os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')
    os.environ['MODELSCOPE_SDK_DEBUG'] = '0'
    
    print(f"[EARLY INIT] 缓存目录设置为: {os.environ['MODELSCOPE_CACHE_DIR']}")
    
    return cache_dir

# 立即执行
_early_set_cache_env()

# 现在导入其他库
import streamlit.web.cli as stcli
import ctypes
import shutil
import subprocess
import time
from datetime import datetime

def clear_old_cache():
    """清除旧的缓存文件夹，移动到过往信息文件夹"""
    try:
        # 获取程序目录
        if getattr(sys, 'frozen', False):
            program_dir = os.path.dirname(sys.executable)
        else:
            program_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建过往信息文件夹
        archive_dir = os.path.join(program_dir, "过往信息", timestamp)
        
        # 需要移动的文件夹列表
        folders_to_move = [
            ("cache", "cache"),
            ("keyframes", "keyframes"),
            ("chroma_db", "chroma_db")
        ]
        
        moved_count = 0
        
        for folder_name, display_name in folders_to_move:
            folder_path = os.path.join(program_dir, folder_name)
            
            # 如果文件夹存在才移动
            if os.path.exists(folder_path):
                try:
                    # 确保归档目录存在
                    os.makedirs(archive_dir, exist_ok=True)
                    
                    # 目标路径
                    target_path = os.path.join(archive_dir, folder_name)
                    
                    # 移动文件夹
                    shutil.move(folder_path, target_path)
                    print(f"✅ 已移动 {display_name} 文件夹到归档目录")
                    moved_count += 1
                    
                except Exception as e:
                    print(f"⚠️ 移动 {display_name} 文件夹时出错: {str(e)}")
            else:
                # 文件夹不存在，不做任何操作
                pass
        
        if moved_count > 0:
            print(f"\n📦 已将 {moved_count} 个文件夹归档到: {archive_dir}")
        else:
            print("ℹ️ 没有需要清理的缓存文件夹")
            
    except Exception as e:
        print(f"⚠️ 清除缓存时出错: {str(e)}")
        print("继续启动程序...")

def is_admin():
    """检查是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def get_program_dir():
    """获取程序目录的绝对路径"""
    if getattr(sys, 'frozen', False):
        # 打包后的程序
        return os.path.dirname(sys.executable)
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))

def get_cache_dir():
    """获取缓存目录的绝对路径（在程序目录下）"""
    program_dir = get_program_dir()
    cache_dir = os.path.join(program_dir, '.cache')
    return cache_dir

def force_set_cache_env():
    """强制设置缓存目录环境变量，防止用户环境变量干扰"""
    cache_dir = get_cache_dir()
    
    # 强制设置ModelScope缓存目录
    os.environ['MODELSCOPE_CACHE_DIR'] = os.path.join(cache_dir, 'modelscope', 'hub')
    os.environ['MODELSCOPE_SDK_DEBUG'] = '0'
    
    # 强制设置HuggingFace缓存目录
    os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'huggingface', 'transformers')
    
    # 强制设置Torch缓存目录
    os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch')
    
    # 创建所有缓存目录
    for env_var in ['MODELSCOPE_CACHE_DIR', 'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME']:
        dir_path = os.environ[env_var]
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ 设置缓存目录: {env_var} = {dir_path}")
    
    return cache_dir

def check_funasr_models():
    """检查FunASR模型是否已下载"""
    # 强制使用程序目录下的缓存
    cache_dir = get_cache_dir()
    modelscope_cache = os.path.join(cache_dir, 'modelscope', 'hub')
    
    # 检查主要模型文件是否存在
    required_models = [
        'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
    ]

    required_files = ['configuration.json', 'model.pt']
    base_candidates = [os.path.join(modelscope_cache, 'models'), modelscope_cache]

    all_exist = True
    for model_name in required_models:
        found = False
        for base_dir in base_candidates:
            model_path = os.path.join(base_dir, *model_name.split('/'))
            if os.path.isdir(model_path) and all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                found = True
                break
        if not found:
            all_exist = False
            break

    if all_exist:
        print(f"✅ 在 {modelscope_cache} 找到FunASR模型")
        return True, modelscope_cache
    
    return False, modelscope_cache

def create_symbolic_link(source, target):
    """创建符号链接"""
    try:
        if os.path.exists(target):
            if os.path.islink(target):
                os.unlink(target)
            else:
                shutil.rmtree(target)

        # 在Windows上使用junction或符号链接
        if os.name == 'nt':  # Windows
            # 尝试使用mklink创建符号链接
            try:
                subprocess.run(['mklink', '/J', target, source], check=True, shell=True)
                return True
            except:
                # 如果mklink失败，直接复制目录
                print(f"⚠️ 符号链接创建失败，直接复制目录")
                shutil.copytree(source, target)
                return True
        else:  # Linux/macOS
            os.symlink(source, target, target_is_directory=True)
            return True
    except Exception as e:
        print(f"❌ 创建符号链接失败: {e}")
        return False


def download_funasr_models():
    """下载FunASR模型 - 强制下载到程序目录"""
    print("正在下载FunASR语音识别模型...")
    print("这可能需要几分钟时间，请耐心等待...")
    print("如果下载失败，请检查网络连接和存储空间")

    # 强制设置缓存目录
    cache_dir = force_set_cache_env()
    modelscope_cache = os.path.join(cache_dir, 'modelscope', 'hub')

    try:
        # 导入必要的库
        print("正在导入FunASR库...")
        from funasr import AutoModel

        print("正在下载语音识别模型...")
        print(f"🔧 下载位置: {modelscope_cache}")
        
        # 下载主要模型
        model = AutoModel(
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            model_revision="v2.0.4",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_model_revision="v2.0.4",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            punc_model_revision="v2.0.4",
        )

        print("✅ FunASR模型下载完成！")

        # 验证模型完整性
        model_exists, actual_cache_dir = check_funasr_models()
        if model_exists:
            print(f"✅ 模型验证成功: {actual_cache_dir}")
            return True, actual_cache_dir
        else:
            print("⚠️ 模型下载完成但验证失败")
            return False, modelscope_cache

    except ImportError:
        print("❌ 错误：未找到FunASR库，请先安装：pip install funasr")
        return False, modelscope_cache
    except Exception as e:
        print(f"❌ FunASR模型下载失败: {str(e)}")
        print("💡 建议：")
        print("  1. 检查网络连接")
        print("  2. 确保有足够的存储空间（至少5GB）")
        print("  3. 尝试重新运行")
        return False, modelscope_cache

def check_embedding_model():
    """检查Qwen3 Embedding模型是否已下载"""
    cache_dir = get_cache_dir()
    modelscope_cache = os.path.join(cache_dir, 'modelscope', 'hub')

    qwen_dir = os.path.join(modelscope_cache, 'Qwen')
    if not os.path.isdir(qwen_dir):
        return False, None

    candidates = []
    try:
        for name in os.listdir(qwen_dir):
            full_path = os.path.join(qwen_dir, name)
            if not os.path.isdir(full_path):
                continue
            normalized = name.replace('_', '').replace('-', '').replace('.', '').lower()
            if 'qwen3embedding' in normalized:
                candidates.append(full_path)
    except Exception:
        return False, None

    for model_path in candidates:
        if os.path.exists(os.path.join(model_path, 'model.safetensors')) and os.path.exists(os.path.join(model_path, 'config.json')):
            print(f"✅ 找到Qwen3 Embedding模型: {model_path}")
            return True, model_path

    return False, None

def download_embedding_model():
    """下载Qwen3 Embedding模型 - 强制下载到程序目录（参考FunASR方式）"""
    print("正在下载Qwen3 Embedding向量模型...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    # 强制设置缓存目录
    cache_dir = force_set_cache_env()
    modelscope_cache = os.path.join(cache_dir, 'modelscope', 'hub')
    
    try:
        # 导入modelscope
        print("正在导入ModelScope库...")
        from modelscope import snapshot_download
        
        # 使用Qwen3-Embedding-0.6B模型
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        
        print(f"🔧 下载位置: {modelscope_cache}")
        print(f"正在下载 {model_name} ...")
        
        model_path = snapshot_download(
            model_name,
            cache_dir=modelscope_cache
        )
        
        print(f"✅ Qwen3 Embedding模型下载完成: {model_path}")
        return True, model_path
    
    except ImportError:
        print("❌ 错误：未找到ModelScope库，请先安装：pip install modelscope")
        return False, None
    except Exception as e:
        print(f"❌ 模型下载失败: {str(e)}")
        print("💡 建议：")
        print("  1. 检查网络连接")
        print("  2. 确保有足够的存储空间（至少2GB）")
        print("  3. 将使用备用的英文模型")
        return False, None

def check_and_install():
    """检查环境并安装必要组件"""
    print("=" * 60)
    print("视频智能分析处理套件 v3.0 - 环境检查")
    print("=" * 60)

    # 首先强制设置缓存目录
    print("\n初始化缓存目录...")
    cache_dir = force_set_cache_env()
    print(f"✅ 所有模型将存储在: {cache_dir}")

    # 检查FFmpeg
    print("\n检查FFmpeg...")
    
    # 项目内置FFmpeg路径
    builtin_ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ffmpeg_downlaod', 'bin', 'ffmpeg.exe')
    
    # 首先检查项目内置的FFmpeg
    if os.path.exists(builtin_ffmpeg_path):
        try:
            subprocess.run([builtin_ffmpeg_path, '-version'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True)
            print(f"✅ 使用项目内置FFmpeg: {builtin_ffmpeg_path}")
        except:
            print(f"⚠️ 项目内置FFmpeg不可用: {builtin_ffmpeg_path}")
            # 继续检查系统PATH中的FFmpeg
            try:
                subprocess.run(['ffmpeg', '-version'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               check=True)
                print("✅ 使用系统PATH中的FFmpeg")
            except:
                print("❌ 未检测到可用的FFmpeg")
                print("💡 请先运行 setup.bat 安装FFmpeg")
                return False
    else:
        # 检查系统PATH中的FFmpeg
        try:
            subprocess.run(['ffmpeg', '-version'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=True)
            print("✅ 使用系统PATH中的FFmpeg")
        except:
            print("❌ 未检测到可用的FFmpeg")
            print("💡 请先运行 setup.bat 安装FFmpeg")
            return False

    # 检查API密钥
    print("\n检查阿里云API密钥...")
    if not os.environ.get('DASHSCOPE_API_KEY'):
        print("⚠️ 未检测到DASHSCOPE_API_KEY环境变量")
        print("💡 请先运行 setup.bat 设置API密钥")
        print("💡 注意：缺少API密钥将影响部分功能的使用")
    else:
        print("✅ DASHSCOPE_API_KEY已配置")

    # 检查FunASR模型
    print("\n检查FunASR语音识别模型...")
    model_exists, cache_location = check_funasr_models()

    if not model_exists:
        print("未检测到FunASR模型，开始下载...")
        download_success, actual_cache_dir = download_funasr_models()
        if not download_success:
            print("❌ FunASR模型下载失败，语音识别功能可能无法使用")
        else:
            print("✅ FunASR模型下载成功")
    else:
        print("✅ FunASR模型已存在")

    # 检查Qwen3 Embedding模型
    print("\n检查Qwen3 Embedding向量模型...")
    embedding_exists, embedding_path = check_embedding_model()

    if not embedding_exists:
        print("未检测到Qwen3 Embedding模型，开始下载...")
        download_success, embedding_path = download_embedding_model()
        if not download_success:
            print("❌ Qwen3 Embedding模型下载失败，RAG功能将使用备用英文模型")
        else:
            print("✅ Qwen3 Embedding模型下载成功")
    else:
        print("✅ Qwen3 Embedding模型已存在")

    print("\n" + "=" * 60)
    print("环境检查完成，正在启动应用...")
    print("=" * 60)
    return True

if __name__ == "__main__":
    # 首先清除旧的缓存文件夹
    print("=" * 60)
    print("🗑️ 清理旧的缓存文件...")
    print("=" * 60)
    clear_old_cache()
    
    # 然后强制设置环境变量（在检查之前）
    print("\n🔧 强制设置环境变量...")
    cache_dir = force_set_cache_env()
    print(f"✅ 缓存目录已设置: {cache_dir}")
    
    # 检查并安装必要组件
    if check_and_install():
        # 再次强制设置环境变量，确保streamlit子进程也能继承
        cache_dir = force_set_cache_env()
        print(f"\n🔧 最终确认缓存目录: {cache_dir}")
        
        # 显示所有关键环境变量
        print("\n📋 环境变量列表:")
        print(f"  MODELSCOPE_CACHE_DIR = {os.environ.get('MODELSCOPE_CACHE_DIR')}")
        print(f"  HF_HOME = {os.environ.get('HF_HOME')}")
        print(f"  TORCH_HOME = {os.environ.get('TORCH_HOME')}")
        
        if getattr(sys, 'frozen', False):
            current_dir = sys._MEIPASS
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "app.py")

        # 启动Streamlit应用
        sys.argv = ["streamlit", "run", file_path, 
            "--server.enableCORS=true", "--server.enableXsrfProtection=false", 
            "--global.developmentMode=false", "--client.toolbarMode=minimal",
            "--server.maxUploadSize=65536"]  # 上传限制为64GB
        
        sys.exit(stcli.main())
