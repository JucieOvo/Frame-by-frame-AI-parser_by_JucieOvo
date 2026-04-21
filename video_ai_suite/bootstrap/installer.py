#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名称：installer
功能描述：
    项目的正式安装与环境配置模块。
    该模块承接根目录 setup.py 的全部安装职责，包括依赖安装、API 密钥配置与启动脚本生成。

主要组件：
    - main: 安装主入口。
    - install_requirements: 安装项目依赖。
    - configure_api_key: 配置 API 密钥。

依赖说明：
    - Python 标准库: subprocess、pathlib、winreg、platform。

作者：JucieOvo
创建日期：2026-04-20
修改记录：
    - 2026-04-20 JucieOvo: 从根目录 setup.py 迁移正式安装职责到包内。
"""

import os
import sys
import subprocess
import platform
import winreg
from pathlib import Path

# 项目根目录用于统一解析 requirements 和生成的启动脚本位置，避免模块迁移后仍依赖当前工作目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ANSI颜色代码（Windows 10+支持）
class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'

def print_header(text):
    """打印标题"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text):
    """打印信息"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def run_command(cmd, description="", check=True, capture_output=False):
    """运行命令"""
    try:
        if description:
            print(f"正在{description}...")
        
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            return result.returncode == 0, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0, "", ""
    except subprocess.CalledProcessError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", str(e)

def check_python_version():
    """检查Python版本"""
    print_header("第1步：检查Python环境")
    
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    print(f"Python 路径: {sys.executable}")
    
    if version.major == 3 and version.minor >= 8:
        print_success("Python 版本符合要求 (>= 3.8)")
        return True
    else:
        print_error(f"Python 版本过低，需要 3.8+，当前版本: {version.major}.{version.minor}")
        return False

def upgrade_pip():
    """升级pip"""
    print_header("第2步：升级pip")
    
    success, stdout, stderr = run_command(
        f'"{sys.executable}" -m pip install --upgrade pip -q',
        "升级pip"
    )
    
    if success:
        print_success("pip 已升级到最新版本")
    else:
        print_warning("pip 升级失败，继续使用当前版本")
    
    return True

def install_pytorch():
    """安装PyTorch CPU版本"""
    print_header("第3步：安装 PyTorch CPU 版本")
    
    # 检查是否已安装
    success, _, _ = run_command(
        f'"{sys.executable}" -c "import torch"',
        capture_output=True
    )
    
    if success:
        # 获取版本
        _, version, _ = run_command(
            f'"{sys.executable}" -c "import torch; print(torch.__version__)"',
            capture_output=True
        )
        print_success(f"PyTorch 已安装，版本: {version.strip()}")
        return True
    
    print("开始安装 PyTorch CPU 版本（约500MB）...")
    print_info("使用清华镜像源加速下载")
    print()
    
    # 安装CPU版本
    success, _, stderr = run_command(
        f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu',
        "安装PyTorch"
    )
    
    if not success:
        print_warning("官方源失败，尝试清华镜像...")
        success, _, _ = run_command(
            f'"{sys.executable}" -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple',
            "安装PyTorch（镜像源）"
        )
    
    if success:
        print_success("PyTorch CPU 版本安装成功！")
        return True
    else:
        print_error("PyTorch 安装失败")
        print_info("请手动安装: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        return False

def install_requirements():
    """安装requirements.txt中的依赖"""
    print_header("第4步：安装依赖包")
    
    req_file = PROJECT_ROOT / "requirements.txt"
    
    if not req_file.exists():
        print_error("未找到 requirements.txt 文件")
        return False
    
    print(f"从 {req_file} 安装所有依赖...")
    print_info("使用清华大学镜像源加速下载")
    print()
    
    success, _, _ = run_command(
        f'"{sys.executable}" -m pip install -r "{req_file}" -i https://pypi.tuna.tsinghua.edu.cn/simple',
        "安装依赖包"
    )
    
    if not success:
        print_warning("清华镜像失败，尝试官方源...")
        success, _, _ = run_command(
            f'"{sys.executable}" -m pip install -r "{req_file}"',
            "安装依赖包（官方源）"
        )
    
    if success:
        print_success("requirements.txt 依赖安装完成！")
    else:
        print_warning("部分依赖可能安装失败，程序运行时会提示")
    
    return True

def configure_api_key():
    """配置API密钥"""
    print_header("第5步：配置阿里云API密钥")
    
    print("阿里云API密钥用于语音识别和向量模型功能")
    print("如果没有密钥，可以跳过此步骤")
    print()
    
    # 检查是否已设置
    existing_key = os.environ.get('DASHSCOPE_API_KEY')
    if existing_key:
        print_info(f"检测到已有API密钥: {existing_key[:8]}...")
        choice = input("是否重新设置？(Y/N): ").strip().upper()
        if choice != 'Y':
            print_info("跳过API密钥设置")
            return True
    
    choice = input("是否设置阿里云API密钥？(Y/N): ").strip().upper()
    
    if choice == 'Y':
        api_key = input("请输入阿里云API密钥: ").strip()
        
        if api_key:
            try:
                # 设置用户环境变量
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r'Environment',
                    0,
                    winreg.KEY_WRITE
                )
                winreg.SetValueEx(key, 'DASHSCOPE_API_KEY', 0, winreg.REG_SZ, api_key)
                winreg.CloseKey(key)
                
                # 广播更改
                import ctypes
                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x1A
                ctypes.windll.user32.SendMessageTimeoutW(
                    HWND_BROADCAST, WM_SETTINGCHANGE, 0, 'Environment', 0, 1000, None
                )
                
                print_success("API密钥设置成功")
                print_info("提示：新变量将在新打开的终端中生效")
                
                # 在当前进程中也设置
                os.environ['DASHSCOPE_API_KEY'] = api_key
                
            except Exception as e:
                print_error(f"API密钥设置失败: {e}")
                print()
                print("您可以手动设置：")
                print("1. 右键\"此电脑\" - \"属性\"")
                print("2. \"高级系统设置\" - \"环境变量\"")
                print("3. 在\"用户变量\"中新建：")
                print("   变量名: DASHSCOPE_API_KEY")
                print(f"   变量值: {api_key}")
        else:
            print_warning("未输入API密钥，跳过设置")
    else:
        print_info("跳过API密钥设置")
    
    return True

def create_run_script():
    """创建启动脚本"""
    print_header("第6步：创建启动脚本")
    
    # 创建run.bat
    bat_content = f"""@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
echo ========================================
echo   视频智能分析套件 v3.0
echo ========================================
echo.
echo 启动程序...
echo.
"{sys.executable}" "%SCRIPT_DIR%run.py"

if %errorlevel% neq 0 (
    echo.
    echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    echo   程序异常退出
    echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    echo.
    echo 可能的原因：
    echo   1. 依赖包未正确安装
    echo   2. 缺少必要的环境变量
    echo   3. 程序运行出错
    echo.
    pause
    exit /b 1
)
"""
    
    try:
        with open(PROJECT_ROOT / "run.bat", "w", encoding="utf-8") as f:
            f.write(bat_content)
        print_success("已创建 run.bat")
        return True
    except Exception as e:
        print_error(f"创建启动脚本失败: {e}")
        return False

def main():
    """主函数"""
    # 启用ANSI颜色支持（Windows 10+）
    if platform.system() == 'Windows':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 60)
    print("   视频智能分析套件 v3.0")
    print("   Python 环境配置程序")
    print("=" * 60)
    print(f"{Colors.RESET}")
    print()
    print("本程序将自动完成：")
    print("  1. 检查 Python 环境")
    print("  2. 升级 pip")
    print("  3. 安装 PyTorch CPU 版本")
    print("  4. 安装所有依赖包")
    print("  5. 配置 API 密钥")
    print("  6. 创建启动脚本")
    print()
    print_warning("全程需要10-30分钟，请确保网络连接稳定")
    print()
    
    input("按回车键开始...")
    
    try:
        # 执行所有步骤
        if not check_python_version():
            print_error("Python 版本不符合要求，程序退出")
            input("\n按回车键退出...")
            return 1
        
        upgrade_pip()
        
        if not install_pytorch():
            print_warning("PyTorch 安装失败，但继续执行")
        
        install_requirements()

        configure_api_key()
        
        create_run_script()
        
        # 完成
        print()
        print(f"{Colors.BOLD}{Colors.GREEN}")
        print("=" * 60)
        print("   🎉 环境配置完成！")
        print("=" * 60)
        print(f"{Colors.RESET}")
        print()
        print_success("依赖已安装到系统 Python 环境")
        print_success("启动脚本已生成: run.bat")
        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  使用说明")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print("1. 双击 run.bat 启动程序")
        print("2. 首次运行会自动下载模型（约 5GB）")
        print("3. 模型会保存在 .cache 文件夹")
        print("4. 后续启动只需 1-2 分钟")
        print()
        
        # 不自动启动，避免循环
        print()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  安装完成！")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print()
        print("💡 下一步：双击 run.bat 启动程序")
        print()
        print("⚠️  提示：")
        print("   - 首次运行需要下载模型（约5GB，10-30分钟）")
        print("   - 请保持网络连接稳定")
        print()
        
        input("按回车键退出...")
        return 0
        
    except KeyboardInterrupt:
        print()
        print_warning("用户取消操作")
        input("\n按回车键退出...")
        return 1
    except Exception as e:
        print()
        print_error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
