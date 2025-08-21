import os
import shutil
import ctypes
import sys
import winreg
import time
lezi = """
免责声明
本软件（“[ffmpeg傻瓜式自动安装程序]”）按“原样”提供，不提供任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性和不侵权的暗示担保。在任何情况下，作者或版权所有者均不对因软件或软件的使用或其他交易而产生、由软件引起或与之相关的任何索赔、损害或其他责任（无论是合同、侵权还是其他形式的责任）承担任何责任，即使事先被告知此类损害的可能性。
重要提示
本软件可能存在错误、缺陷或不完善之处。
作者不保证软件是：
无错误的。
不间断或可用的。
安全的（不会导致数据丢失、系统损坏或安全漏洞）。
符合你的特定需求或期望。
在法律上、技术上或商业上可行的。
用户自担风险： 你使用、修改、分发本软件或依赖本软件的行为完全由你自己承担风险。你应对使用软件可能导致的任何及所有后果负责，包括但不限于：
数据丢失或损坏。
系统故障或中断。
业务中断。
安全漏洞或数据泄露。
财务损失。
任何其他直接、间接、附带、特殊、后果性或惩罚性损害。
第三方依赖： 本软件可能依赖其他第三方库、服务或组件（统称“依赖项”）。这些依赖项有其自身的许可证和免责声明。本项目的作者不对任何依赖项的功能、安全性、可靠性或合法性负责或提供担保。 你需要自行审查并遵守所有依赖项的条款。
非专业建议： 如果本软件涉及特定领域（如金融、医疗、安全等），其输出或功能不应被视为专业建议。在做出任何依赖软件输出的决策之前，请务必咨询该领域的合格专业人士。
贡献者： 本软件可能包含由社区贡献者提交的代码。项目维护者（作者）会尽力审查贡献，但不保证所有贡献的代码都是安全、无错误或合适的。接受贡献并不意味着维护者对其承担额外的责任。
你的责任
作为软件的用户（或修改者、分发者），你有责任：
在使用前仔细评估软件是否适合你的目的。
在非生产环境中进行充分的测试。
实施适当的安全措施和数据备份。
遵守软件所使用的开源许可证的所有条款。
遵守所有适用的法律和法规。
总结
使用本软件即表示你理解并完全接受本免责声明中的所有条款和风险。如果你不同意这些条款，请不要使用、修改或分发本软件。
本程序无任何政治目的，没有任何政治影射
__________
欢迎使用ffmpeg傻瓜式自动安装程序😁😁😁
本程序开源免费，如果你花钱购买本程序，请诉诸法律
在开始之前，请确保配套的ffmpeg_downlaod文件夹与本程序在同一目录
by CN_榨汁Ovo  愿世界和平
"""
print(lezi)
time.sleep(15)
def is_admin():
    """检查是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def move_folder():
    """移动文件夹并设置环境变量"""
    source_folder = "ffmpeg_downlaod"
    target_base = r"C:\Program Files (x86)"
    target_folder = os.path.join(target_base, source_folder)
    bin_path = os.path.join(target_folder, "bin")
    
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：当前目录下未找到 '{source_folder}' 文件夹")
        return
    
    # 创建目标父目录（如果不存在）
    os.makedirs(target_base, exist_ok=True)
    
    # 如果目标文件夹已存在则删除
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"已移除旧文件夹: {target_folder}")
    
    # 移动文件夹
    shutil.move(source_folder, target_base)
    print(f"成功移动文件夹到: {target_folder}")
    
    # 验证bin目录是否存在
    if not os.path.exists(bin_path):
        print(f"警告：目标目录中未找到 'bin' 文件夹: {bin_path}")
        return
    
    # 获取当前用户环境变量PATH
    reg_key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        "Environment",
        0,
        winreg.KEY_READ | winreg.KEY_WRITE
    )
    
    try:
        path_value, _ = winreg.QueryValueEx(reg_key, "PATH")
    except FileNotFoundError:
        path_value = ""
    
    # 添加新路径到PATH
    new_paths = [p.strip() for p in path_value.split(os.pathsep) if p.strip()]
    if bin_path not in new_paths:
        new_paths.append(bin_path)
        updated_path = os.pathsep.join(new_paths)
        
        # 更新注册表
        winreg.SetValueEx(reg_key, "PATH", 0, winreg.REG_EXPAND_SZ, updated_path)
        winreg.CloseKey(reg_key)
        
        # 广播环境变量变更
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x1A, 0, "Environment", 0, 1000, None)
        print(f"成功添加PATH变量: {bin_path}")
    else:
        print("PATH中已存在该路径，无需添加")
        winreg.CloseKey(reg_key)

if __name__ == "__main__":
    # 请求管理员权限
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(
            None, "runas", sys.executable, " ".join(sys.argv), None, None, 1
        )
        sys.exit()
    
    # 执行主操作
    try:
        move_folder()
        print("操作完成！请重启命令行使环境变量生效")
    except Exception as e:
        print(f"操作失败: {str(e)}")
        input("按回车键退出...")