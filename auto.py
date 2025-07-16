import os
import glob
import base64
import time
import subprocess  # 添加subprocess模块用于调用ffmpeg
from tqdm import tqdm
from openai import OpenAI

lezi = """
免责声明
本软件（“[视频切分转文字程序]”）按“原样”提供，不提供任何明示或暗示的担保，包括但不限于对适销性、特定用途适用性和不侵权的暗示担保。在任何情况下，作者或版权所有者均不对因软件或软件的使用或其他交易而产生、由软件引起或与之相关的任何索赔、损害或其他责任（无论是合同、侵权还是其他形式的责任）承担任何责任，即使事先被告知此类损害的可能性。
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
1. 本程序需要用户自行在本地设置 DASHSCOPE_API_KEY 环境变量
2. 该密钥仅存储在用户本地环境变量中，程序运行时仅在内存中临时读取
3. 本程序不会以任何形式:
   - 将API密钥传输到外部服务器
   - 将API密钥写入日志/文件
   - 持久化存储API密钥
4. 用户需自行保管好API密钥，本程序开发者不承担因密钥泄露导致的任何责任

使用本程序即表示您同意:
- 您是该API密钥的合法持有者
- 您已了解密钥泄露的风险
- 您自愿承担使用该API密钥的所有责任
__________
欢迎使用 视频切分转文字v1.00 程序😁😁😁
本程序开源免费，如果你花钱购买本程序，请诉诸法律
by CN_榨汁Ovo  愿世界和平
"""
print(lezi)
time.sleep(15)

caozuoliucheng = """
1、请将视频命名为input.mp4后将其与本程序置于同一目录[建议置于同一文件夹，否则可能导致桌面出现大量临时文件或导致可能的程序错误]
2、运行本程序
3、本程序运行成功后会有一个[草稿.txt]出现在同一目录下，这是对每x帧的分析结果，用户可将[草稿.txt]上传到支持长上下文的LLM进行整理，本程序暂不提供此功能，可等待后续版本更新
"""
print(caozuoliucheng)
time.sleep(5)

# 检查环境变量中的API密钥
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    error_msg = """
    ============================================
    错误: 未找到 DASHSCOPE_API_KEY 环境变量
    ============================================
    将自动为您运行 apikey轻松配 程序
    请按照程序内提示配置您的apikey
    注意: 此API密钥是程序运行的必要条件!您的API_key储存在本地环境变量中，本程序采用https加密API_key不会造成任何形式的API_key泄露！且不为此承担任何法律责任
    ============================================
    """
    print(error_msg)
    time.sleep(3)
    subprocess.Popen("ffmpeg_auto_download.exe")
    

# 创建OpenAI客户端
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def split_video_to_frames(fps):
    """使用ffmpeg将视频切分为图片序列"""
    print(f"\n开始切分视频，帧率: {fps} fps...")
    cmd = [
        'ffmpeg',
        '-i', 'input.mp4',          # 输入文件
        '-vf', f'fps={fps}',        # 设置帧率
        '-f', 'image2',             # 输出为图片序列
        '-c:v', 'png',              # 使用PNG编码
        '-compression_level', '1',  # 压缩级别1（速度较快）
        '%04d.png'                  # 输出文件名格式
    ]
    
    try:
        # 运行ffmpeg命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("视频切分完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"视频切分失败: {e.stderr}")
        return False
    except FileNotFoundError:
        print("未找到ffmpeg，请确保ffmpeg已安装并添加到系统PATH，将自动为您运行ffmpeg安装程序")
        subprocess.Popen("ffmpeg_auto_download.exe")
        return False

def read_image_as_base64(image_path):
    """读取图片并转换为Base64编码"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def get_sorted_image_files():
    """获取当前目录下按编号排序的图片文件列表（支持JPG、JPEG和PNG）"""
    image_files = []
    for ext in ('.jpg', '.jpeg', '.png'):
        image_files.extend(glob.glob(f"*{ext}"))
    
    filtered_files = []
    for file in image_files:
        try:
            num = int(os.path.splitext(file)[0])
            filtered_files.append((num, file))
        except ValueError:
            continue
    
    sorted_files = sorted(filtered_files, key=lambda x: x[0])
    return [file for _, file in sorted_files]

def get_mime_type(filename):
    """根据文件扩展名返回对应的 MIME 类型"""
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    elif ext == '.png':
        return 'image/png'
    else:
        return 'application/octet-stream'

def process_image_group(group, output_file, max_retries=3):
    """处理一组图片并保存结果（带重试机制）"""
    print(f"\n处理图片组: {group}")
    
    # 准备图片内容
    image_file = group[0]
    mime_type = get_mime_type(image_file)
    base64_image = read_image_as_base64(image_file)
    image_url = f"data:{mime_type};base64,{base64_image}"

    # 构造消息（兼容OpenAI格式）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "请详细解析这张图片的内容，描述其主要内容"}
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # 使用OpenAI客户端调用API
            completion = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=messages,
                stream=False
            )

            # 提取结果
            result = completion.choices[0].message.content
            processing_time = time.time() - start_time

            # 保存结果到文件
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"图片组: {group}\n")
                f.write(f"处理时间: {processing_time:.1f}秒\n")
                f.write("解析结果:\n")
                f.write(result.strip() + "\n")
                f.write("-" * 80 + "\n\n")

            print(f"✅ 成功处理 ({processing_time:.1f}秒)")
            return True
            
        except Exception as e:
            print(f"⚠️ 处理错误: {str(e)}")
            # 指数退避重试
            sleep_time = min(2 ** attempt, 30)
            print(f"⏳ 等待 {sleep_time} 秒后重试...")
            time.sleep(sleep_time)

    # 所有重试失败
    print(f"❌ 处理失败: {group}")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"图片组 {group} 处理失败\n\n")
    return False

def main():
    output_file = "草稿.txt"

    # 检查视频文件是否存在
    if not os.path.exists("input.mp4"):
        print("错误: 未找到input.mp4文件")
        print("请将视频文件命名为input.mp4并放在本程序同一目录下")
        return False
    
    # 检查是否已有图片
    if not get_sorted_image_files():
        # 询问用户切分帧率
        try:
            fps = float(input("\n请输入视频切分的帧率[每秒x帧](fps): "))
            if fps <= 0:
                print("错误: 帧率必须是正数")
                return False
        except ValueError:
            print("错误: 请输入有效的数字")
            return False
        
        # 切分视频
        if not split_video_to_frames(fps):
            print("视频切分失败，程序终止")
            return False
        time.sleep(1)  # 等待文件系统更新
    
    # 初始化输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"图片解析报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    # 获取排序后的图片列表
    sorted_images = get_sorted_image_files()
    if not sorted_images:
        print("未找到任何图片 (支持格式: .jpg, .png .jpeg)")
        return False

    print(f"找到 {len(sorted_images)} 张图片，开始处理...")

    # 单张图片分组
    total_groups = len(sorted_images)
    for i in tqdm(range(total_groups), desc="处理进度"):
        group = [sorted_images[i]]
        process_image_group(group, output_file)
        time.sleep(1)  # 请求间隔

if __name__ == "__main__":
    main()