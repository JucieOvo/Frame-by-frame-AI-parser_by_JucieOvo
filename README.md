Frame-by-frame AI Parser (视频解析工具包)
<p align="center"> <strong>🚀 智能视频内容解析工具 | 将视频自动转换为结构化文字描述</strong> </p><p align="center"> <img src="https://img.shields.io/badge/Release-v2.00-green" alt="Release"> <img src="https://img.shields.io/badge/Dependency-FFmpeg-important" alt="FFmpeg"> <img src="https://img.shields.io/badge/Powered%20By-AI%20Models-brightgreen" alt="AI Powered"> </p>
Frame-by-frame-AI-parser 是一款集视频处理、AI图像分析和环境自动配置于一体的智能工具包。它能够自动解析视频内容，通过先进的视觉大模型（VLM）将每一帧画面转换为连贯、精准的自然语言描述，极大地简化了视频内容分析、生成字幕和场景摘要的工作流。

✨ 核心特性
🤖 多模型支持：无缝接入 阿里通义千问（Qwen-VL） 等云端大模型，同时支持本地部署的 Ollama 模型，兼顾性能与隐私。

🎬 智能视频抽帧：采用差分算法自动过滤静态或重复的无意义帧，只提取关键画面，显著提升处理效率与解析质量。

🌐 现代化Web界面：基于 Streamlit 构建直观、优雅的图形化操作界面，告别繁琐的命令行操作，提供流畅的用户体验。

⚙️ 一站式环境配置：运行前自动检测系统环境，无需手动干预，即可自动安装配置 FFmpeg 并验证 API 密钥，真正做到开箱即用。

📄 高质量文本生成：生成包含场景、物体、动作、人物关系等关键信息的详细且连贯的文字描述。

🛠️ 工作原理
输入视频：通过Web界面上传您的视频文件。

智能抽帧：程序利用FFmpeg和差分算法提取关键帧图像。

AI 解析：将帧图像发送至配置好的视觉大模型进行深度分析。

生成描述：模型返回对每帧画面的文字理解。

输出结果：最终将所有帧的描述组合成完整的、按时间线排列的文字报告。

📦 快速开始
直接运行（推荐）
前往 Release页面 下载最新的 auto.exe 可执行文件。

双击运行，程序将自动检测并配置所需环境（FFmpeg）。

根据引导，在自动打开的Web页面中配置您的API密钥（如需使用云端模型）。

上传视频，开始解析！

从源码运行
如果您希望贡献代码或进行修改，请从源码run.py运行：

bash
# 1. 克隆本仓库
git clone https://github.com/JucieOvo/Frame-by-frame-AI-parser_by_JucieOvo.git
cd Frame-by-frame-AI-parser_by_JucieOvo

# 2. 启动主程序
python run.py
📋 使用前提
操作系统: Windows 10/11

网络连接: 用于下载依赖和调用AI API。

API 密钥 (可选): 如果您使用云端模型（如Qwen-VL），需要提前准备相应的API Key。程序内会引导您配置。

🆚 版本历史
v2.00 (最新版本)： 引入智能差分抽帧、全新的Streamlit Web UI、支持Ollama本地模型。

v1.00： 初始版本，提供基础的视频拆帧、Qwen-VL API调用和自动化环境配置功能。

🤝 贡献
我们欢迎任何形式的贡献！无论是提交 Bug、提出新功能建议，还是直接发起 Pull Request，都非常感谢。

📄 许可证
本项目基于 MIT License 开源。
