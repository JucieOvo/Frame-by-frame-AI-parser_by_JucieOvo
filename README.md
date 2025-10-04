# 🎬 视频智能分析处理套件 v3.0

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**基于 AI 的智能视频分析工具**

支持关键帧提取 | 图像分析 | 语音识别 | RAG 智能检索

[安装指南](#-快速安装) • [使用文档](#-使用说明) • [常见问题](#-常见问题) • [更新日志](#-更新日志)

</div>

---

## 📖 项目简介

视频智能分析处理套件是一个基于 AI 的综合视频分析工具，能够自动提取关键帧、分析图像内容、识别语音，并提供基于 RAG（检索增强生成）的智能问答功能。

### ✨ 核心特性

- 🎥 **智能关键帧提取** - 基于场景变化自动检测关键帧
- 🖼️ **多模态图像分析** - 支持阿里云 Qwen-VL 和本地 Ollama 模型
- 🎤 **高精度语音识别** - 基于本地 FunASR 的中文语音转录
- 🔍 **RAG 智能检索** - 使用本地 Qwen3 Embedding 的语义搜索
- 📊 **多维度分析报告** - 完整报告、速读摘要、智能问答
- 🦺 **CPU 支持** - 无需 GPU 也能运行，简化依赖配置
- 🚀 **流水线并行处理** - 抽帧、语音识别、图片分析同时进行，性能提升 30-69%
- 🌐 **可视化界面** - 基于 Streamlit 的现代化 Web 界面

---

## 🚀 快速安装

### 方式一：自动安装（推荐）⭐

**Windows 10/11 用户（推荐）：**

```bash
# 双击运行安装脚本
运行环境配置.bat
几次同意与输入apikey
等待环境配置完成...

# 启动程序
run.bat
```

**预计时间**: 15 分钟内（首次安装）

> 💡 **推荐使用批处理版本** - 自动配置环境，简化手动操作

---

### 方式二：手动安装

<details>
<summary>点击展开详细步骤</summary>

#### 1. 下载项目文件

下载项目压缩包并解压到指定目录。

#### 2. 安装 Python 环境

如果系统未安装 Python，请先安装 Python 3.8+：
- 下载地址: https://www.python.org/downloads/
- 安装时勾选 "Add Python to PATH"

#### 3. 安装依赖包

```bash
# 使用国内镜像（推荐，更快）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用官方源
pip install -r requirements.txt
```

#### 4. 配置 FFmpeg

项目已内置 FFmpeg，位于 `ffmpeg_downlaod/bin/ffmpeg.exe`，程序会自动调用。

#### 5. 配置 API 密钥（使用阿里云模型时）

```bash
# 设置环境变量
setx DASHSCOPE_API_KEY "your-api-key-here"
```

#### 6. 运行程序

```bash
python run.py
```

</details>

---

## 📋 系统要求

### 必需配置
- ✅ Windows 10/11 (64位)
- ✅ Python 3.8+
- ✅ 8GB+ 内存
- ✅ 15GB+ 磁盘空间
- ✅ FFmpeg（已内置）

### GPU 加速（可选）
- ✅ NVIDIA GPU（支持CUDA）
- ✅ 兼容的显卡驱动
- ✅ 兼容的CUDA驱动

---

## 💡 使用说明

### 基础工作流

1. **启动程序** → 运行 `run.bat` 或以源码运行 `python run.py`
2. **上传视频** → 支持 MP4, AVI, MOV, MKV 等格式
3. **配置模型** → 选择阿里云或 Ollama 模型
4. **开始分析** → 自动提取关键帧并分析
5. **查看结果** → 实时查看分析进度和结果
6. **智能检索** → 切换到"多方式分析"使用 RAG 功能

### 核心功能模块

#### 🎥 视频分析模块
- **功能描述**: 自动场景检测和关键帧提取
- **技术实现**: 使用 PySceneDetect 进行场景检测
- **输出结果**: 提取的关键帧图像和场景分析报告

#### 🎤 语音识别模块
- **功能描述**: 自动提取视频音频并进行中文语音转录
- **技术实现**: 基于 FunASR Paraformer 模型
- **输出结果**: 完整的语音文字转录文本

#### 🔍 RAG 智能检索模块
- **功能描述**: 基于语义的智能问答和内容检索
- **技术实现**: LangChain + ChromaDB + Qwen3 Embedding
- **检索策略**: 
  - **相似度搜索** - 默认策略，速度快
  - **MMR 搜索** - 减少冗余，结果多样化
  - **阈值搜索** - 精准匹配，过滤低相关度

### 分析报告类型

#### 📊 完整分析报告
- **特点**: 连贯的视频内容总结
- **适用场景**: 需要全面了解视频内容的场景
- **输出格式**: 结构化的详细报告

#### 📝 速读摘要
- **特点**: 快速了解核心内容
- **适用场景**: 时间有限，需要快速浏览
- **输出格式**: 简洁的要点总结

#### 🔍 智能检索问答
- **特点**: 基于语义的精准问答
- **适用场景**: 针对特定问题的查询
- **输出格式**: 带引用的问答结果

---

## ⚡ 性能提升

### 流水线并行处理架构

v3.0 引入了革命性的**流水线并行处理**技术，大幅提升视频分析效率：

#### 优化前（串行处理）
```
抽取关键帧（100%）→ 等待ASR（100%）→ 分析图片（100%）
总时间 = 抽帧时间 + ASR时间 + 分析时间
```

#### 优化后（流水线并行）
```
抽帧线程：  ████████████████████
ASR线程：   ████████████████████████
分析线程：     ████████████████████████
总时间 ≈ max(三者中最长的)
```

#### 性能提升数据

| 视频长度 | 关键帧数 | 优化前耗时 | 优化后耗时 | 性能提升 |
|---------|---------|-----------|-----------|---------|
| 5分钟   | 40帧    | 8分钟     | 3分钟     | **62%** ⬇️ |
| 10分钟  | 80帧    | 13分钟    | 4分钟     | **69%** ⬇️ |
| 20分钟  | 150帧   | 22分钟    | 8分钟     | **64%** ⬇️ |

#### 核心优势

- 🚀 **边抽边传** - 关键帧提取完成即刻上传分析
- ⚡ **三线程并行** - 抽帧、ASR、图片分析同时进行
- 💪 **资源充分利用** - CPU（抽帧）、网络（上传）、CPU/GPU（ASR）同时工作
- 🎯 **智能队列管理** - 生产者-消费者模式，避免内存溢出
- 🛡️ **智能模式切换** - 阿里云API使用并行，Ollama本地模型使用串行

---

## 📊 技术架构

### 核心框架
- **Web 界面**: Streamlit - 现代化 Web 应用框架
- **深度学习**: PyTorch (CPU/GPU) - 深度学习框架
- **视频处理**: OpenCV, PySceneDetect, FFmpeg - 视频处理工具链
- **语音识别**: FunASR (ModelScope) - 中文语音识别引擎

### AI 模型集成
- **多模态模型**: Qwen3-VL-Plus / Ollama 视觉模型
- **文本模型**: Qwen3-Plus / Ollama 文本模型
- **语音识别**: FunASR Paraformer - 专业中文语音识别
- **Embedding**: Qwen3-Embedding-0.6B - 中文语义向量模型

### RAG 技术栈
- **向量数据库**: ChromaDB - 轻量级向量数据库
- **Embedding 框架**: LangChain + Sentence Transformers
- **文本分割**: RecursiveCharacterTextSplitter - 智能文本分割

---

## 🎯 应用场景

### 📹 视频内容审核
- **应用描述**: 快速了解视频内容，进行内容审核
- **技术优势**: 自动提取关键信息，提高审核效率

### 📝 会议记录整理
- **应用描述**: 提取会议要点和语音内容
- **技术优势**: 自动转录和摘要，减少人工整理时间

### 🎓 教学视频分析
- **应用描述**: 提取关键知识点和教学内容
- **技术优势**: 智能分析教学重点，辅助学习

### 📺 媒体内容分析
- **应用描述**: 分析新闻、纪录片等内容
- **技术优势**: 深度内容理解，提取核心信息

### 🔍 视频检索
- **应用描述**: 基于内容的智能检索
- **技术优势**: 语义搜索，精准匹配用户需求

### 📊 内容总结
- **应用描述**: 自动生成视频摘要
- **技术优势**: 多维度分析，生成全面总结

---

## 🔧 配置选项

### 模型后端配置

#### 阿里云 DashScope（推荐配置）
- **优点**: 速度快、质量高、支持最新模型
- **缺点**: 需要 API Key，按使用量计费
- **适用模型**: Qwen3-VL-Plus, Qwen-VL-Plus, Qwen-Plus 等

#### Ollama（本地部署配置）
- **优点**: 完全免费、数据隐私、离线使用
- **缺点**: 需要本地部署，占用 GPU 资源，配置复杂，硬件要求高
- **适用模型**: LLaVA, Qwen, Mistral 等开源模型

### 检索策略配置

- **相似度搜索** - 默认策略，响应速度快
- **MMR 搜索** - 多样化结果，减少重复内容
- **阈值搜索** - 精准匹配，过滤低质量结果

---

## 📁 项目结构

```
视频转文字正式发布版v3.00/
├── py原文件/                    # 源代码目录
│   ├── app.py                   # 主程序文件（Streamlit 应用）
│   ├── run.py                   # 启动脚本（环境检查、模型下载）
│   ├── requirements.txt          # Python 依赖包列表
│   ├── setup.py                 # 安装配置文件
│   ├── 使用说明.md              # 详细使用说明文档
│   ├── 运行环境配置.bat         # Windows 环境配置脚本
│   └── ffmpeg_downlaod/         # FFmpeg 工具目录
│       └── bin/
│           └── ffmpeg.exe       # FFmpeg 可执行文件
├── python-3.12.10-amd64.exe     # Python 安装包
└── README.md                    # 项目说明文档（本文件）
```

### 运行时生成目录

程序运行时会自动创建以下目录（不会上传到版本控制）：

```
├── venv/                        # Python 虚拟环境
├── .cache/                      # 模型缓存目录（约 5GB）
├── keyframes/                   # 关键帧输出目录
├── cache/                       # 分析结果缓存目录
├── chroma_db/                   # 向量数据库目录
└── 过往信息/                    # 历史数据归档目录
```

---

## 🐛 常见问题解答

<details>
<summary><b>Q: 安装过程很慢怎么办？</b></summary>

**解决方案**: 使用国内镜像源加速下载

```bash
# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
</details>

<details>
<summary><b>Q: 首次运行程序卡住不动？</b></summary>

**原因分析**: 正常现象，程序正在下载 AI 模型文件

**解决方案**: 
- 保持网络畅通，等待 10-30 分钟
- 模型文件约 5GB，下载完成后会缓存到本地
- 后续运行无需重新下载
</details>

<details>
<summary><b>Q: GPU 没有被正确识别和使用？</b></summary>

**排查步骤**:
1. 检查 CUDA 版本（需要 11.8+）
2. 更新 NVIDIA 显卡驱动到最新版本
3. 确认 PyTorch 支持 CUDA
4. 程序会自动回退到 CPU 模式运行

**验证命令**:
```python
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示 GPU 型号
```
</details>

<details>
<summary><b>Q: 提示 FFmpeg 未找到？</b></summary>

**解决方案**:
1. 项目已内置 FFmpeg，程序会自动调用
2. 如果仍有问题，手动设置 FFmpeg 路径
3. 或下载 FFmpeg 并添加到系统 PATH

**下载地址**: https://www.gyan.dev/ffmpeg/builds/
</details>

<details>
<summary><b>Q: API 调用失败或报错？</b></summary>

**排查步骤**:
1. 检查 API Key 是否正确设置
2. 验证网络连接是否正常
3. 确认阿里云账户余额充足
4. 检查环境变量设置

**验证命令**:
```bash
echo %DASHSCOPE_API_KEY%  # 应该显示你的 API Key
```
</details>

---

## 🔄 版本更新日志

### v3.0 (2025-10-03)

#### ✨ 新增功能特性
- **RAG 智能检索**: 新增基于语义的智能问答功能
- **Qwen3 Embedding**: 集成中文语义向量模型
- **多种检索策略**: 支持相似度、MMR、阈值搜索
- **分析报告生成**: 新增完整报告和速读摘要
- **查询历史记录**: 保存用户查询历史
- **Token 用量统计**: 实时显示 API 使用情况和费用预估

#### 🚀 性能优化改进
- **🔥 流水线并行处理**: 抽帧、语音识别、图片分析同时进行，性能提升 30-69%
- **⚡ 边抽边传优化**: 关键帧提取完成即刻上传分析，无需等待全部抽帧完成
- **🚀 三线程并行架构**: 采用生产者-消费者模式，充分利用 CPU、网络、GPU 资源
- **GPU 加速支持**: 优化 CUDA 加速性能
- **并发处理优化**: 支持 60 QPM 速率限制
- **向量数据库优化**: 改进持久化存储机制
- **模型缓存管理**: 优化模型加载和缓存策略

#### 🐛 问题修复列表
- **向量数据库锁定**: 修复文件占用问题
- **FunASR 卡死**: 解决语音识别进程阻塞
- **异步并发问题**: 修复 session_state 同步问题
- **关键帧提取优化**: 改进提取效率和稳定性
- **ChromaDB 兼容性**: 修复数据库文件格式问题

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目！

### 开发环境设置

```bash
# 1. 克隆或下载项目
# 2. 创建 Python 虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装开发依赖
pip install -r requirements.txt

# 4. 运行开发版本
python run.py
```

### 代码规范要求
- 每个模块都要写详细的中文注释
- 保持代码高可读性和模块化设计
- 遵循 PEP 8 Python 编码规范
- 确保删除模块后不影响程序整体运行

### 提交规范
- 提交信息使用中文描述
- 关联相关的 Issue 编号
- 确保代码通过基本测试

---

## 📞 技术支持

### 联系方式
- **QQ**: 153115068（请备注来意）
- **邮箱**: 153115068@qq.com
- **议题**: [GitHub Issues](https://github.com/JucieOvo/Frame-by-frame-AI-parser_by_JucieOvo/issues)

### Bug 反馈要求
请提供以下信息以便快速定位问题：
- 操作系统版本信息
- Python 版本号
- GPU 型号和驱动版本（如果使用）
- 错误截图或详细日志
- 问题复现的具体步骤

---

## 📜 许可证信息

本项目采用 [MIT License](LICENSE) 开源协议

**许可证要点**:
- 允许商业使用
- 允许修改和分发
- 允许私人使用
- 不提供担保
- 保留版权声明

---



## ⭐ 项目支持

如果这个项目对您有帮助，请给个 Star 支持一下！⭐

您的支持是我们持续改进的动力！

---


## 🎓 学习资源

### 官方文档
- [Qwen 模型文档](https://help.aliyun.com/zh/dashscope/)
- [FunASR 使用指南](https://github.com/alibaba-damo-academy/FunASR)
- [Streamlit 文档](https://docs.streamlit.io/)
- [LangChain 中文文档](https://python.langchain.com/)

### 技术博客
- AI 视频分析技术原理
- RAG 系统架构设计
- 多模态模型应用实践
- 性能优化技巧分享

### 社区交流
- 技术问题讨论
- 功能需求建议
- 使用经验分享
- 性能优化方案

---

<div align="center">

**Made with ❤️ by CN_榨汁Ovo**

*技术让生活更美好 🚀 愿世界和平🕊️*

</div>


