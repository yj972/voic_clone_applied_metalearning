---
title: Voice Clone App
emoji: 😻
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Voice Clone App 语音克隆应用

一个基于深度学习的语音克隆系统，支持快速适应新说话人的声音特征。本项目使用元学习方法提高模型的泛化能力，并提供了用户友好的Web界面。

A deep learning-based voice cloning system that supports quick adaptation to new speakers. This project uses meta-learning to improve model generalization and provides a user-friendly web interface.

## 功能特点 Features

- 🎯 快速适应：仅需5-10秒的参考音频
- 🌍 支持中英文：可处理中文和英文文本输入
- 🔄 元学习增强：使用元学习提高声音克隆质量
- 📊 数据分析：自动收集和分析克隆效果
- 🖥️ 简单界面：提供网页交互界面

## 安装 Installation

1. 克隆仓库 Clone the repository:
```bash
git clone https://github.com/yourusername/voice-clone-app.git
cd voice-clone-app
```

2. 创建虚拟环境 Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. 安装依赖 Install dependencies:
```bash
pip install -r requirements.txt
```

## 使用方法 Usage

### 本地运行 Local Run

```bash
python app.py
```

然后在浏览器中打开 http://localhost:7860

### Hugging Face Spaces

访问我们的 [Hugging Face Space](https://huggingface.co/spaces/pupunpu/voice-clone-app) 直接使用在线版本。

## 项目结构 Project Structure

```
voice-clone-app/
├── src/
│   ├── deploy/
│   │   └── voice_clone.py      # 核心语音克隆系统
│   │   └── meta_adapter.py     # 元学习适配器
│   └── data_collection/
│       └── collector.py        # 数据收集和分析
├── app.py                      # Gradio Web界面
├── requirements.txt            # 项目依赖
└── README.md                   # 项目文档
```

## 技术细节 Technical Details

- 语音编码器：使用 SpeechBrain 的说话人识别模型
- 文本转语音：基于 Microsoft SpeechT5
- 元学习模块：自定义的快速适应网络
- 数据分析：自动计算多个评估指标

## 评估指标 Evaluation Metrics

- 特征相似度：衡量声音特征的相似程度
- 音频长度比例：比较生成音频和参考音频的长度
- 音量统计：分析音量变化
- 特征适应程度：评估元学习模块的效果

## 贡献 Contributing

欢迎提交 Issue 和 Pull Request！

## 许可证 License

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢 Acknowledgments

- Microsoft SpeechT5
- SpeechBrain
- Hugging Face
- Gradio
