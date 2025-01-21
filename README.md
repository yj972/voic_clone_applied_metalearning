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

# Simple Voice Clone

一个简单但强大的语音克隆系统，可以模仿任何声音说出指定的文本。

## 功能特点

- 简单易用的网页界面
- 支持实时录音或上传音频文件
- 支持中文和英文文本
- 基于最新的AI模型

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行应用：
```bash
python simple_voice_clone.py
```

3. 打开浏览器访问：http://localhost:7860

## 使用方法

1. 录制或上传一段参考音频（5-10秒最佳）
2. 输入要转换的文本
3. 点击提交，等待生成结果

## 技术栈

- SpeechT5 (微软语音模型)
- Speechbrain (语音特征提取)
- Gradio (Web界面)
- PyTorch (深度学习框架)

## 注意事项

- 首次运行时需要下载模型，可能需要等待一段时间
- 建议在安静的环境下录音
- 生成的语音质量取决于输入音频的质量

## License

MIT
