import gradio as gr
import torch
from pathlib import Path
import os
import sys

# 添加项目根目录到 Python 路径 (Python path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deploy.voice_clone import VoiceCloneSystem

# 创建临时目录用于存储音频文件
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

# 初始化语音克隆系统 (Voice Clone System)
# 根据是否有GPU来选择运行设备
system = VoiceCloneSystem(device="cpu" if not torch.cuda.is_available() else "cuda")

def clone_voice(text: str, reference_audio) -> str:
    """
    语音克隆的 Gradio 接口函数
    
    参数:
        text: 要转换的文本
        reference_audio: 参考音频文件路径 (reference audio path)
        
    返回:
        生成的音频文件路径 (output audio path)
    """
    try:
        # 生成克隆语音 (generate cloned speech)
        speech = system.clone_voice(text, [reference_audio])
        
        # 保存生成的音频
        output_path = str(TEMP_DIR / "output.wav")
        system.save_audio(speech, output_path)
        
        return output_path
        
    except Exception as e:
        raise gr.Error(str(e))

# 创建 Gradio 网页界面
demo = gr.Interface(
    fn=clone_voice,
    inputs=[
        # 文本输入框
        gr.Textbox(
            label="输入文本 | Input Text",
            placeholder="请输入要转换的文本... | Enter the text to convert...",
            lines=3
        ),
        # 音频输入组件
        gr.Audio(
            label="参考音频 | Reference Audio",
            sources=["microphone", "upload"],  # 支持麦克风录制和文件上传
            type="filepath"
        )
    ],
    outputs=gr.Audio(label="生成的语音 | Generated Speech"),
    title="🎤 语音克隆系统 | Voice Cloning System",
    description="""
    上传或录制参考音频（5-10秒最佳），输入文本，系统会生成具有相同声音特征的语音。支持中文和英文！
    
    Upload or record a reference audio (5-10s is best), input text, and the system will generate speech with similar voice characteristics. Supports both English and Chinese!
    
    ## 使用说明 | Instructions
    1. 录制/上传参考音频 | Record/Upload reference audio
    2. 输入要转换的文本 | Input text to convert
    3. 点击提交并等待 | Click submit and wait
    
    ## 注意事项 | Notes
    - 请在安静的环境下录音 | Record in a quiet environment
    - 保持适当的录音距离和音量 | Maintain proper distance and volume
    - 首次运行需要下载模型，可能需要等待一段时间 | First run may take time to download models
    """,
    examples=[
        ["你好，这是一段测试文本。", None],
        ["Hello, this is a test message.", None],
    ]
)

# 启动应用
if __name__ == "__main__":
    demo.launch() 