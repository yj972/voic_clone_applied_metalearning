import gradio as gr
import torch
from pathlib import Path
import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.deploy.voice_clone import VoiceCloneSystem
from src.models.whisper import WhisperModel
from src.data_collection.collector import DataCollector

# 创建目录
TEMP_DIR = Path("temp")
DATA_DIR = Path("collected_data")
for dir_path in [TEMP_DIR, DATA_DIR]:
    dir_path.mkdir(exist_ok=True)

# 初始化数据收集器
data_collector = DataCollector(str(DATA_DIR))

# 初始化不同的克隆系统
class CloneSystemManager:
    def __init__(self):
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.systems = {
            "baseline": VoiceCloneSystem(device=self.device),  # 基础版本
            "whisper_enhanced": None,  # 用Whisper增强的版本
            "hybrid": None  # 混合学习版本
        }
        self.current_system = "baseline"
    
    def get_system(self, system_name):
        return self.systems.get(system_name)

system_manager = CloneSystemManager()

def clone_voice(text: str, reference_audio: str, system_type: str, user_rating: int = 3) -> str:
    """
    语音克隆的Gradio接口函数
    
    参数:
        text: 要转换的文本
        reference_audio: 参考音频文件路径
        system_type: 使用的克隆系统类型
        user_rating: 用户评分(1-5)
    返回:
        生成的音频文件路径
    """
    try:
        # 生成会话ID
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 获取选定的克隆系统
        system = system_manager.get_system(system_type)
        if system is None:
            raise gr.Error(f"系统 {system_type} 尚未实现")
            
        # 生成克隆语音
        speech = system.clone_voice(text, [reference_audio])
        
        # 保存生成的音频
        output_path = str(TEMP_DIR / f"output_{session_id}.wav")
        system.save_audio(speech, output_path)
        
        # 收集数据
        try:
            # 加载参考音频
            ref_waveform, sr = torchaudio.load(reference_audio)
            
            # 保存会话数据
            session_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "system_type": system_type,
                "text": text,
                "user_rating": user_rating,
                "reference_audio": reference_audio,
                "output_audio": output_path,
            }
            
            # 保存到JSON文件
            data_path = DATA_DIR / f"session_{session_id}.json"
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"数据收集错误: {str(e)}")
        
        return output_path
        
    except Exception as e:
        raise gr.Error(str(e))

# 创建Gradio界面
demo = gr.Interface(
    fn=clone_voice,
    inputs=[
        gr.Textbox(
            label="输入文本 | Input Text",
            placeholder="请输入要转换的文本... | Enter the text to convert...",
            lines=3
        ),
        gr.Audio(
            label="参考音频 | Reference Audio",
            sources=["microphone", "upload"],
            type="filepath"
        ),
        gr.Radio(
            choices=["baseline", "whisper_enhanced", "hybrid"],
            value="baseline",
            label="克隆系统类型 | Clone System Type",
            info="选择不同的语音克隆方案进行测试"
        ),
        gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="效果评分 | Rating",
            info="请为生成的语音效果打分(1-5分)"
        )
    ],
    outputs=[
        gr.Audio(label="生成的语音 | Generated Speech"),
    ],
    title="🎤 语音克隆系统测试平台 | Voice Cloning Test Platform",
    description="""
    ## 测试不同的语音克隆方案 | Test Different Voice Cloning Approaches
    
    ### 可选方案 | Available Systems:
    1. Baseline: 基础SpeechT5 + 元学习
    2. Whisper Enhanced: 使用Whisper增强的特征提取 (开发中)
    3. Hybrid: 混合学习方案 (开发中)
    
    ### 使用说明 | Instructions:
    1. 选择克隆系统类型 | Select clone system type
    2. 录制/上传参考音频 | Record/Upload reference audio
    3. 输入要转换的文本 | Input text to convert
    4. 为生成效果打分 | Rate the generation quality
    5. 点击提交并等待 | Click submit and wait
    
    ### 评分标准 | Rating Criteria:
    - 5分: 声音非常相似，自然流畅
    - 4分: 声音相似，比较自然
    - 3分: 声音一般，有些不自然
    - 2分: 声音差异大，不够自然
    - 1分: 声音完全不同，非常不自然
    """,
    examples=[
        ["你好，这是一段测试文本。", None, "baseline", 3],
        ["Hello, this is a test message.", None, "baseline", 3],
    ]
)

# 启动应用
if __name__ == "__main__":
    demo.launch() 