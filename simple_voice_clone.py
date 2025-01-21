from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
import gradio as gr
from pathlib import Path
import numpy as np

class SimpleVoiceCloner:
    def __init__(self):
        # 加载所需模型
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb"
        )
        
    def clone_voice(self, text, audio_file):
        """一步完成语音克隆"""
        # 1. 加载并处理参考音频
        wav, sr = torchaudio.load(audio_file)
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)  # 转单声道
        
        # 2. 提取说话人特征
        with torch.no_grad():
            embed = self.speaker_encoder.encode_batch(wav)
            embed = embed.squeeze()
        
        # 3. 生成新语音
        inputs = self.processor(text=text, return_tensors="pt")
        speech = self.tts.generate_speech(
            inputs["input_ids"], 
            embed.unsqueeze(0),
            vocoder=self.vocoder
        )
        
        # 4. 保存并返回结果
        output_path = "output.wav"
        torchaudio.save(output_path, speech.unsqueeze(0), 16000)
        return output_path

# 创建Gradio界面
def create_ui():
    cloner = SimpleVoiceCloner()
    
    def process(text, audio):
        try:
            return cloner.clone_voice(text, audio)
        except Exception as e:
            return str(e)
    
    demo = gr.Interface(
        fn=process,
        inputs=[
            gr.Textbox(label="输入文本", placeholder="请输入要转换的文本..."),
            gr.Audio(label="参考音频", source="microphone", type="filepath")
        ],
        outputs=gr.Audio(label="生成的语音"),
        title="简单语音克隆系统",
        description="上传或录制参考音频，输入文本，系统会生成相似声音的语音。",
    )
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch() 