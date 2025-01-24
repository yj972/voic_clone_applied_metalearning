"""
语音识别模型封装类
基于Whisper-large-v3-turbo模型,CPU环境优化版本
"""

import torch
import torchaudio
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Optional, Union, List

class WhisperModel:
    """Whisper-Turbo模型CPU优化版本"""
    
    def __init__(self, task="transcribe", language="zh"):
        """
        初始化Whisper-Turbo模型(CPU优化版本)
        Args:
            task: 任务类型,transcribe或translate,默认transcribe
            language: 目标语言,默认中文
        """
        self.device = "cpu"
        self.model_id = "openai/whisper-large-v3-turbo"
        self.task = task
        self.language = language
        
        # 音频处理参数
        self.sample_rate = 16000
        self.max_duration = 15  # 减小最大处理时长以降低内存占用
        
        # 模型参数
        self.chunk_length_s = 15  # 更小的分块以减少内存使用
        self.batch_size = 1  # CPU环境使用较小的batch_size
        
        # 暂时不加载模型,只在需要时才加载
        self.model = None
        self.processor = None
        
    def load_model(self):
        """
        加载模型和处理器(CPU优化版本)
        只在首次使用时加载,节省内存
        """
        if self.model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # CPU使用float32
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_flash_attention_2=False  # CPU环境禁用flash attention
            )
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained(self.model_id)
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        预处理音频文件(CPU优化版本)
        Args:
            audio_path: 音频文件路径
        Returns:
            处理后的音频张量
        """
        try:
            # 使用numpy先处理以减少内存使用
            waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 重采样
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # 标准化音量
            waveform = waveform / (waveform.abs().max() + 1e-7)
            
            return waveform
            
        except Exception as e:
            print(f"音频预处理错误: {str(e)}")
            raise
            
    def transcribe(self, audio_path: str) -> str:
        """
        将音频转录为文本(CPU优化版本)
        Args:
            audio_path: 音频文件路径
        Returns:
            转录的文本
        """
        try:
            self.load_model()  # 确保模型已加载
            
            # 预处理音频
            waveform = self.preprocess_audio(audio_path)
            
            # 使用处理器准备输入
            inputs = self.processor(
                waveform, 
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                chunk_length_s=self.chunk_length_s,
                batch_size=self.batch_size
            )
            
            # 生成文本(使用较少的beam数量)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    task=self.task,
                    language=self.language,
                    num_beams=2,  # 减少beam数量以提高速度
                    max_new_tokens=128  # 减少最大token数
                )
            
            # 解码输出
            transcription = self.processor.batch_decode(
                outputs, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            print(f"转录过程错误: {str(e)}")
            raise
        
    def __call__(self, audio_path: str) -> str:
        """
        便捷调用接口
        Args:
            audio_path: 音频文件路径
        Returns:
            转录的文本
        """
        return self.transcribe(audio_path) 