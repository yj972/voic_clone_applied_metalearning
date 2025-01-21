import torch
import torchaudio
from pathlib import Path
from typing import List, Union
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from ..meta_learning.meta_adapter import MetaAdapter
from ..data_collection.collector import DataCollector
import uuid

class VoiceCloneSystem:
    """语音克隆系统：将输入文本转换为目标说话人的语音"""
    
    def __init__(self, device: str = "cpu"):
        """
        初始化语音克隆系统
        
        参数:
            device: 使用的设备，'cpu' 或 'cuda'
        """
        self.device = device
        print("正在加载模型...")
        
        # 加载说话人编码器 (Speaker Encoder)
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="tmp/spkrec-xvect-voxceleb",
            run_opts={"device": device}
        )
        
        # 加载文本到语音模型 (Text-to-Speech Model)
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(device)
        
        # 加载声码器 (Vocoder)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(device)
        
        # 初始化元学习适配器 (Meta-Learning Adapter)
        self.meta_adapter = MetaAdapter(
            input_dim=512,  # 说话人特征维度
            hidden_dim=256  # 隐藏层维度
        ).to(device)
        
        # 初始化数据收集器 (Data Collector)
        self.data_collector = DataCollector()
        
        print("模型加载完成！")
        
    def process_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        处理音频：重采样、降噪和音量标准化
        
        参数:
            waveform: 输入音频波形
            sr: 采样率 (sample rate)
        """
        # 重采样到16kHz (Resample to 16kHz)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        
        # 确保音频是单声道 (Ensure mono audio)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 音量标准化 (Volume normalization)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # 去除静音部分 (Remove silence)
        envelope = torch.abs(waveform)
        threshold = 0.05 * torch.max(envelope)
        mask = envelope.squeeze() > threshold
        if torch.sum(mask) > 0:
            start = torch.where(mask)[0][0]
            end = torch.where(mask)[0][-1]
            waveform = waveform[:, start:end+1]
        
        # 标准化音频长度（5秒）(Standardize audio length)
        target_length = 16000 * 5
        current_length = waveform.shape[1]
        
        if current_length > target_length:
            # 如果太长，截取中间部分
            start = (current_length - target_length) // 2
            waveform = waveform[:, start:start + target_length]
        elif current_length < target_length:
            # 如果太短，用边缘值填充
            left_pad = (target_length - current_length) // 2
            right_pad = target_length - current_length - left_pad
            waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad), mode='replicate')
            
        return waveform
        
    def extract_speaker_embedding(
        self,
        audio_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        从参考音频中提取说话人特征
        
        参数:
            audio_paths: 参考音频文件路径列表
        """
        embeddings = []
        
        for audio_path in audio_paths:
            try:
                # 加载音频 (Load audio)
                waveform, sr = torchaudio.load(str(audio_path))
                
                # 处理音频 (Process audio)
                waveform = self.process_audio(waveform, sr)
                
                # 提取特征 (Extract features)
                with torch.no_grad():
                    # 确保输入维度正确 [batch, time]
                    if waveform.dim() == 2:
                        waveform = waveform.squeeze(0)
                    
                    # 提取特征并处理维度
                    embedding = self.speaker_encoder.encode_batch(waveform.unsqueeze(0).to(self.device))
                    embedding = embedding.squeeze()  # 移除所有维度为1的维度
                    
                    # 特征标准化 (Feature normalization)
                    embedding = embedding / (torch.norm(embedding) + 1e-8)
                    
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"处理音频时出错 {audio_path}: {str(e)}")
                raise
        
        # 使用元学习适配器进行快速适应 (Quick adaptation using meta-learning)
        self.meta_adapter.adapt_to_speaker(embeddings)
        
        # 计算适应后的平均特征 (Calculate adapted mean embedding)
        mean_embedding = torch.stack([
            self.meta_adapter.get_adapted_embedding(emb) for emb in embeddings
        ]).mean(dim=0)
        
        # 确保最终维度正确 [1, 512]
        if mean_embedding.dim() == 1:
            mean_embedding = mean_embedding.unsqueeze(0)
        
        return mean_embedding
        
    def generate_speech(
        self,
        text: str,
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        生成语音
        
        参数:
            text: 输入文本
            speaker_embedding: 说话人特征向量
        """
        try:
            # 处理输入文本 (Process input text)
            inputs = self.processor(text=text, return_tensors="pt")
            
            # 确保说话人特征维度正确
            if speaker_embedding.dim() != 2 or speaker_embedding.size(1) != 512:
                raise ValueError(f"说话人特征维度应为 [1, 512]，但得到 {speaker_embedding.shape}")
            
            # 生成语音 (Generate speech)
            speech = self.tts_model.generate_speech(
                inputs["input_ids"].to(self.device),
                speaker_embedding.to(self.device),
                vocoder=self.vocoder,
                do_sample=True,  # 使用采样而不是贪婪解码
                temperature=0.7,  # 控制生成的随机性
                length_penalty=1.0,  # 长度惩罚
                repetition_penalty=1.2,  # 重复惩罚
            )
            
            # 后处理生成的音频 (Post-process generated audio)
            speech = speech / (torch.max(torch.abs(speech)) + 1e-8)  # 音量标准化
            
            return speech
            
        except Exception as e:
            print(f"生成语音时出错: {str(e)}")
            raise
        
    def clone_voice(
        self,
        text: str,
        reference_audio_paths: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        主函数：克隆声音
        
        参数:
            text: 要转换的文本
            reference_audio_paths: 参考音频文件路径列表
        """
        try:
            # 生成会话ID
            session_id = str(uuid.uuid4())[:8]
            
            # 加载参考音频
            reference_audio = None
            for path in reference_audio_paths:
                waveform, sr = torchaudio.load(str(path))
                waveform = self.process_audio(waveform, sr)
                if reference_audio is None:
                    reference_audio = waveform
                else:
                    reference_audio = torch.cat([reference_audio, waveform], dim=1)
            
            # 1. 提取说话人特征
            reference_embedding = self.extract_speaker_embedding(reference_audio_paths)
            
            # 2. 获取适应后的特征
            adapted_embedding = self.meta_adapter.get_adapted_embedding(reference_embedding)
            
            # 3. 生成语音
            generated_audio = self.generate_speech(text, adapted_embedding)
            
            # 4. 计算评估指标
            metrics = self.data_collector.calculate_metrics(
                reference_embedding=reference_embedding,
                adapted_embedding=adapted_embedding,
                reference_audio=reference_audio,
                generated_audio=generated_audio
            )
            
            # 5. 保存会话数据
            self.data_collector.save_session_data(
                session_id=session_id,
                reference_audio=reference_audio,
                generated_audio=generated_audio,
                reference_embedding=reference_embedding,
                adapted_embedding=adapted_embedding,
                text=text,
                metrics=metrics
            )
            
            return generated_audio
            
        except Exception as e:
            print(f"克隆声音时出错: {str(e)}")
            raise
        
    def save_audio(
        self,
        waveform: torch.Tensor,
        output_path: Union[str, Path],
        sample_rate: int = 16000
    ):
        """
        保存音频文件
        
        参数:
            waveform: 音频波形
            output_path: 输出文件路径
            sample_rate: 采样率
        """
        try:
            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 标准化音量
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # 保存音频
            torchaudio.save(
                str(output_path),
                waveform.unsqueeze(0).cpu(),
                sample_rate
            )
        except Exception as e:
            print(f"保存音频时出错: {str(e)}")
            raise 