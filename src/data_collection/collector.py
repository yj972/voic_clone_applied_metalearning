import torch
import torchaudio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from scipy.spatial.distance import cosine

class DataCollector:
    """数据收集器：记录和分析语音克隆的数据"""
    
    def __init__(self, data_dir: str = "collected_data"):
        """
        初始化数据收集器
        
        参数:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.audio_dir = self.data_dir / "audio"  # 存储音频文件
        self.features_dir = self.data_dir / "features"  # 存储特征数据
        self.metrics_dir = self.data_dir / "metrics"  # 存储评估指标
        
        for dir_path in [self.audio_dir, self.features_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def save_session_data(
        self,
        session_id: str,
        reference_audio: torch.Tensor,
        generated_audio: torch.Tensor,
        reference_embedding: torch.Tensor,
        adapted_embedding: torch.Tensor,
        text: str,
        metrics: Dict[str, float]
    ):
        """
        保存会话数据
        
        参数:
            session_id: 会话ID
            reference_audio: 参考音频
            generated_audio: 生成的音频
            reference_embedding: 原始特征向量
            adapted_embedding: 适应后的特征向量
            text: 输入文本
            metrics: 评估指标
        """
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存音频文件
        ref_path = self.audio_dir / f"{session_id}_ref_{timestamp}.wav"
        gen_path = self.audio_dir / f"{session_id}_gen_{timestamp}.wav"
        
        torchaudio.save(str(ref_path), reference_audio, 16000)
        torchaudio.save(str(gen_path), generated_audio, 16000)
        
        # 保存特征向量
        feature_path = self.features_dir / f"{session_id}_{timestamp}.pt"
        torch.save({
            "reference_embedding": reference_embedding,
            "adapted_embedding": adapted_embedding
        }, feature_path)
        
        # 保存会话信息和指标
        session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "text": text,
            "reference_audio_path": str(ref_path),
            "generated_audio_path": str(gen_path),
            "feature_path": str(feature_path),
            "metrics": metrics
        }
        
        # 保存为JSON文件
        json_path = self.metrics_dir / f"{session_id}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
            
    def calculate_metrics(
        self,
        reference_embedding: torch.Tensor,
        adapted_embedding: torch.Tensor,
        reference_audio: torch.Tensor,
        generated_audio: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        参数:
            reference_embedding: 参考音频的特征向量
            adapted_embedding: 适应后的特征向量
            reference_audio: 参考音频
            generated_audio: 生成的音频
            
        返回:
            包含各项指标的字典
        """
        metrics = {}
        
        # 1. 特征相似度 (Feature Similarity)
        ref_emb_np = reference_embedding.cpu().numpy()
        adp_emb_np = adapted_embedding.cpu().numpy()
        metrics["feature_similarity"] = 1 - cosine(ref_emb_np.flatten(), adp_emb_np.flatten())
        
        # 2. 音频长度比例 (Duration Ratio)
        ref_len = reference_audio.shape[-1]
        gen_len = generated_audio.shape[-1]
        metrics["duration_ratio"] = gen_len / ref_len
        
        # 3. 音量统计 (Volume Statistics)
        ref_vol = torch.abs(reference_audio).mean().item()
        gen_vol = torch.abs(generated_audio).mean().item()
        metrics["volume_ratio"] = gen_vol / ref_vol
        
        # 4. 特征适应程度 (Adaptation Degree)
        metrics["adaptation_degree"] = torch.norm(
            adapted_embedding - reference_embedding
        ).item()
        
        return metrics
        
    def analyze_session(self, session_id: str) -> Dict[str, Any]:
        """
        分析特定会话的数据
        
        参数:
            session_id: 会话ID
            
        返回:
            分析结果
        """
        # 获取会话的所有数据文件
        session_files = list(self.metrics_dir.glob(f"{session_id}_*.json"))
        
        if not session_files:
            raise ValueError(f"未找到会话 {session_id} 的数据")
            
        # 读取所有会话数据
        session_data = []
        for file_path in session_files:
            with open(file_path, "r", encoding="utf-8") as f:
                session_data.append(json.load(f))
                
        # 分析指标趋势
        metrics_trend = {
            "feature_similarity": [],
            "duration_ratio": [],
            "volume_ratio": [],
            "adaptation_degree": []
        }
        
        for data in session_data:
            for metric_name, metric_values in metrics_trend.items():
                metric_values.append(data["metrics"][metric_name])
                
        # 计算统计信息
        analysis = {
            "session_id": session_id,
            "total_samples": len(session_data),
            "metrics_mean": {
                k: float(np.mean(v)) for k, v in metrics_trend.items()
            },
            "metrics_std": {
                k: float(np.std(v)) for k, v in metrics_trend.items()
            },
            "time_range": {
                "start": session_data[0]["timestamp"],
                "end": session_data[-1]["timestamp"]
            }
        }
        
        return analysis
        
    def get_all_sessions(self) -> List[str]:
        """
        获取所有会话ID
        
        返回:
            会话ID列表
        """
        # 从文件名中提取会话ID
        session_ids = set()
        for file_path in self.metrics_dir.glob("*.json"):
            session_id = file_path.stem.split("_")[0]
            session_ids.add(session_id)
        return list(session_ids)
        
    def export_analysis(self, output_path: str = "analysis_report.json"):
        """
        导出所有数据的分析报告
        
        参数:
            output_path: 输出文件路径
        """
        # 获取所有会话
        sessions = self.get_all_sessions()
        
        # 分析每个会话
        analyses = {}
        for session_id in sessions:
            try:
                analyses[session_id] = self.analyze_session(session_id)
            except Exception as e:
                print(f"分析会话 {session_id} 时出错: {str(e)}")
                
        # 计算总体统计信息
        overall_stats = {
            "total_sessions": len(sessions),
            "total_samples": sum(a["total_samples"] for a in analyses.values()),
            "average_metrics": {}
        }
        
        # 计算所有会话的平均指标
        if analyses:
            metrics_sum = {}
            for analysis in analyses.values():
                for metric, value in analysis["metrics_mean"].items():
                    metrics_sum[metric] = metrics_sum.get(metric, 0) + value
                    
            overall_stats["average_metrics"] = {
                metric: value / len(analyses)
                for metric, value in metrics_sum.items()
            }
            
        # 保存报告
        report = {
            "overall_statistics": overall_stats,
            "session_analyses": analyses
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2) 