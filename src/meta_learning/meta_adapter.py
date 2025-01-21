import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MetaAdapter(nn.Module):
    """
    元学习适配器 (Meta-Learning Adapter)
    用于快速适应新说话人的声音特征
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # 输入特征维度
        hidden_dim: int = 256,  # 隐藏层维度
        num_layers: int = 2,    # 网络层数
    ):
        """
        初始化元学习适配器
        
        参数:
            input_dim: 输入特征维度 (embedding dimension)
            hidden_dim: 隐藏层维度 (hidden dimension)
            num_layers: 网络层数 (number of layers)
        """
        super().__init__()
        
        # 特征提取网络 (Feature Extraction Network)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 初始化权重 (Initialize weights)
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入特征 (input features)
            
        返回:
            适配后的特征 (adapted features)
        """
        # 残差连接 (Residual Connection)
        return x + self.feature_extractor(x)
        
    def adapt_to_speaker(
        self,
        reference_embeddings: List[torch.Tensor],
        num_steps: int = 5,
        learning_rate: float = 0.01
    ) -> None:
        """
        快速适应新说话人 (Quick Adaptation)
        
        参数:
            reference_embeddings: 参考音频的特征向量列表
            num_steps: 适应步数 (adaptation steps)
            learning_rate: 学习率 (learning rate)
        """
        # 准备优化器 (Prepare optimizer)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 计算目标中心向量 (Calculate target centroid)
        target_centroid = torch.stack(reference_embeddings).mean(dim=0)
        
        # 快速适应循环 (Quick adaptation loop)
        for step in range(num_steps):
            total_loss = 0
            
            # 对每个参考样本进行适应
            for embedding in reference_embeddings:
                # 前向传播
                adapted_embedding = self(embedding)
                
                # 计算损失：
                # 1. 与目标中心向量的距离 (Distance to centroid)
                centroid_loss = F.mse_loss(adapted_embedding, target_centroid)
                
                # 2. 保持原有特征的约束 (Feature preservation)
                preservation_loss = F.mse_loss(adapted_embedding, embedding)
                
                # 总损失
                loss = centroid_loss + 0.1 * preservation_loss
                total_loss += loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 打印训练信息
            if (step + 1) % 1 == 0:
                print(f"适应步骤 {step + 1}/{num_steps}, 损失: {total_loss.item():.4f}")
                
    def get_adapted_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        获取适应后的特征向量
        
        参数:
            embedding: 输入特征向量
            
        返回:
            适应后的特征向量
        """
        with torch.no_grad():
            return self(embedding) 