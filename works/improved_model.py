"""
改进的灾害检测模型
支持灾害类型分类和时间起止预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DisasterTimePredictor(nn.Module):
    """灾害检测和时间预测模型"""
    
    def __init__(self, frame_feature_dim: int = 2048, num_classes: int = 7):
        super(DisasterTimePredictor, self).__init__()
        
        # 灾害类型分类器
        self.disaster_classifier = nn.Sequential(
            nn.Linear(frame_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 时间预测器（起始帧）
        self.start_time_predictor = nn.Sequential(
            nn.Linear(frame_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 时间预测器（结束帧）
        self.end_time_predictor = nn.Sequential(
            nn.Linear(frame_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 早期预警预测器（在灾害开始前预测灾害类型）
        self.early_warning_predictor = nn.Sequential(
            nn.Linear(frame_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        前向传播
        x: 视频特征 [batch_size, feature_dim]
        返回: (disaster_logits, start_time_logits, end_time_logits, early_warning_logits)
        """
        disaster_logits = self.disaster_classifier(x)
        start_time_logits = self.start_time_predictor(x).squeeze(-1)
        end_time_logits = self.end_time_predictor(x).squeeze(-1)
        early_warning_logits = self.early_warning_predictor(x)
        
        return disaster_logits, start_time_logits, end_time_logits, early_warning_logits


class AttentionDisasterPredictor(nn.Module):
    """基于注意力机制的灾害检测和时间预测模型"""
    
    def __init__(self, frame_feature_dim: int = 2048, num_classes: int = 7, 
                 num_heads: int = 8, num_layers: int = 2):
        super(AttentionDisasterPredictor, self).__init__()
        
        self.frame_feature_dim = frame_feature_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 多头注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(frame_feature_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(frame_feature_dim)
            for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(frame_feature_dim, frame_feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(frame_feature_dim * 4, frame_feature_dim)
        )
        
        # 灾害类型分类器
        self.disaster_classifier = nn.Sequential(
            nn.Linear(frame_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 时间预测器（起始帧）
        self.start_time_predictor = nn.Sequential(
            nn.Linear(frame_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 时间预测器（结束帧）
        self.end_time_predictor = nn.Sequential(
            nn.Linear(frame_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        前向传播
        x: 视频特征 [batch_size, feature_dim]
        返回: (disaster_logits, start_time_logits, end_time_logits)
        """
        # 添加序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 通过注意力层
        for i in range(self.num_layers):
            attn_output, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms[i](x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + ff_output
        
        # 移除序列维度
        x = x.squeeze(1)  # [batch_size, feature_dim]
        
        disaster_logits = self.disaster_classifier(x)
        start_time_logits = self.start_time_predictor(x).squeeze(-1)
        end_time_logits = self.end_time_predictor(x).squeeze(-1)
        
        return disaster_logits, start_time_logits, end_time_logits


class DisasterLoss(nn.Module):
    """灾害检测和时间预测的联合损失函数"""
    
    def __init__(self, disaster_weight: Optional[torch.Tensor] = None,
                 time_loss_weight: float = 0.5,
                 early_warning_weight: float = 0.3):
        super(DisasterLoss, self).__init__()
        self.disaster_loss = nn.CrossEntropyLoss(weight=disaster_weight)
        self.time_loss_weight = time_loss_weight
        self.early_warning_weight = early_warning_weight
        
        # 时间预测准确度统计
        self.time_accuracy = []
        self.time_accuracy_1s = []
    
    def forward(self, disaster_logits: torch.Tensor, disaster_labels: torch.Tensor,
                start_time_logits: torch.Tensor, end_time_logits: torch.Tensor,
                start_time_labels: torch.Tensor, end_time_labels: torch.Tensor,
                early_warning_logits: torch.Tensor, early_warning_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算损失
        返回: (total_loss, disaster_loss, time_loss, early_warning_loss)
        """
        # 灾害分类损失
        d_loss = self.disaster_loss(disaster_logits, disaster_labels)
        
        # 时间预测损失（只对有时间标签的样本计算）
        valid_time_mask = (start_time_labels >= 0) & (end_time_labels >= 0)
        
        if valid_time_mask.sum() > 0:
            start_loss = F.mse_loss(
                start_time_logits[valid_time_mask], 
                start_time_labels[valid_time_mask].float()
            )
            end_loss = F.mse_loss(
                end_time_logits[valid_time_mask], 
                end_time_labels[valid_time_mask].float()
            )
            t_loss = (start_loss + end_loss) / 2
            
            # 计算时间预测准确度
            with torch.no_grad():
                start_pred = torch.sigmoid(start_time_logits[valid_time_mask])
                end_pred = torch.sigmoid(end_time_logits[valid_time_mask])
                start_true = start_time_labels[valid_time_mask].float()
                end_true = end_time_labels[valid_time_mask].float()
                
                # 计算准确度（相差1秒内就算成功）
                start_error = (start_pred - start_true).abs()
                end_error = (end_pred - end_true).abs()
                
                # 将误差转换为秒（假设30fps）
                start_error_sec = start_error / 30.0
                end_error_sec = end_error / 30.0
                
                # 统计准确度
                start_acc = (start_error_sec <= 1.0).float().mean().item()
                end_acc = (end_error_sec <= 1.0).float().mean().item()
                self.time_accuracy_1s.extend([start_acc, end_acc])
        else:
            t_loss = torch.tensor(0.0, device=disaster_logits.device)
        
        # 早期预警损失（只对有早期预警标签的样本计算）
        valid_early_mask = (early_warning_labels >= 0)
        
        if valid_early_mask.sum() > 0:
            ew_loss = self.disaster_loss(early_warning_logits[valid_early_mask], early_warning_labels[valid_early_mask])
        else:
            ew_loss = torch.tensor(0.0, device=disaster_logits.device)
        
        # 总损失
        total_loss = d_loss + self.time_loss_weight * t_loss + self.early_warning_weight * ew_loss
        
        return total_loss, d_loss, t_loss, ew_loss
    
    def get_time_accuracy(self) -> float:
        """获取时间预测准确度（1秒内）"""
        if not self.time_accuracy_1s:
            return 0.0
        return sum(self.time_accuracy_1s) / len(self.time_accuracy_1s)
    
    def reset_metrics(self):
        """重置统计指标"""
        self.time_accuracy_1s = []
