"""
特征提取模块
使用预训练的ResNet提取帧级特征，并提取时序特征
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_packed_sequence
import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class FrameFeatureExtractor(nn.Module):
    """帧级特征提取器，使用ResNet50"""
    
    def __init__(self, feature_dim: int = 2048):
        super(FrameFeatureExtractor, self).__init__()
        # 使用预训练的ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # 移除最后的分类层，保留特征提取部分
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # 添加全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim
        
        # 冻结预训练模型的参数（可选）
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取帧特征
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
        Returns:
            特征向量 [batch_size, feature_dim]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class TemporalFeatureExtractor(nn.Module):
    """时序特征提取器，使用LSTM提取视频的时序特征"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, 
                 num_layers: int = 2, dropout: float = 0.5):
        super(TemporalFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出维度是双向LSTM的两倍
        self.output_dim = hidden_dim * 2
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取时序特征
        Args:
            x: 输入特征序列 [batch_size, seq_len, input_dim]
            lengths: 每个序列的实际长度（用于处理变长序列）
        Returns:
            时序特征 [batch_size, seq_len, output_dim]
        """
        if lengths is not None:
            # 使用pack_padded_sequence处理变长序列
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        out, (h_n, c_n) = self.lstm(x)
        
        if lengths is not None:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        return out


class VideoFeatureExtractor:
    """视频特征提取器，整合帧级和时序特征提取"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 frame_sample_rate: int = 1):
        """
        Args:
            device: 计算设备
            frame_sample_rate: 帧采样率（每隔多少帧采样一次）
        """
        self.device = torch.device(device)
        self.frame_sample_rate = frame_sample_rate
        
        # 初始化帧特征提取器
        self.frame_extractor = FrameFeatureExtractor().to(self.device)
        self.frame_extractor.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frame_features(self, video_path: str, 
                               max_frames: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        从视频中提取帧特征
        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数（None表示提取所有帧）
        Returns:
            帧特征数组 [num_frames, feature_dim] 和帧时间戳列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_features = []
        frame_timestamps = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按采样率采样帧
            if frame_count % self.frame_sample_rate == 0:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # 预处理
                frame_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    features = self.frame_extractor(frame_tensor)
                    frame_features.append(features.cpu().numpy()[0])
                    
                    # 计算时间戳（秒）
                    timestamp = frame_count / fps if fps > 0 else frame_count * 0.033
                    frame_timestamps.append(timestamp)
                
                # 限制最大帧数
                if max_frames and len(frame_features) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        if len(frame_features) == 0:
            raise ValueError(f"视频中没有提取到有效帧: {video_path}")
        
        return np.array(frame_features), frame_timestamps
    
    def extract_video_features(self, video_path: str, 
                               max_frames: Optional[int] = None) -> np.ndarray:
        """
        提取整个视频的特征（平均池化后的帧特征）
        Args:
            video_path: 视频文件路径
            max_frames: 最大帧数
        Returns:
            视频特征向量 [feature_dim]
        """
        frame_features, _ = self.extract_frame_features(video_path, max_frames)
        # 使用平均池化得到视频级别的特征
        video_feature = np.mean(frame_features, axis=0)
        return video_feature


class MultiScaleFeatureExtractor(VideoFeatureExtractor):
    """多尺度特征提取器，提取不同尺度的特征"""
    
    def extract_multiscale_features(self, video_path: str, 
                                    scales: List[int] = [1, 2, 4]) -> np.ndarray:
        """
        提取多尺度特征
        Args:
            video_path: 视频文件路径
            scales: 不同的帧采样率
        Returns:
            多尺度特征向量
        """
        all_features = []
        
        for scale in scales:
            self.frame_sample_rate = scale
            features, _ = self.extract_frame_features(video_path)
            # 对每个尺度进行平均池化
            scale_feature = np.mean(features, axis=0)
            all_features.append(scale_feature)
        
        # 拼接所有尺度的特征
        return np.concatenate(all_features, axis=0)

