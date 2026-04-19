"""
完整的数据处理模块
包含灾害类型检测和时间起止定位功能
"""
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Optional
import os


class DisasterAnnotation:
    """灾害标注类"""
    def __init__(self, video_name: str, disaster_type: str, 
                 time_info: Optional[str] = None):
        self.video_name = video_name
        self.disaster_type = disaster_type
        self.start_time = None
        self.end_time = None
        self.has_time_info = False
        
        if time_info and time_info != '无':
            self._parse_time_info(time_info)
    
    def _parse_time_info(self, time_info: str):
        """解析时间信息"""
        try:
            # 提取起始和结束时间
            start_match = re.search(r'起始时刻[：:]\s*(\d+):(\d+)', time_info)
            end_match = re.search(r'结束时刻[：:]\s*(\d+):(\d+)', time_info)
            
            if start_match and end_match:
                self.start_time = int(start_match.group(1)) * 60 + int(start_match.group(2))
                self.end_time = int(end_match.group(1)) * 60 + int(end_match.group(2))
                self.has_time_info = True
                
                # 计算早期预警时间（灾害开始前1秒）
                self.early_warning_time = max(0, self.start_time - 1)
        except Exception as e:
            print(f"解析时间信息失败: {time_info}, 错误: {e}")
    
    def __repr__(self):
        return f"DisasterAnnotation(video={self.video_name}, type={self.disaster_type}, " \
               f"time={self.start_time}-{self.end_time if self.has_time_info else 'N/A'})"


class DisasterDatasetProcessor:
    """灾害数据集处理器"""
    
    # 标准化的灾害类型映射
    DISASTER_TYPE_MAPPING = {
        '林火': 'forestfire',
        '林火\n': 'forestfire',
        '森林火灾': 'forestfire',
        '森林火灾\n': 'forestfire',
        '城市内涝': 'waterlogging',
        '城市内涝\n': 'waterlogging',
        '山洪': 'mountainFloods',
        '山洪\n': 'mountainFloods',
        '泥石流': 'debrisFlow',
        '泥石流\n': 'debrisFlow',
        '滑坡': 'landslide',
        '滑坡\n': 'landslide',
        '流域性洪水': 'riverOverflow',
        '流域性洪水\n': 'riverOverflow',
        '地裂缝': 'landslide',
        '地面塌陷': 'landslide',
        '地面沉降': 'landslide',
        '地震': 'landslide',
        '无': 'normal'
    }
    
    def __init__(self, data_dir: str, annotation_file: str):
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file)
        self.annotations = []
        self.video_files = set()
        self._load_data()
    
    def _load_data(self):
        """加载标注数据和视频文件"""
        print("加载标注文件...")
        df = pd.read_excel(self.annotation_file)
        
        # 收集所有实际视频文件
        print("收集视频文件...")
        for group_dir in self.data_dir.iterdir():
            if group_dir.is_dir() and not group_dir.name.startswith('.'):
                for batch_dir in group_dir.iterdir():
                    if batch_dir.is_dir():
                        for video_file in batch_dir.glob('*.mp4'):
                            self.video_files.add(video_file.name)
        
        print(f"找到 {len(self.video_files)} 个视频文件")
        
        # 处理标注数据
        print("处理标注数据...")
        valid_count = 0
        for _, row in df.iterrows():
            video_name = row['视频名称']
            disaster_type_raw = row['灾害类型']
            time_info = row['灾害现象起始与结束时刻']
            
            # 检查视频是否存在
            if video_name not in self.video_files:
                continue
            
            # 标准化灾害类型
            disaster_type = self._normalize_disaster_type(disaster_type_raw)
            
            if disaster_type:
                annotation = DisasterAnnotation(video_name, disaster_type, time_info)
                self.annotations.append(annotation)
                valid_count += 1
        
        print(f"有效标注数量: {valid_count}")
    
    def _normalize_disaster_type(self, disaster_type: str) -> Optional[str]:
        """标准化灾害类型"""
        if pd.isna(disaster_type):
            return 'normal'
        
        # 去除空白和换行符
        disaster_type = str(disaster_type).strip()
        
        # 检查是否在映射表中
        if disaster_type in self.DISASTER_TYPE_MAPPING:
            return self.DISASTER_TYPE_MAPPING[disaster_type]
        
        # 处理复合灾害类型（如"山洪、泥石流"）
        types = disaster_type.split('、')
        for t in types:
            t = t.strip()
            if t in self.DISASTER_TYPE_MAPPING:
                return self.DISASTER_TYPE_MAPPING[t]
        
        # 如果都找不到，返回None
        return None
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        distribution = {}
        for ann in self.annotations:
            disaster_type = ann.disaster_type
            distribution[disaster_type] = distribution.get(disaster_type, 0) + 1
        return distribution
    
    def get_time_info_samples(self, n: int = 10) -> List[DisasterAnnotation]:
        """获取有时间信息的样本"""
        time_samples = [ann for ann in self.annotations if ann.has_time_info]
        return time_samples[:n]


class DisasterVideoDataset(Dataset):
    """灾害视频数据集"""
    
    def __init__(self, annotations: List[DisasterAnnotation], 
                 data_dir: str, max_frames: int = 32,
                 frame_sample_rate: int = 1, 
                 include_time_prediction: bool = False):
        self.annotations = annotations
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.frame_sample_rate = frame_sample_rate
        self.include_time_prediction = include_time_prediction
        
        # 类别映射
        self.class_to_idx = {
            'normal': 0,
            'landslide': 1,
            'debrisFlow': 2,
            'forestfire': 3,
            'mountainFloods': 4,
            'riverOverflow': 5,
            'waterlogging': 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 灾害类型到时间标签的映射
        self.disaster_to_time_idx = {
            'normal': 0,
            'landslide': 1,
            'debrisFlow': 2,
            'forestfire': 3,
            'mountainFloods': 4,
            'riverOverflow': 5,
            'waterlogging': 6
        }
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # 查找视频文件
        video_path = self._find_video_path(annotation.video_name)
        
        # 读取视频帧
        frames = self._extract_frames(video_path)
        
        # 获取标签
        label = self.class_to_idx[annotation.disaster_type]
        
        # 时间信息（如果需要）
        time_label = None
        early_warning_label = None
        if self.include_time_prediction and annotation.has_time_info:
            # 将时间转换为帧索引
            start_frame, end_frame = self._convert_time_to_frames(annotation, len(frames))
            time_label = torch.LongTensor([start_frame, end_frame])
            
            # 早期预警标签：在灾害开始前1秒预测灾害类型
            # 如果灾害类型不是normal，则早期预警标签为该灾害类型
            # 如果灾害类型是normal，则早期预警标签为normal
            if annotation.disaster_type != 'normal':
                early_warning_label = label
            else:
                early_warning_label = label
        
        return {
            'frames': torch.FloatTensor(frames),
            'label': torch.LongTensor([label]),
            'video_name': annotation.video_name,
            'time_label': time_label if time_label is not None else torch.LongTensor([-1, -1]),
            'early_warning_label': early_warning_label if early_warning_label is not None else torch.LongTensor([-1]),
            'original_time_info': (annotation.start_time, annotation.end_time) if annotation.has_time_info else (None, None)
        }
    
    def _find_video_path(self, video_name: str) -> Path:
        """查找视频文件的完整路径"""
        for group_dir in self.data_dir.iterdir():
            if group_dir.is_dir() and not group_dir.name.startswith('.'):
                for batch_dir in group_dir.iterdir():
                    if batch_dir.is_dir():
                        video_path = batch_dir / video_name
                        if video_path.exists():
                            return video_path
        raise FileNotFoundError(f"找不到视频文件: {video_name}")
    
    def _extract_frames(self, video_path: Path) -> np.ndarray:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 默认帧率
        
        frame_interval = max(1, int(fps / self.frame_sample_rate))
        
        count = 0
        while len(frames) < self.max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            
            count += 1
        
        cap.release()
        
        # 如果帧数不足，进行填充
        while len(frames) < self.max_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        return np.array(frames)
    
    def _convert_time_to_frames(self, annotation: DisasterAnnotation, 
                              num_frames: int) -> Tuple[int, int]:
        """将时间转换为帧索引"""
        if not annotation.has_time_info:
            return 0, num_frames - 1
        
        # 假设视频时长为60秒（可以根据实际情况调整）
        video_duration = 60
        start_frame = int((annotation.start_time / video_duration) * num_frames)
        end_frame = int((annotation.end_time / video_duration) * num_frames)
        
        # 确保在有效范围内
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(0, min(end_frame, num_frames - 1))
        
        return start_frame, end_frame


def collate_fn(batch):
    """自定义collate函数"""
    frames = torch.stack([item['frames'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    video_names = [item['video_name'] for item in batch]
    
    result = {
        'frames': frames,
        'labels': labels,
        'video_names': video_names
    }
    
    # 如果有时间标签
    if batch[0]['time_label'][0] != -1:
        time_labels = torch.stack([item['time_label'] for item in batch])
        result['time_labels'] = time_labels
        result['original_time_info'] = [item['original_time_info'] for item in batch]
    
    return result


def prepare_dataset(data_dir: str, annotation_file: str, 
                 test_size: float = 0.2, 
                 max_samples: Optional[int] = None,
                 balance_classes: bool = False,
                 class_sample_counts: Optional[Dict[str, int]] = None) -> Tuple:
    """
    准备数据集
    返回: (train_annotations, test_annotations, processor)
    
    Args:
        data_dir: 数据目录
        annotation_file: 标注文件路径
        test_size: 测试集比例（默认0.2）
        max_samples: 最大样本数（None表示使用全部样本）
        balance_classes: 是否平衡类别（默认False）
        class_sample_counts: 每个类别的样本数（None表示自动计算）
    """
    processor = DisasterDatasetProcessor(data_dir, annotation_file)
    
    # 显示类别分布
    print("\n类别分布:")
    distribution = processor.get_class_distribution()
    for disaster_type, count in sorted(distribution.items()):
        print(f"  {disaster_type}: {count}")
    
    # 如果指定了每个类别的样本数
    if class_sample_counts:
        print(f"\n使用指定的类别样本数:")
        annotations = []
        
        # 按照指定的数量选择每个类别的样本
        for disaster_type, target_count in class_sample_counts.items():
            disaster_class_annotations = [ann for ann in processor.annotations if ann.disaster_type == disaster_type]
            
            if len(disaster_class_annotations) >= target_count:
                # 随机选择指定数量的样本
                np.random.shuffle(disaster_class_annotations)
                selected_annotations = disaster_class_annotations[:target_count]
                annotations.extend(selected_annotations)
                print(f"  {disaster_type}: {len(selected_annotations)} 个样本（目标: {target_count}，可用: {len(disaster_class_annotations)}）")
            else:
                # 如果样本不足，使用全部样本
                annotations.extend(disaster_class_annotations)
                print(f"  {disaster_type}: {len(disaster_class_annotations)} 个样本（目标: {target_count}，可用: {len(disaster_class_annotations)}，样本不足使用全部）")
        
        print(f"\n总样本数: {len(annotations)}")
    elif balance_classes:
        # 如果需要平衡类别
        print("\n使用类别平衡采样...")
        annotations = processor.annotations
        
        # 按类别分组
        class_groups = defaultdict(list)
        for ann in annotations:
            class_groups[ann.disaster_type].append(ann)
        
        # 计算每个类别的目标样本数
        num_classes = len(class_groups)
        target_samples_per_class = min([len(anns) for anns in class_groups.values()])
        print(f"每个类别的目标样本数: {target_samples_per_class}")
        
        # 对每个类别进行采样
        balanced_annotations = []
        for disaster_type, anns in class_groups.items():
            if len(anns) >= target_samples_per_class:
                # 如果样本足够，随机采样
                np.random.shuffle(anns)
                balanced_annotations.extend(anns[:target_samples_per_class])
            else:
                # 如果样本不足，使用全部样本
                balanced_annotations.extend(anns)
                print(f"警告: {disaster_type} 类别样本不足（{len(anns)}），使用全部样本")
        
        print(f"平衡后总样本数: {len(balanced_annotations)}")
        annotations = balanced_annotations
    elif max_samples and len(processor.annotations) > max_samples:
        # 如果指定了最大样本数，进行采样
        print(f"\n从 {len(processor.annotations)} 个样本中采样 {max_samples} 个")
        annotations = processor.annotations[:max_samples]
    else:
        # 使用全部样本
        annotations = processor.annotations
    
    # 划分训练集和测试集
    np.random.shuffle(annotations)
    
    split_idx = int(len(annotations) * (1 - test_size))
    train_annotations = annotations[:split_idx]
    test_annotations = annotations[split_idx:]
    
    print(f"\n训练集: {len(train_annotations)} 个样本")
    print(f"测试集: {len(test_annotations)} 个样本")
    
    # 显示训练集和测试集的类别分布
    print("\n训练集类别分布:")
    train_distribution = defaultdict(int)
    for ann in train_annotations:
        train_distribution[ann.disaster_type] += 1
    for disaster_type, count in sorted(train_distribution.items()):
        print(f"  {disaster_type}: {count}")
    
    print("\n测试集类别分布:")
    test_distribution = defaultdict(int)
    for ann in test_annotations:
        test_distribution[ann.disaster_type] += 1
    for disaster_type, count in sorted(test_distribution.items()):
        print(f"  {disaster_type}: {count}")
    
    return train_annotations, test_annotations, processor


if __name__ == '__main__':
    # 测试数据处理器
    data_dir = 'e:/机器学习大作业'
    annotation_file = 'e:/机器学习大作业/总标注.xlsx'
    
    train_ann, test_ann, processor = prepare_dataset(
        data_dir, annotation_file, test_size=0.2
    )
    
    # 显示有时间信息的样本
    print("\n有时间信息的样本（前5个）:")
    time_samples = processor.get_time_info_samples(5)
    for sample in time_samples:
        print(sample)
