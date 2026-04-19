import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import json

sys.path.append('e:/机器学习大作业')
from data_processor import DisasterAnnotation, DisasterDatasetProcessor

def prepare_balanced_dataset_420():
    """
    准备平衡数据集，共420个样本，每个类别60个
    """
    print("开始准备平衡数据集（420个样本，每个类别60个）...")
    
    processor = DisasterDatasetProcessor(
        data_dir='e:/机器学习大作业',
        annotation_file='e:/机器学习大作业/总标注.xlsx'
    )
    
    class_sample_counts = {
        'normal': 60,
        'forestfire': 60,
        'waterlogging': 60,
        'debrisFlow': 60,
        'mountainFloods': 60,
        'landslide': 60,
        'riverOverflow': 60
    }
    
    annotations = []
    for disaster_type, target_count in class_sample_counts.items():
        disaster_class_annotations = [ann for ann in processor.annotations if ann.disaster_type == disaster_type]
        print(f"类别 {disaster_type}: 找到 {len(disaster_class_annotations)} 个样本，选择 {target_count} 个")
        
        if len(disaster_class_annotations) >= target_count:
            np.random.shuffle(disaster_class_annotations)
            selected_annotations = disaster_class_annotations[:target_count]
            annotations.extend(selected_annotations)
        else:
            print(f"警告: 类别 {disaster_type} 的样本数不足 {target_count}，使用全部 {len(disaster_class_annotations)} 个")
            annotations.extend(disaster_class_annotations)
    
    print(f"总共选择了 {len(annotations)} 个样本")
    
    np.random.shuffle(annotations)
    
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    
    train_size = int(len(annotations) * train_split)
    val_size = int(len(annotations) * val_split)
    
    train_annotations = annotations[:train_size]
    val_annotations = annotations[train_size:train_size + val_size]
    test_annotations = annotations[train_size + val_size:]
    
    print(f"训练集: {len(train_annotations)} 个样本")
    print(f"验证集: {len(val_annotations)} 个样本")
    print(f"测试集: {len(test_annotations)} 个样本")
    
    def count_classes(ann_list):
        counts = defaultdict(int)
        for ann in ann_list:
            counts[ann.disaster_type] += 1
        return dict(counts)
    
    print("\n训练集类别分布:")
    train_counts = count_classes(train_annotations)
    for cls, count in sorted(train_counts.items()):
        print(f"  {cls}: {count}")
    
    print("\n验证集类别分布:")
    val_counts = count_classes(val_annotations)
    for cls, count in sorted(val_counts.items()):
        print(f"  {cls}: {count}")
    
    print("\n测试集类别分布:")
    test_counts = count_classes(test_annotations)
    for cls, count in sorted(test_counts.items()):
        print(f"  {cls}: {count}")
    
    output_dir = 'e:/机器学习大作业/算法对比实验'
    os.makedirs(output_dir, exist_ok=True)
    
    def save_annotations(ann_list, filename):
        data = []
        for ann in ann_list:
            data.append({
                'video_path': ann.video_name,
                'disaster_type': ann.disaster_type,
                'start_frame': 0,
                'end_frame': 0
            })
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8-sig')
        print(f"已保存 {filename}: {len(data)} 个样本")
    
    save_annotations(train_annotations, 'train_420.csv')
    save_annotations(val_annotations, 'val_420.csv')
    save_annotations(test_annotations, 'test_420.csv')
    
    print(f"\n数据集准备完成！文件保存在: {output_dir}")
    
    return train_annotations, val_annotations, test_annotations

if __name__ == '__main__':
    train_ann, val_ann, test_ann = prepare_balanced_dataset_420()
