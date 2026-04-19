"""
改进的训练脚本
解决训练波动大的问题
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from datetime import datetime
import json

from data_processor import DisasterVideoDataset, prepare_dataset, collate_fn
from improved_model import DisasterTimePredictor, AttentionDisasterPredictor, DisasterLoss
from feature_extractor import FrameFeatureExtractor


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                grad_clip: float = 1.0):
    """训练一个epoch，添加梯度裁剪"""
    model.train()
    running_loss = 0.0
    running_d_loss = 0.0
    running_t_loss = 0.0
    running_ew_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for batch in pbar:
        frames = batch['frames'].to(device)
        disaster_labels = batch['labels'].to(device)
        
        # 时间标签（如果有）
        if 'time_labels' in batch:
            start_time_labels = batch['time_labels'][:, 0].to(device)
            end_time_labels = batch['time_labels'][:, 1].to(device)
        else:
            start_time_labels = torch.full((frames.size(0),), -1, device=device)
            end_time_labels = torch.full((frames.size(0),), -1, device=device)
        
        # 早期预警标签（如果有）
        if 'early_warning_labels' in batch:
            early_warning_labels = batch['early_warning_labels'].to(device)
        else:
            early_warning_labels = torch.full((frames.size(0),), -1, device=device)
        
        optimizer.zero_grad()
        
        # 提取特征
        with torch.no_grad():
            frame_extractor = FrameFeatureExtractor().to(device)
            frame_extractor.eval()
            batch_size = frames.size(0)
            seq_len = frames.size(1)
            # frames的形状是[batch_size, seq_len, 224, 224, 3]
            # 需要转换为[batch_size*seq_len, 3, 224, 224]
            frames_flat = frames.permute(0, 1, 4, 2, 3).contiguous()
            frames_flat = frames_flat.view(-1, 3, 224, 224)
            features = frame_extractor(frames_flat)
            features = features.view(batch_size, seq_len, -1)
            # 平均池化
            video_features = features.mean(dim=1)
        
        # 前向传播
        disaster_logits, start_time_logits, end_time_logits, early_warning_logits = model(video_features)
        
        # 计算损失
        loss, d_loss, t_loss, ew_loss = criterion(
            disaster_logits, disaster_labels,
            start_time_logits, end_time_logits,
            start_time_labels, end_time_labels,
            early_warning_logits, early_warning_labels
        )
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        running_d_loss += d_loss.item()
        running_t_loss += t_loss.item()
        running_ew_loss += ew_loss.item()
        
        _, predicted = torch.max(disaster_logits.data, 1)
        total += disaster_labels.size(0)
        correct += (predicted == disaster_labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(disaster_labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'd_loss': f'{d_loss.item():.4f}',
            't_loss': f'{t_loss.item():.4f}',
            'ew_loss': f'{ew_loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_d_loss = running_d_loss / len(dataloader)
    epoch_t_loss = running_t_loss / len(dataloader)
    epoch_ew_loss = running_ew_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_d_loss, epoch_t_loss, epoch_ew_loss, epoch_acc, all_preds, all_labels


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_d_loss = 0.0
    running_t_loss = 0.0
    running_ew_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_time_preds_start = []
    all_time_preds_end = []
    all_time_labels_start = []
    all_time_labels_end = []
    all_early_preds = []
    all_early_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch in pbar:
            frames = batch['frames'].to(device)
            disaster_labels = batch['labels'].to(device)
            
            # 时间标签（如果有）
            if 'time_labels' in batch:
                start_time_labels = batch['time_labels'][:, 0].to(device)
                end_time_labels = batch['time_labels'][:, 1].to(device)
            else:
                start_time_labels = torch.full((frames.size(0),), -1, device=device)
                end_time_labels = torch.full((frames.size(0),), -1, device=device)
            
            # 早期预警标签（如果有）
            if 'early_warning_labels' in batch:
                early_warning_labels = batch['early_warning_labels'].to(device)
            else:
                early_warning_labels = torch.full((frames.size(0),), -1, device=device)
            
            # 提取特征
            frame_extractor = FrameFeatureExtractor().to(device)
            frame_extractor.eval()
            batch_size = frames.size(0)
            seq_len = frames.size(1)
            frames_flat = frames.permute(0, 1, 4, 2, 3).contiguous()
            frames_flat = frames_flat.view(-1, 3, 224, 224)
            features = frame_extractor(frames_flat)
            features = features.view(batch_size, seq_len, -1)
            video_features = features.mean(dim=1)
            
            # 前向传播
            disaster_logits, start_time_logits, end_time_logits, early_warning_logits = model(video_features)
            
            # 计算损失
            loss, d_loss, t_loss, ew_loss = criterion(
                disaster_logits, disaster_labels,
                start_time_logits, end_time_logits,
                start_time_labels, end_time_labels,
                early_warning_logits, early_warning_labels
            )
            
            # 统计
            running_loss += loss.item()
            running_d_loss += d_loss.item()
            running_t_loss += t_loss.item()
            running_ew_loss += ew_loss.item()
            
            _, predicted = torch.max(disaster_logits.data, 1)
            total += disaster_labels.size(0)
            correct += (predicted == disaster_labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(disaster_labels.cpu().numpy())
            
            # 时间预测
            valid_time_mask = (start_time_labels >= 0) & (end_time_labels >= 0)
            if valid_time_mask.sum() > 0:
                all_time_preds_start.extend(start_time_logits[valid_time_mask].cpu().numpy())
                all_time_preds_end.extend(end_time_logits[valid_time_mask].cpu().numpy())
                all_time_labels_start.extend(start_time_labels[valid_time_mask].cpu().numpy())
                all_time_labels_end.extend(end_time_labels[valid_time_mask].cpu().numpy())
            
            # 早期预警
            valid_early_mask = (early_warning_labels >= 0)
            if valid_early_mask.sum() > 0:
                _, early_predicted = torch.max(early_warning_logits[valid_early_mask], 1)
                all_early_preds.extend(early_predicted.cpu().numpy())
                all_early_labels.extend(early_warning_labels[valid_early_mask].cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'd_loss': f'{d_loss.item():.4f}',
                't_loss': f'{t_loss.item():.4f}',
                'ew_loss': f'{ew_loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_d_loss = running_d_loss / len(dataloader)
    epoch_t_loss = running_t_loss / len(dataloader)
    epoch_ew_loss = running_ew_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # 计算时间预测准确度（1秒内）
    if len(all_time_preds_start) > 0:
        time_preds_start = np.array(all_time_preds_start)
        time_preds_end = np.array(all_time_preds_end)
        time_labels_start = np.array(all_time_labels_start)
        time_labels_end = np.array(all_time_labels_end)
        
        # 将预测转换为秒
        time_preds_start_sec = (torch.sigmoid(torch.from_numpy(time_preds_start)) * 30.0).numpy()
        time_preds_end_sec = (torch.sigmoid(torch.from_numpy(time_preds_end)) * 30.0).numpy()
        time_labels_start_sec = time_labels_start / 30.0
        time_labels_end_sec = time_labels_end / 30.0
        
        # 计算准确度（相差1秒内就算成功）
        start_acc_1s = (np.abs(time_preds_start_sec - time_labels_start_sec) <= 1.0).mean()
        end_acc_1s = (np.abs(time_preds_end_sec - time_labels_end_sec) <= 1.0).mean()
        time_acc_1s = (start_acc_1s + end_acc_1s) / 2
    else:
        time_acc_1s = 0.0
    
    # 计算早期预警准确度
    if len(all_early_preds) > 0:
        early_acc = (np.array(all_early_preds) == np.array(all_early_labels)).mean()
    else:
        early_acc = 0.0
    
    return (epoch_loss, epoch_d_loss, epoch_t_loss, epoch_ew_loss, epoch_acc, 
            all_preds, all_labels,
            all_time_preds_start, all_time_preds_end,
            all_time_labels_start, all_time_labels_end,
            all_early_preds, all_early_labels,
            time_acc_1s, early_acc)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cm


def plot_training_history(history, save_path):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 灾害分类损失
    axes[0, 1].plot(history['train_d_loss'], label='Train Disaster Loss')
    axes[0, 1].plot(history['val_d_loss'], label='Val Disaster Loss')
    axes[0, 1].set_title('Disaster Classification Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 时间预测损失
    axes[1, 0].plot(history['train_t_loss'], label='Train Time Loss')
    axes[1, 0].plot(history['val_t_loss'], label='Val Time Loss')
    axes[1, 0].set_title('Time Prediction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 准确率曲线
    axes[1, 1].plot(history['train_acc'], label='Train Acc')
    axes[1, 1].plot(history['val_acc'], label='Val Acc')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def calculate_metrics(y_true, y_pred, class_names):
    """计算评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # 每个类别的指标
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                  output_dict=True, zero_division=0)
    
    return metrics, report


def main():
    parser = argparse.ArgumentParser(description='改进的训练脚本')
    parser.add_argument('--data_dir', type=str, default='e:/机器学习大作业',
                       help='数据目录路径')
    parser.add_argument('--annotation_file', type=str, default='e:/机器学习大作业/总标注.xlsx',
                       help='标注文件路径')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'attention'],
                       help='模型类型')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--max_frames', type=int, default=32,
                       help='最大帧数')
    parser.add_argument('--frame_sample_rate', type=int, default=1,
                       help='帧采样率')
    parser.add_argument('--max_samples', type=int, default=400,
                       help='最大样本数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_stable',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='计算设备')
    parser.add_argument('--include_time', action='store_true',
                       help='是否包含时间预测')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--time_loss_weight', type=float, default=0.1,
                       help='时间损失权重')
    parser.add_argument('--balance_classes', action='store_true',
                       help='是否平衡类别（每个类别使用相同数量的样本）')
    parser.add_argument('--class_sample_counts', type=str, default=None,
                       help='每个类别的样本数（格式：normal:200,forestfire:200,waterlogging:190,debrisFlow:100,mountainFloods:100,landslide:100,riverOverflow:60）')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据集
    print("准备数据集...")
    
    # 解析类别样本数
    class_sample_counts = None
    if args.class_sample_counts:
        class_sample_counts = {}
        for item in args.class_sample_counts.split(','):
            key, value = item.split(':')
            class_sample_counts[key.strip()] = int(value.strip())
        print(f"使用指定的类别样本数: {class_sample_counts}")
    
    train_annotations, test_annotations, processor = prepare_dataset(
        args.data_dir, args.annotation_file, 
        test_size=0.2, max_samples=args.max_samples, 
        balance_classes=args.balance_classes,
        class_sample_counts=class_sample_counts
    )
    
    # 创建数据集
    train_dataset = DisasterVideoDataset(
        train_annotations, args.data_dir,
        max_frames=args.max_frames,
        frame_sample_rate=args.frame_sample_rate,
        include_time_prediction=args.include_time
    )
    test_dataset = DisasterVideoDataset(
        test_annotations, args.data_dir,
        max_frames=args.max_frames,
        frame_sample_rate=args.frame_sample_rate,
        include_time_prediction=args.include_time
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # 创建模型
    num_classes = 7
    device = torch.device(args.device)
    
    if args.model_type == 'simple':
        model = DisasterTimePredictor(
            frame_feature_dim=2048,
            num_classes=num_classes
        ).to(device)
    elif args.model_type == 'attention':
        model = AttentionDisasterPredictor(
            frame_feature_dim=2048,
            num_classes=num_classes
        ).to(device)
    
    print(f"\n模型类型: {args.model_type}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):}")
    
    # 计算类别权重（处理类别不平衡）
    class_counts = [len([ann for ann in train_annotations if ann.disaster_type == cls]) 
                   for cls in ['normal', 'landslide', 'debrisFlow', 'forestfire', 
                              'mountainFloods', 'riverOverflow', 'waterlogging']]
    # 使用更平滑的权重
    total_samples = sum(class_counts)
    class_weights = torch.FloatTensor([total_samples / (count + 1e-6) for count in class_counts])
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    
    print(f"类别权重: {class_weights}")
    
    # 损失函数和优化器
    criterion = DisasterLoss(disaster_weight=class_weights, time_loss_weight=args.time_loss_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 训练循环
    best_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_d_loss': [], 'val_d_loss': [],
        'train_t_loss': [], 'val_t_loss': [],
        'train_ew_loss': [], 'val_ew_loss': [],
        'train_acc': [], 'val_acc': [],
        'time_acc_1s': [], 'early_acc': []
    }
    
    class_names = ['normal', 'landslide', 'debrisFlow', 'forestfire', 
                 'mountainFloods', 'riverOverflow', 'waterlogging']
    
    print("\n开始训练...")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度裁剪: {args.grad_clip}")
    print(f"时间损失权重: {args.time_loss_weight}")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*60)
        
        # 训练
        train_loss, train_d_loss, train_t_loss, train_ew_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, args.grad_clip
        )
        
        # 验证
        (val_loss, val_d_loss, val_t_loss, val_ew_loss, val_acc, 
         val_preds, val_labels, 
         all_time_preds_start, all_time_preds_end,
         all_time_labels_start, all_time_labels_end,
         all_early_preds, all_early_labels,
         time_acc_1s, early_acc) = validate(
            model, test_loader, criterion, device
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_d_loss'].append(train_d_loss)
        history['val_d_loss'].append(val_d_loss)
        history['train_t_loss'].append(train_t_loss)
        history['val_t_loss'].append(val_t_loss)
        history['train_ew_loss'].append(train_ew_loss)
        history['val_ew_loss'].append(val_ew_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['time_acc_1s'].append(time_acc_1s)
        history['early_acc'].append(early_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'model_type': args.model_type,
                'class_names': class_names
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"\n保存最佳模型，验证准确率: {best_acc:.2f}%")
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
    
    # 最终评估
    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
    print("\n进行最终评估...")
    
    (val_loss, val_d_loss, val_t_loss, val_ew_loss, val_acc, 
     val_preds, val_labels, 
     all_time_preds_start, all_time_preds_end,
     all_time_labels_start, all_time_labels_end,
     all_early_preds, all_early_labels,
     time_acc_1s, early_acc) = validate(
        model, test_loader, criterion, device
    )
    
    # 计算评估指标
    metrics, report = calculate_metrics(val_labels, val_preds, class_names)
    
    print("\n评估指标:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision_macro']:.4f}")
    print(f"召回率: {metrics['recall_macro']:.4f}")
    print(f"F1分数: {metrics['f1_macro']:.4f}")
    print(f"时间预测准确率（1秒内）: {time_acc_1s:.2%}")
    print(f"早期预警准确率: {early_acc:.2%}")
    
    print("\n分类报告:")
    for cls in class_names:
        if cls in report:
            print(f"{cls}:")
            print(f"  精确率: {report[cls]['precision']:.4f}")
            print(f"  召回率: {report[cls]['recall']:.4f}")
            print(f"  F1分数: {report[cls]['f1-score']:.4f}")
    
    # 保存结果
    results = {
        'best_acc': best_acc,
        'final_metrics': metrics,
        'classification_report': report,
        'history': history
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成可视化结果
    print("\n生成可视化结果...")
    viz_script_path = 'e:/机器学习大作业/visualize_results.py'
    if os.path.exists(viz_script_path):
        import subprocess
        subprocess.run(['python', viz_script_path], cwd='e:/机器学习大作业')
        print(f"可视化结果已保存到: {save_dir}")
    else:
        print(f"警告: 可视化脚本不存在: {viz_script_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, cm_path)
    print(f"\n混淆矩阵已保存到: {cm_path}")
    
    # 绘制训练历史
    history_path = os.path.join(save_dir, 'training_history.png')
    plot_training_history(history, history_path)
    print(f"训练历史已保存到: {history_path}")
    
    print(f"\n所有结果已保存到: {save_dir}")


if __name__ == '__main__':
    main()
