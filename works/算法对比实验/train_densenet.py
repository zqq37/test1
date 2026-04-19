import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

sys.path.append('e:/机器学习大作业')
from data_processor import DisasterVideoDataset, DisasterAnnotation

class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(DenseNetClassifier, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        
        for param in self.densenet.parameters():
            param.requires_grad = False
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x的形状是[batch, seq_len, height, width, channels]
        # 对32帧进行平均池化，得到[batch, height, width, channels]
        x = x.mean(dim=1)
        # 转换为[batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        return self.densenet(x)

def load_annotations(csv_file):
    df = pd.read_csv(csv_file)
    annotations = []
    for _, row in df.iterrows():
        annotations.append(DisasterAnnotation(row['video_path'], row['disaster_type']))
    return annotations

def custom_collate_fn(batch):
    result = {}
    for key in batch[0]:
        if key == 'frames':
            result[key] = torch.stack([item[key] for item in batch])
        elif key == 'label':
            result[key] = torch.cat([item[key] for item in batch], dim=0)
        else:
            result[key] = [item[key] for item in batch]
    return result

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name):
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for batch_idx, batch in enumerate(train_bar):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
            for batch_idx, batch in enumerate(val_bar):
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Epoch Time: {epoch_time:.2f}s')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = 'e:/机器学习大作业/算法对比实验/models'
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, f'{model_name}_best.pth'))
            print(f'  Saved best model with val acc: {best_val_acc:.2f}%')
    
    return history, best_val_acc

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (all_labels == i)
        if class_mask.sum() > 0:
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            class_accuracy[class_name] = class_acc
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'class_accuracy': class_accuracy,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    data_dir = 'e:/机器学习大作业/算法对比实验'
    train_csv = os.path.join(data_dir, 'train_420.csv')
    val_csv = os.path.join(data_dir, 'val_420.csv')
    test_csv = os.path.join(data_dir, 'test_420.csv')
    
    if not os.path.exists(train_csv):
        print("数据集文件不存在，请先运行 prepare_data_420.py")
        return
    
    print("加载训练集...")
    train_annotations = load_annotations(train_csv)
    print(f"训练集样本数: {len(train_annotations)}")
    
    print("加载验证集...")
    val_annotations = load_annotations(val_csv)
    print(f"验证集样本数: {len(val_annotations)}")
    
    print("加载测试集...")
    test_annotations = load_annotations(test_csv)
    print(f"测试集样本数: {len(test_annotations)}")
    
    train_dataset = DisasterVideoDataset(train_annotations, data_dir='e:/机器学习大作业', max_frames=32)
    val_dataset = DisasterVideoDataset(val_annotations, data_dir='e:/机器学习大作业', max_frames=32)
    test_dataset = DisasterVideoDataset(test_annotations, data_dir='e:/机器学习大作业', max_frames=32)
    
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    model = DenseNetClassifier(num_classes=7, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    model_name = 'DenseNet'
    num_epochs = 8
    
    print(f"\n开始训练 {model_name} 模型...")
    print(f"训练轮数: {num_epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: 0.0001")
    
    history, best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name)
    
    print(f"\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load(os.path.join(data_dir, 'models', f'{model_name}_best.pth'), map_location=device))
    
    class_names = ['normal', 'landslide', 'debrisFlow', 'forestfire', 'mountainFloods', 'riverOverflow', 'waterlogging']
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    
    print(f"\n{model_name} 测试集结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  精确率 (宏平均): {test_metrics['precision_macro']:.4f}")
    print(f"  召回率 (宏平均): {test_metrics['recall_macro']:.4f}")
    print(f"  F1分数 (宏平均): {test_metrics['f1_macro']:.4f}")
    print(f"\n各类别准确率:")
    for class_name, acc in test_metrics['class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")
    
    results = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'training_history': history,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': 0.0001,
            'num_epochs': num_epochs,
            'optimizer': 'AdamW',
            'scheduler': 'ReduceLROnPlateau'
        }
    }
    
    results_dir = 'e:/机器学习大作业/算法对比实验/results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f'{model_name}_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {os.path.join(results_dir, f'{model_name}_results.json')}")

if __name__ == '__main__':
    main()
