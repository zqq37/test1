# 算法对比实验

## 实验概述

本实验对比四种深度学习算法在灾害视频检测任务上的性能：
- **ResNet50**: 深度残差网络
- **VGG16**: 视觉几何组网络
- **EfficientNet**: 高效卷积神经网络
- **DenseNet**: 密集连接网络

## 数据集

使用平衡数据集，共420个样本，每个类别60个样本：
- Normal（正常）: 60
- Landslide（山体滑坡）: 60
- Debris Flow（泥石流）: 60
- Forest Fire（森林火灾）: 60
- Mountain Floods（山洪）: 60
- River Overflow（流域性洪水）: 60
- Waterlogging（城市内涝）: 60

数据集划分：
- 训练集: 70% (294个样本)
- 验证集: 15% (63个样本)
- 测试集: 15% (63个样本)

## 使用方法

### 1. 准备数据集

```bash
cd e:/机器学习大作业/算法对比实验
python prepare_data_420.py
```

该脚本会：
- 从总标注文件中选择420个平衡样本
- 按照训练集、验证集、测试集划分
- 保存为CSV文件

### 2. 训练各个算法

#### ResNet50

```bash
python train_resnet50.py
```

#### VGG16

```bash
python train_vgg16.py
```

#### EfficientNet

```bash
python train_efficientnet.py
```

#### DenseNet

```bash
python train_densenet.py
```

每个训练脚本会：
- 训练30个epoch
- 使用AdamW优化器，学习率0.0001
- 使用ReduceLROnPlateau学习率调度器
- 保存最佳模型权重
- 在测试集上评估性能
- 保存训练历史和测试结果到JSON文件

### 3. 算法对比分析

```bash
python compare_algorithms.py
```

该脚本会：
- 加载所有算法的训练结果
- 生成性能对比表
- 使用层次分析法(AHP)计算综合评分
- 生成对比图表：
  - 算法性能对比图
  - 训练曲线对比图
  - 混淆矩阵对比图
- 生成综合报告

## 评估指标

### 性能指标
- **准确率 (Accuracy)**: 整体分类准确率
- **精确率 (Precision)**: 宏平均精确率
- **召回率 (Recall)**: 宏平均召回率
- **F1分数 (F1-Score)**: 宏平均F1分数

### 效率指标
- **平均每轮训练时间**: 单个epoch的平均训练时间
- **总训练时间**: 完整训练过程的累计时间

## 层次分析法 (AHP)

使用层次分析法对算法进行综合评分，准则权重如下：

| 准则 | 权重 | 说明 |
|------|------|------|
| Test Accuracy | 0.35 | 测试集准确率 |
| Test F1-Score | 0.25 | 测试集F1分数 |
| Test Precision | 0.15 | 测试集精确率 |
| Test Recall | 0.15 | 测试集召回率 |
| Total Training Time | 0.10 | 总训练时间（越小越好） |

## 输出文件

### 数据集文件
- `train_420.csv`: 训练集标注
- `val_420.csv`: 验证集标注
- `test_420.csv`: 测试集标注

### 模型文件
- `models/ResNet50_best.pth`: ResNet50最佳模型
- `models/VGG16_best.pth`: VGG16最佳模型
- `models/EfficientNet_best.pth`: EfficientNet最佳模型
- `models/DenseNet_best.pth`: DenseNet最佳模型

### 结果文件
- `results/ResNet50_results.json`: ResNet50训练结果
- `results/VGG16_results.json`: VGG16训练结果
- `results/EfficientNet_results.json`: EfficientNet训练结果
- `results/DenseNet_results.json`: DenseNet训练结果

### 对比分析文件
- `results/algorithm_comparison.png`: 算法性能对比图
- `results/training_curves_comparison.png`: 训练曲线对比图
- `results/confusion_matrices_comparison.png`: 混淆矩阵对比图
- `results/comparison_report.txt`: 综合对比报告

## 训练参数

所有算法使用相同的训练参数：
- **Epochs**: 8
- **Batch Size**: 8
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau
- **Loss Function**: CrossEntropyLoss
- **Image Size**: 224×224×3
- **Num Frames**: 32

## 模型架构

### ResNet50
- 预训练ResNet50（冻结特征提取器）
- 全局平均池化
- 全连接层: 256 + ReLU + Dropout(0.5)
- 输出层: 7类

### VGG16
- 预训练VGG16（冻结特征提取器）
- 全连接层: 256 + ReLU + Dropout(0.5)
- 输出层: 7类

### EfficientNet-B0
- 预训练EfficientNet-B0（冻结特征提取器）
- 全连接层: 256 + ReLU + Dropout(0.5)
- 输出层: 7类

### DenseNet121
- 预训练DenseNet121（冻结特征提取器）
- 全连接层: 256 + ReLU + Dropout(0.5)
- 输出层: 7类

## 注意事项

1. **数据准备**: 必须先运行`prepare_data_420.py`准备数据集
2. **训练顺序**: 可以并行运行不同算法的训练脚本
3. **GPU支持**: 建议使用GPU加速训练
4. **结果保存**: 所有结果会自动保存到相应目录
5. **对比分析**: 需要所有算法训练完成后才能运行对比分析

## 预期结果

根据花卉识别系统的经验，预期结果：
- **ResNet50**: 平衡性能和效率
- **VGG16**: 参数量大，可能较慢
- **EfficientNet**: 高效，可能性能较好
- **DenseNet**: 密集连接，可能性能较好

最终推荐算法将基于层次分析法的综合评分确定。
