import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False

class AlgorithmComparator:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.models = ['ResNet50', 'VGG16', 'EfficientNet', 'DenseNet']
        self.results = {}
        self.load_results()
    
    def load_results(self):
        for model_name in self.models:
            result_file = os.path.join(self.results_dir, f'{model_name}_results.json')
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    self.results[model_name] = json.load(f)
                print(f"加载 {model_name} 结果成功")
            else:
                print(f"警告: {model_name} 结果文件不存在")
    
    def create_comparison_table(self):
        data = []
        for model_name in self.models:
            if model_name in self.results:
                result = self.results[model_name]
                test_metrics = result['test_metrics']
                history = result['training_history']
                
                avg_train_time = np.mean(history['epoch_time'])
                total_train_time = np.sum(history['epoch_time'])
                
                data.append({
                    'Model': model_name,
                    'Validation Accuracy (%)': result['best_val_acc'],
                    'Test Accuracy': test_metrics['accuracy'],
                    'Test Precision': test_metrics['precision_macro'],
                    'Test Recall': test_metrics['recall_macro'],
                    'Test F1-Score': test_metrics['f1_macro'],
                    'Avg Epoch Time (s)': avg_train_time,
                    'Total Training Time (s)': total_train_time
                })
        
        df = pd.DataFrame(data)
        return df
    
    def calculate_gulin_weights(self, criteria):
        """
        使用古林法计算权重
        """
        num_criteria = len(criteria)
        weights = {}
        
        print("\n=== 古林法计算权重 ===")
        
        for i, criterion in enumerate(criteria):
            print(f"\n准则 {criterion}:")
            
            comparisons = {}
            for j, model in enumerate(self.models):
                if model in self.results:
                    comparisons[model] = 1.0
            
            for i in range(len(self.models)):
                for j in range(i+1, len(self.models)):
                    model1 = self.models[i]
                    model2 = self.models[j]
                    
                    if model1 in self.results and model2 in self.results:
                        val1 = self.get_criterion_value(model1, criterion)
                        val2 = self.get_criterion_value(model2, criterion)
                        
                        if criterion in ['Avg Epoch Time (s)', 'Total Training Time (s)']:
                            ratio = val2 / val1 if val1 > 0 else 1.0
                        else:
                            ratio = val1 / val2 if val2 > 0 else 1.0
                        
                        comparisons[model1] *= ratio
                        comparisons[model2] *= (1.0 / ratio if ratio > 0 else 1.0)
            
            total = sum(comparisons.values())
            for model in comparisons:
                weights[model] = comparisons[model] / total if total > 0 else 0
            
            print(f"  权重: {weights}")
        
        return weights
    
    def get_criterion_value(self, model, criterion):
        if model not in self.results:
            return 0.0
        
        result = self.results[model]
        test_metrics = result['test_metrics']
        history = result['training_history']
        
        if criterion == 'Validation Accuracy (%)':
            return result['best_val_acc']
        elif criterion == 'Test Accuracy':
            return test_metrics['accuracy']
        elif criterion == 'Test Precision':
            return test_metrics['precision_macro']
        elif criterion == 'Test Recall':
            return test_metrics['recall_macro']
        elif criterion == 'Test F1-Score':
            return test_metrics['f1_macro']
        elif criterion == 'Avg Epoch Time (s)':
            return np.mean(history['epoch_time'])
        elif criterion == 'Total Training Time (s)':
            return np.sum(history['epoch_time'])
        
        return 0.0
    
    def calculate_ahp_weights(self, criteria):
        """
        使用层次分析法计算权重
        """
        print("\n=== 层次分析法 (AHP) ===")
        
        criteria_weights = {
            'Test Accuracy': 0.35,
            'Test F1-Score': 0.25,
            'Test Precision': 0.15,
            'Test Recall': 0.15,
            'Total Training Time (s)': 0.10
        }
        
        print("准则权重:")
        for criterion, weight in criteria_weights.items():
            print(f"  {criterion}: {weight}")
        
        model_scores = {}
        for model in self.models:
            if model in self.results:
                total_score = 0.0
                for criterion, weight in criteria_weights.items():
                    value = self.get_criterion_value(model, criterion)
                    
                    if criterion == 'Total Training Time (s)':
                        normalized_value = 1.0 / (value + 1e-6)
                    else:
                        normalized_value = value
                    
                    total_score += weight * normalized_value
                
                model_scores[model] = total_score
        
        total = sum(model_scores.values())
        final_weights = {k: v/total if total > 0 else 0 for k, v in model_scores.items()}
        
        print("\n最终得分:")
        for model, score in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {score:.4f}")
        
        return final_weights
    
    def plot_comparison(self):
        df = self.create_comparison_table()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Comparison Results', fontsize=16, fontweight='bold')
        
        metrics = ['Validation Accuracy (%)', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score']
        
        ax1 = axes[0, 0]
        x = np.arange(len(df['Model']))
        width = 0.15
        
        for i, metric in enumerate(metrics[:3]):
            offset = (i - 1) * width
            ax1.bar(x + offset, df[metric], width, label=metric)
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Model'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for i, metric in enumerate(metrics[3:]):
            offset = (i - 0.5) * width
            ax2.bar(x + offset, df[metric], width, label=metric)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Score')
        ax2.set_title('Additional Performance Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Model'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.bar(df['Model'], df['Avg Epoch Time (s)'], color='steelblue')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Average Epoch Time')
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = axes[1, 1]
        ax4.bar(df['Model'], df['Total Training Time (s)'], color='coral')
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Total Training Time')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'algorithm_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存到: {output_file}")
        
        return fig
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        ax1 = axes[0, 0]
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['train_loss']) + 1)
                ax1.plot(epochs, history['train_loss'], label=model_name, color=colors[i], linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['val_loss']) + 1)
                ax2.plot(epochs, history['val_loss'], label=model_name, color=colors[i], linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['train_acc']) + 1)
                ax3.plot(epochs, history['train_acc'], label=model_name, color=colors[i], linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Training Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['val_acc']) + 1)
                ax4.plot(epochs, history['val_acc'], label=model_name, color=colors[i], linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Validation Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'training_curves_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"训练曲线对比图已保存到: {output_file}")
        
        return fig
    
    def plot_confusion_matrices(self):
        class_names = ['Normal', 'Landslide', 'Debris Flow', 'Forest Fire', 'Mountain Floods', 'River Overflow', 'Waterlogging']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, model_name in enumerate(self.models):
            if model_name in self.results and i < len(axes_flat):
                cm = np.array(self.results[model_name]['test_metrics']['confusion_matrix'])
                
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names,
                           ax=axes_flat[i], cbar_kws={'label': 'Normalized Score'})
                
                axes_flat[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
                axes_flat[i].set_xlabel('Predicted Label')
                axes_flat[i].set_ylabel('True Label')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'confusion_matrices_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵对比图已保存到: {output_file}")
        
        return fig
    
    def generate_report(self):
        print("\n" + "="*60)
        print("算法对比实验报告")
        print("="*60)
        
        df = self.create_comparison_table()
        print("\n【性能对比表】")
        print(df.to_string(index=False))
        
        print("\n【层次分析法 (AHP) 综合评分】")
        ahp_weights = self.calculate_ahp_weights([])
        
        print("\n【最佳算法推荐】")
        best_model = max(ahp_weights.items(), key=lambda x: x[1])
        print(f"根据综合评分，最佳算法为: {best_model[0]} (得分: {best_model[1]:.4f})")
        
        ranking = sorted(ahp_weights.items(), key=lambda x: x[1], reverse=True)
        print("\n算法排名:")
        for i, (model, score) in enumerate(ranking, 1):
            print(f"  {i}. {model}: {score:.4f}")
        
        report_file = os.path.join(self.results_dir, 'comparison_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("算法对比实验报告\n")
            f.write("="*60 + "\n\n")
            
            f.write("【性能对比表】\n")
            f.write(df.to_string(index=False) + "\n\n")
            
            f.write("【层次分析法 (AHP) 综合评分】\n")
            f.write("准则权重:\n")
            f.write("  Test Accuracy: 0.35\n")
            f.write("  Test F1-Score: 0.25\n")
            f.write("  Test Precision: 0.15\n")
            f.write("  Test Recall: 0.15\n")
            f.write("  Total Training Time: 0.10\n\n")
            
            f.write("算法排名:\n")
            for i, (model, score) in enumerate(ranking, 1):
                f.write(f"  {i}. {model}: {score:.4f}\n")
            
            f.write(f"\n【最佳算法推荐】\n")
            f.write(f"根据综合评分，最佳算法为: {best_model[0]} (得分: {best_model[1]:.4f})\n")
        
        print(f"\n报告已保存到: {report_file}")
        
        return ahp_weights

def main():
    results_dir = 'e:/机器学习大作业/算法对比实验/results'
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        print("请先运行各个算法的训练脚本")
        return
    
    comparator = AlgorithmComparator(results_dir)
    
    if len(comparator.results) == 0:
        print("没有找到任何算法结果文件")
        print("请先运行以下训练脚本:")
        print("  - python train_resnet50.py")
        print("  - python train_vgg16.py")
        print("  - python train_efficientnet.py")
        print("  - python train_densenet.py")
        return
    
    print(f"\n找到 {len(comparator.results)} 个算法的结果")
    
    comparator.plot_comparison()
    comparator.plot_training_curves()
    comparator.plot_confusion_matrices()
    
    ahp_weights = comparator.generate_report()
    
    print("\n" + "="*60)
    print("算法对比分析完成！")
    print("="*60)

if __name__ == '__main__':
    main()
