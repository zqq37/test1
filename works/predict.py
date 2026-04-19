"""
灾害检测和时间预测推理脚本
用于对新视频进行预测
"""
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from improved_model import AttentionDisasterPredictor, DisasterTimePredictor
from feature_extractor import FrameFeatureExtractor
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
import matplotlib.font_manager as fm
import platform
import os

# 配置中文字体
system = platform.system()
if system == 'Windows':
    # Windows系统常见中文字体
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simkai.ttf',
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            matplotlib.rcParams['font.family'] = font_prop.get_name()
            matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()]
            print(f"使用字体: {font_path}")
            break
    else:
        # 如果找不到字体文件，尝试使用字体名称
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        print("使用字体名称: SimHei")
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class DisasterPredictor:
    """灾害预测器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.feature_extractor = FrameFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        # 类别映射
        self.idx_to_class = {
            0: 'normal',
            1: 'landslide',
            2: 'debrisFlow',
            3: 'forestfire',
            4: 'mountainFloods',
            5: 'riverOverflow',
            6: 'waterlogging'
        }
        
        self.class_names_chinese = {
            'normal': '正常',
            'landslide': '滑坡',
            'debrisFlow': '泥石流',
            'forestfire': '森林火灾',
            'mountainFloods': '山洪',
            'riverOverflow': '流域性洪水',
            'waterlogging': '城市内涝'
        }
    
    def _load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 根据模型类型创建模型
        model_type = checkpoint.get('model_type', 'attention')
        num_classes = len(checkpoint.get('class_names', ['normal', 'landslide', 'debrisFlow', 
                                                              'forestfire', 'mountainFloods', 
                                                              'riverOverflow', 'waterlogging']))
        
        if model_type == 'attention':
            model = AttentionDisasterPredictor(
                frame_feature_dim=2048,
                num_classes=num_classes
            ).to(self.device)
        else:
            model = DisasterTimePredictor(
                frame_feature_dim=2048,
                num_classes=num_classes
            ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        print(f"模型加载成功: {model_type}")
        print(f"模型准确率: {checkpoint.get('best_acc', 'N/A'):.2f}%")
        
        return model
    
    def extract_frames(self, video_path: str, max_frames: int = 32, 
                     frame_sample_rate: int = 1) -> np.ndarray:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # 默认帧率
        
        frame_interval = max(1, int(fps / frame_sample_rate))
        
        count = 0
        while len(frames) < max_frames and cap.isOpened():
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
        while len(frames) < max_frames:
            frames.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        return np.array(frames)
    
    def predict(self, video_path: str, max_frames: int = 32, 
               frame_sample_rate: int = 1, visualize: bool = False):
        """对视频进行预测"""
        print(f"\n处理视频: {Path(video_path).name}")
        
        # 提取帧
        frames = self.extract_frames(video_path, max_frames, frame_sample_rate)
        print(f"提取了 {len(frames)} 帧")
        
        # 转换为tensor
        frames_tensor = torch.FloatTensor(frames).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            batch_size, seq_len = frames_tensor.shape[0], frames_tensor.shape[1]
            # frames的形状是[batch_size, seq_len, 224, 224, 3]
            # 需要转换为[batch_size*seq_len, 3, 224, 224]
            frames_flat = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
            frames_flat = frames_flat.view(-1, 3, 224, 224)
            features = self.feature_extractor(frames_flat)
            features = features.view(batch_size, seq_len, -1)
            video_features = features.mean(dim=1)
            
            # 预测
            disaster_logits, start_time_logits, end_time_logits, early_warning_logits = self.model(video_features)
            
            # 获取预测结果
            disaster_probs = torch.softmax(disaster_logits, dim=1)
            disaster_pred = torch.argmax(disaster_probs, dim=1).item()
            disaster_confidence = disaster_probs[0, disaster_pred].item()
            
            # 时间预测（转换为帧索引）
            start_frame = int(torch.sigmoid(start_time_logits).item() * max_frames)
            end_frame = int(torch.sigmoid(end_time_logits).item() * max_frames)
            
            # 确保时间范围合理
            start_frame = max(0, min(start_frame, max_frames - 1))
            end_frame = max(0, min(end_frame, max_frames - 1))
            
            # 如果起始时间大于结束时间，交换
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame
        
        # 转换为时间（假设视频时长为60秒）
        video_duration = 60
        start_time_sec = (start_frame / max_frames) * video_duration
        end_time_sec = (end_frame / max_frames) * video_duration
        
        # 构建结果
        disaster_type_en = self.idx_to_class[disaster_pred]
        disaster_type_cn = self.class_names_chinese[disaster_type_en]
        
        result = {
            'video_name': Path(video_path).name,
            'disaster_type_en': disaster_type_en,
            'disaster_type_cn': disaster_type_cn,
            'confidence': disaster_confidence,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time_sec': start_time_sec,
            'end_time_sec': end_time_sec,
            'all_probabilities': {
                self.class_names_chinese[self.idx_to_class[i]]: disaster_probs[0, i].item()
                for i in range(len(self.idx_to_class))
            }
        }
        
        # 可视化
        if visualize:
            self._visualize_prediction(frames, result)
        
        return result
    
    def _visualize_prediction(self, frames: np.ndarray, result: dict):
        """可视化预测结果"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 显示视频帧序列（前4帧）
        num_frames_to_show = min(4, len(frames))
        for i in range(num_frames_to_show):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(frames[i])
            ax.set_title(f"帧{i+1}", fontsize=10)
            ax.axis('off')
        
        # 显示起始帧和结束帧
        if result['start_frame'] < len(frames):
            ax = fig.add_subplot(gs[1, 0])
            ax.imshow(frames[result['start_frame']])
            ax.set_title(f"起始帧 (第{result['start_frame']}帧)", fontsize=10, color='green', fontweight='bold')
            ax.axis('off')
        
        if result['end_frame'] < len(frames):
            ax = fig.add_subplot(gs[1, 1])
            ax.imshow(frames[result['end_frame']])
            ax.set_title(f"结束帧 (第{result['end_frame']}帧)", fontsize=10, color='red', fontweight='bold')
            ax.axis('off')
        
        # 显示概率分布
        ax = fig.add_subplot(gs[1, 2:])
        probs = result['all_probabilities']
        classes = list(probs.keys())
        values = list(probs.values())
        
        # 按概率排序
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        colors = ['green' if i == 0 else 'gray' for i in range(len(sorted_values))]
        bars = ax.barh(sorted_classes, sorted_values, color=colors)
        ax.set_title('灾害类型概率分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('概率', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 在柱状图上显示概率值
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2%}', ha='left', va='center', fontsize=9)
        
        # 显示预测信息
        ax = fig.add_subplot(gs[2, :])
        info_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                          灾害检测结果                                          ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  视频名称: {result['video_name']:<65} ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  灾害类型: {result['disaster_type_cn']:<15} 英文名: {result['disaster_type_en']:<15} ║
        ║  置信度: {result['confidence']:.2%}{"":<40} ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  时间预测:{"":<15} ║
        ║    起始时间: {result['start_time_sec']:.1f}秒 (第{result['start_frame']}帧){"":<20} ║
        ║    结束时间: {result['end_time_sec']:.1f}秒 (第{result['end_frame']}帧){"":<20} ║
        ║    持续时间: {result['end_time_sec'] - result['start_time_sec']:.1f}秒{"":<20} ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  各类别概率:{"":<15} ║
        """
        
        # 添加各类别概率
        for class_name, prob in sorted(zip(sorted_classes, sorted_values)):
            marker = "★ " if class_name == result['disaster_type_cn'] else "  "
            info_text += f"║    {marker}{class_name:<12} {prob:>6.2%}{"":<40} ║\n"
        
        info_text += "╚══════════════════════════════════════════════════════════════════════════════╝"
        
        ax.text(0.05, 0.95, info_text, fontsize=9, 
                verticalalignment='top', family='sans-serif',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.axis('off')
        ax.set_title('预测详情', fontsize=12, fontweight='bold')
        
        # 保存可视化
        save_dir = Path('./predictions')
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{Path(result['video_name']).stem}_prediction.png"
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存到: {save_path}")
        plt.close()
    
    def predict_batch(self, video_paths: list, max_frames: int = 32,
                    frame_sample_rate: int = 1, visualize: bool = False):
        """批量预测多个视频"""
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict(video_path, max_frames, frame_sample_rate, visualize)
                results.append(result)
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                results.append(None)
        
        return results
    
    def print_result(self, result: dict):
        """打印预测结果"""
        print("\n" + "="*60)
        print("灾害检测结果")
        print("="*60)
        print(f"视频名称: {result['video_name']}")
        print(f"\n灾害类型: {result['disaster_type_cn']}")
        print(f"英文名称: {result['disaster_type_en']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"\n时间预测:")
        print(f"  起始时间: {result['start_time_sec']:.1f}秒 (第{result['start_frame']}帧)")
        print(f"  结束时间: {result['end_time_sec']:.1f}秒 (第{result['end_frame']}帧)")
        print(f"  持续时间: {result['end_time_sec'] - result['start_time_sec']:.1f}秒")
        print(f"\n各类别概率:")
        for cls, prob in sorted(result['all_probabilities'].items(), 
                              key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {prob:.2%}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='灾害检测和时间预测')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--video_path', type=str, required=True,
                       help='视频文件或目录路径')
    parser.add_argument('--max_frames', type=int, default=32,
                       help='最大帧数')
    parser.add_argument('--frame_sample_rate', type=int, default=1,
                       help='帧采样率')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--batch', action='store_true',
                       help='是否批量处理目录中的所有视频')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = DisasterPredictor(args.model_path)
    
    # 处理视频
    video_path = Path(args.video_path)
    
    if args.batch and video_path.is_dir():
        # 批量处理目录中的所有视频
        video_files = list(video_path.glob('*.mp4'))
        print(f"找到 {len(video_files)} 个视频文件")
        
        results = predictor.predict_batch(
            [str(vf) for vf in video_files],
            max_frames=args.max_frames,
            frame_sample_rate=args.frame_sample_rate,
            visualize=args.visualize
        )
        
        # 打印所有结果
        for i, result in enumerate(results, 1):
            if result:
                print(f"\n【第 {i} 个视频】")
                predictor.print_result(result)
        
        # 保存结果到JSON
        import json
        save_dir = Path('./predictions')
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = save_dir / f'batch_results_{timestamp}.json'
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([r for r in results if r is not None], f, 
                     indent=2, ensure_ascii=False)
        
        print(f"\n批量结果已保存到: {save_path}")
        
    elif video_path.is_file():
        # 处理单个视频
        result = predictor.predict(
            str(video_path),
            max_frames=args.max_frames,
            frame_sample_rate=args.frame_sample_rate,
            visualize=args.visualize
        )
        
        predictor.print_result(result)
    else:
        print(f"错误: {args.video_path} 不是有效的文件或目录")


if __name__ == '__main__':
    main()
