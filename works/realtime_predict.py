"""
实时灾害检测和时间预测脚本
在视频播放时实时显示预测结果和起止时间
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from improved_model import DisasterTimePredictor
from feature_extractor import FrameFeatureExtractor
import platform
import os

# 配置中文字体
import matplotlib
import matplotlib.font_manager as fm
system = platform.system()
if system == 'Windows':
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
            break
    else:
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
elif system == 'Darwin':
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
else:
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class RealTimeDisasterPredictor:
    """实时灾害预测器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        model_type = checkpoint.get('model_type', 'simple')
        class_names = checkpoint.get('class_names', ['normal', 'landslide', 'debrisFlow', 
                                                    'forestfire', 'mountainFloods', 
                                                    'riverOverflow', 'waterlogging'])
        
        print(f"模型类型: {model_type}")
        print(f"模型准确率: {checkpoint.get('best_acc', 0):.2%}")
        
        # 创建模型
        if model_type == 'simple':
            self.model = DisasterTimePredictor(num_classes=len(class_names)).to(self.device)
        else:
            self.model = DisasterTimePredictor(num_classes=len(class_names)).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        # 加载特征提取器
        self.feature_extractor = FrameFeatureExtractor().to(self.device)
        self.feature_extractor.eval()
        
        self.class_names = class_names
        self.class_names_cn = ['正常', '滑坡', '泥石流', '森林火灾', '山洪', '流域性洪水', '城市内涝']
        
        # 帧缓存
        self.frame_buffer = []
        self.max_buffer_size = 32
        
        # 预测历史
        self.prediction_history = []
        self.max_history = 10
        
    def preprocess_frame(self, frame: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        """预处理帧"""
        frame = cv2.resize(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def predict_frame(self, frame: np.ndarray) -> dict:
        """预测单帧"""
        frame_processed = self.preprocess_frame(frame)
        frame_tensor = torch.FloatTensor(frame_processed).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(frame_tensor)
            features = features.view(1, -1)
            
            disaster_logits, start_time_logits, end_time_logits, early_warning_logits = self.model(features)
            
            disaster_probs = torch.softmax(disaster_logits, dim=1)
            disaster_pred = torch.argmax(disaster_probs, dim=1).item()
            disaster_confidence = disaster_probs[0, disaster_pred].item()
            
            # 时间预测
            start_time_prob = torch.sigmoid(start_time_logits).item()
            end_time_prob = torch.sigmoid(end_time_logits).item()
            
            return {
                'disaster_type': self.class_names[disaster_pred],
                'disaster_type_cn': self.class_names_cn[disaster_pred],
                'disaster_type_en': self.class_names[disaster_pred],
                'confidence': disaster_confidence,
                'start_time_prob': start_time_prob,
                'end_time_prob': end_time_prob,
                'all_probabilities': {
                    self.class_names_cn[i]: disaster_probs[0, i].item()
                    for i in range(len(self.class_names))
                }
            }
    
    def draw_prediction(self, frame: np.ndarray, prediction: dict, frame_idx: int, total_frames: int) -> np.ndarray:
        """在帧上绘制预测结果"""
        frame_display = frame.copy()
        
        # 获取预测结果
        disaster_type = prediction['disaster_type_en']
        confidence = prediction['confidence']
        start_time_prob = prediction['start_time_prob']
        end_time_prob = prediction['end_time_prob']
        all_probs = prediction['all_probabilities']
        
        # 计算起止时间
        start_frame = int(start_time_prob * total_frames)
        end_frame = int(end_time_prob * total_frames)
        start_time = start_frame / 30.0  # 假设30fps
        end_time = end_frame / 30.0
        
        # 绘制背景框
        overlay = frame_display.copy()
        cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
        frame_display = cv2.addWeighted(overlay, 0.7, frame_display, 0.3, 0.0)
        
        # 绘制标题（使用英文）
        title = f"Disaster Type: {disaster_type}"
        cv2.putText(frame_display, title, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制置信度
        conf_text = f"Confidence: {confidence:.2%}"
        cv2.putText(frame_display, conf_text, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制时间预测
        time_text = f"Time Range: {start_time:.1f}s - {end_time:.1f}s"
        cv2.putText(frame_display, time_text, (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_text = f"Frame Range: {start_frame} - {end_frame}"
        cv2.putText(frame_display, frame_text, (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制进度条
        progress = frame_idx / total_frames
        cv2.rectangle(frame_display, (20, 160), (420, 180), (100, 100, 100), -1)
        cv2.rectangle(frame_display, (20, 160), (20 + int(400 * progress), 180), (0, 255, 0), -1)
        cv2.putText(frame_display, f"{progress:.1%}", (200, 175), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 绘制当前帧位置
        current_pos_text = f"Current: Frame {frame_idx}/{total_frames} ({frame_idx/30.0:.1f}s)"
        cv2.putText(frame_display, current_pos_text, (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制起止帧位置
        if start_frame <= frame_idx <= end_frame:
            cv2.circle(frame_display, (50, 250), 10, (0, 255, 0), -1)
            cv2.putText(frame_display, "Start", (45, 255), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        
        if end_frame <= frame_idx:
            cv2.circle(frame_display, (100, 250), 10, (0, 0, 255), -1)
            cv2.putText(frame_display, "End", (95, 255), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        
        # 绘制概率分布（使用英文）
        y_pos = 270
        class_names_en = ['Normal', 'Landslide', 'DebrisFlow', 'ForestFire', 
                        'MountainFloods', 'RiverOverflow', 'Waterlogging']
        class_mapping = dict(zip(self.class_names_cn, class_names_en))
        
        for i, (class_name, prob) in enumerate(sorted(all_probs.items(), key=lambda x: x[1], reverse=True)):
            if i < 5:  # 只显示前5个
                class_name_en = class_mapping.get(class_name, class_name)
                prob_text = f"{class_name_en}: {prob:.1%}"
                cv2.putText(frame_display, prob_text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 20
        
        return frame_display
    
    def process_video(self, video_path: str, skip_frames: int = 5, speed_multiplier: float = 1.0, loop: bool = True):
        """处理视频并实时显示预测结果"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息:")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps}")
        print(f"  分辨率: {width}x{height}")
        print(f"  时长: {total_frames/fps:.1f}秒")
        print(f"\n控制按键:")
        print(f"  'q' - 退出")
        print(f"  'p' - 暂停/继续")
        print(f"  '空格' - 暂停")
        print(f"  'Enter' - 继续")
        print(f"  's' - 单步播放")
        print(f"  'r' - 重新开始")
        print(f"  'l' - 切换循环播放")
        print(f"  '+' - 加快播放速度")
        print(f"  '-' - 减慢播放速度")
        print(f"\n播放速度: {speed_multiplier:.1f}x")
        
        frame_idx = 0
        paused = False
        loop_playback = loop
        speed_multiplier = 1.0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    if loop_playback:
                        print("\n重新开始播放")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    else:
                        print("\n视频播放完成")
                        break
                
                # 跳帧
                skip_count = max(1, int(skip_frames * speed_multiplier))
                for _ in range(skip_count - 1):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                
                if not ret:
                    if loop_playback:
                        print("\n重新开始播放")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    else:
                        print("\n视频播放完成")
                        break
                
                # 预测
                prediction = self.predict_frame(frame)
                
                # 绘制预测结果
                frame_display = self.draw_prediction(frame, prediction, frame_idx, total_frames)
                
                # 显示
                cv2.imshow('Real-time Disaster Detection', frame_display)
                
                frame_idx += int(skip_frames * speed_multiplier)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC键或q键退出
                print("\n退出播放")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'} playback")
            elif key == 32:  # 空格键：暂停
                paused = True
                print("\nPaused playback")
            elif key == 13:  # Enter键：继续
                paused = False
                print("\nResumed playback")
            elif key == ord('s'):
                # s键：单步播放
                if paused:
                    ret, frame = cap.read()
                    if ret:
                        prediction = self.predict_frame(frame)
                        frame_display = self.draw_prediction(frame, prediction, frame_idx, total_frames)
                        cv2.imshow('Real-time Disaster Detection', frame_display)
                        frame_idx += 1
                else:
                    paused = True
                    print("\nPaused playback")
            elif key == ord('r'):
                # r键：重新开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                print("\nRestarted playback")
            elif key == ord('l'):
                # l键：切换循环播放
                loop_playback = not loop_playback
                print(f"\nLoop playback: {'Enabled' if loop_playback else 'Disabled'}")
            elif key == ord('+') or key == ord('='):
                # +键：加快播放速度
                speed_multiplier = min(speed_multiplier * 1.5, 5.0)
                print(f"\nPlayback speed: {speed_multiplier:.1f}x")
            elif key == ord('-') or key == ord('_'):
                # -键：减慢播放速度
                speed_multiplier = max(speed_multiplier / 1.5, 0.2)
                print(f"\nPlayback speed: {speed_multiplier:.1f}x")
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='实时灾害检测和时间预测')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--video_path', type=str, required=True,
                       help='视频文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='跳过帧数 (默认: 5)')
    parser.add_argument('--speed_multiplier', type=float, default=1.0,
                       help='播放速度倍数 (默认: 1.0, 越大越快)')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = RealTimeDisasterPredictor(args.model_path, args.device)
    
    # 处理视频
    predictor.process_video(args.video_path, args.skip_frames, args.speed_multiplier)


if __name__ == '__main__':
    main()
