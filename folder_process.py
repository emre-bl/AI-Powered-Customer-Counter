import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict

class EnhancedCustomerCounter:
    def __init__(self, folder_path, show_gui=True, frame_skip=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8n.pt').to(self.device)
        self.folder_path = folder_path
        self.results = {}
        self.show_gui = show_gui
        self.frame_skip = frame_skip
        
        self.track_history = defaultdict(list)
        self.initial_positions = {}
        self.counted_ids = set()
        self.total_count = 0
        
        self.colors = {'counted': (0, 255, 0), 'uncounted': (0, 0, 255)}
        self.frame_size = 640
        self.detection_roi = None
        self.entry_line = None

    def select_roi_interactively(self, frame):
        """ROI seçimi her durumda etkin"""
        roi = cv2.selectROI("ALGILAMA ALANI SEC (Her durumda aktif)", frame, showCrosshair=True)
        cv2.destroyAllWindows()
        
        self.detection_roi = {
            'x': int(roi[0]),
            'y': int(roi[1]),
            'w': int(roi[2]),
            'h': int(roi[3])
        }
        self.entry_line = self.detection_roi['y'] + self.detection_roi['h'] // 2

    def is_within_roi(self, x, y):
        return (self.detection_roi['x'] <= x <= self.detection_roi['x'] + self.detection_roi['w'] and
                self.detection_roi['y'] <= y <= self.detection_roi['y'] + self.detection_roi['h'])

    def has_valid_crossing(self, track_id):
        if track_id not in self.initial_positions or len(self.track_history[track_id]) < 2:
            return False
            
        prev_y = self.track_history[track_id][-2][1]
        current_y = self.track_history[track_id][-1][1]
        return prev_y <= self.entry_line and current_y > self.entry_line

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        writer = None
        if self.show_gui:
            writer = cv2.VideoWriter(
                f"output_{os.path.basename(video_path)}",
                cv2.VideoWriter_fourcc(*'mp4v'),
                int(cap.get(cv2.CAP_PROP_FPS) // self.frame_skip),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )  # Parantez düzeltildi
        
        video_count = 0
        self.track_history.clear()
        self.initial_positions.clear()
        self.counted_ids.clear()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // self.frame_skip)
        with tqdm(total=total_frames,  # Parantez düzeltildi
                desc=os.path.basename(video_path)[:25].ljust(25),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Kalan: {remaining}]") as pbar:
            
            frame_idx = 0
            while cap.isOpened():
                for _ in range(self.frame_skip-1):
                    if not cap.grab():
                        break
                ret, frame = cap.retrieve()
                
                if not ret:
                    break

                results = self.model.track(frame, imgsz=self.frame_size, conf=0.6, classes=0, 
                                         persist=True, tracker="bytetrack.yaml", verbose=False)

                if results[0].boxes.id is not None:
                    for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(),
                                           results[0].boxes.id.int().cpu().numpy()):
                        x, y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        if self.is_within_roi(x, y):
                            self.track_history[track_id].append((x, y))
                            if track_id not in self.initial_positions:
                                self.initial_positions[track_id] = y
                            if len(self.track_history[track_id]) > 15:
                                self.track_history[track_id].pop(0)
                            if self.has_valid_crossing(track_id):
                                video_count += 1
                                self.counted_ids.add(track_id)
                            if self.show_gui:
                                color = self.colors['counted'] if track_id in self.counted_ids else self.colors['uncounted']
                                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                                cv2.putText(frame, f"ID:{track_id}", (int(box[0]), int(box[1])-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if self.show_gui:
                    cv2.rectangle(frame, (self.detection_roi['x'], self.detection_roi['y']),
                                (self.detection_roi['x']+self.detection_roi['w'], 
                                 self.detection_roi['y']+self.detection_roi['h']), (0,255,0), 2)
                    cv2.putText(frame, f"Toplam: {video_count}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    writer.write(frame)
                    cv2.imshow("Customer Counter", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                pbar.update(1)
                frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        return video_count

    def process_folder(self):
        sample_video = next((f for f in os.listdir(self.folder_path) 
                          if f.lower().endswith(('.avi'))), None)
        if not sample_video:
            raise ValueError("Klasörde video bulunamadı!")
        
        cap = cv2.VideoCapture(os.path.join(self.folder_path, sample_video))
        ret, frame = cap.read()
        if ret:
            self.select_roi_interactively(frame)
        cap.release()
        
        video_files = [f for f in os.listdir(self.folder_path) 
                     if f.lower().endswith(('.avi'))]
        total = 0
        
        print("\n\033[1mVideo Processing\033[0m")
        print("▬"*40)
        for video_file in video_files:
            video_path = os.path.join(self.folder_path, video_file)
            count = self.process_video(video_path)
            self.results[video_file] = count
            total += count
            print(f"▸ \033[34m{video_file[:25]:<25}\033[0m: \033[32m{count:>4}\033[0m")
        
        print("\n\033[1mReport\033[0m")
        print("═"*40)
        print(f"\033[1mTotal Count of Customer\033[0m{' ':>30}: \033[32m{total:>4}\033[0m\n")

if __name__ == "__main__":
    """
    Çalıştırırken : python folder_process.py <video_klasör_yolu>
    Eğer GUI'yi görmek istemiyorsanız: show_gui=False 
    Eğer daha hızlı ama daha az doğru sonuçlar almak istiyorsanız: frame_skip değerini artırın
    """


    parser = argparse.ArgumentParser(description='Geliştirilmiş Müşteri Sayım Sistemi')
    parser.add_argument('folder_path', help='Video klasör yolu')
    
    args = parser.parse_args()

    show_gui = False
    frame_skip = 2
    
    counter = EnhancedCustomerCounter(
        args.folder_path, 
        show_gui=show_gui,
        frame_skip=frame_skip
    )
    counter.process_folder()