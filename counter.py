import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse

class AdvancedCustomerCounter:
    def __init__(self, video_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO('yolov8m.pt').to(self.device)
        self.model.fuse()
        
        self.track_history = defaultdict(lambda: [])
        self.initial_positions = {}  # ID'ye göre başlangıç pozisyonu
        self.counted_ids = set()
        self.total_count = 0
        
        self.detection_roi = None  # Algılama alanı
        self.entry_line = None     # Giriş çizgisi (y koordinatı)
        self.selecting_roi = True
        
        self.video_path = video_path
        self.colors = {
            'counted': (0, 255, 0),
            'uncounted': (0, 0, 255)
        }
        self.frame_size = 640

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise IOError(f"Video dosyası açılamadı: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.output_path = f"output_{self.video_path.split('/')[-1]}"
        self.writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

    def select_roi_interactively(self, frame):
        # Algılama alanı seçimi
        cv2.namedWindow('ALGILAMA ALANI SEC', cv2.WINDOW_NORMAL)
        roi = cv2.selectROI('ALGILAMA ALANI SEC', frame, showCrosshair=True)
        cv2.destroyAllWindows()
        
        self.detection_roi = {
            'x': int(roi[0]),
            'y': int(roi[1]),
            'w': int(roi[2]),
            'h': int(roi[3])
        }
        
        # Giriş çizgisi seçimi
        cv2.namedWindow('GIRIS CIZGISI SEC', cv2.WINDOW_NORMAL)
        print("Giriş çizgisini belirlemek için iki noktaya tıklayın")
        self.line_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.line_points) < 2:
                    self.line_points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0,0,255), -1)
                    cv2.imshow('GIRIS CIZGISI SEC', frame)
                
                if len(self.line_points) == 2:
                    self.entry_line = (self.line_points[0][1] + self.line_points[1][1]) // 2
                    cv2.destroyAllWindows()
        
        cv2.setMouseCallback('GIRIS CIZGISI SEC', mouse_callback)
        cv2.imshow('GIRIS CIZGISI SEC', frame)
        cv2.waitKey(0)

    def is_within_detection_area(self, x, y):
        return (self.detection_roi['x'] <= x <= self.detection_roi['x'] + self.detection_roi['w'] and
                self.detection_roi['y'] <= y <= self.detection_roi['y'] + self.detection_roi['h'])

    def has_valid_crossing(self, track_id):
        # İlk pozisyon kontrolü
        if track_id not in self.initial_positions:
            return False
            
        initial_y = self.initial_positions[track_id]
        track = self.track_history[track_id]
        
        # İlk pozisyon çizginin altındaysa sayma
        if initial_y > self.entry_line:
            return False
            
        # Son iki pozisyon kontrolü
        if len(track) < 2:
            return False
            
        prev_y = track[-2][1]
        current_y = track[-1][1]
        
        # Yukarıdan aşağıya geçiş kontrolü
        return prev_y <= self.entry_line and current_y > self.entry_line

    def draw_visuals(self, frame):
        # Detection ROI
        cv2.rectangle(frame,
                    (self.detection_roi['x'], self.detection_roi['y']),
                    (self.detection_roi['x']+self.detection_roi['w'], self.detection_roi['y']+self.detection_roi['h']),
                    (0,255,0), 2)
        
        # Entry Line
        cv2.line(frame,
                (0, self.entry_line),
                (self.width, self.entry_line),
                (255,0,255), 2)
        
        # Counter
        cv2.putText(frame, f"GIREN MUSTERI: {self.total_count}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    def process_frame(self, frame):
        results = self.model.track(
            frame, 
            persist=True, 
            classes=0, 
            tracker="bytetrack.yaml", 
            imgsz=self.frame_size,
            half=True,
            conf=0.6,
            verbose=False
        )

        if results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2

            if self.is_within_detection_area(x_center, y_center):
                # Yeni tespit edilenler için başlangıç pozisyonunu kaydet
                if track_id not in self.initial_positions:
                    self.initial_positions[track_id] = y_center
                
                self.track_history[track_id].append((x_center, y_center))
                
                # Performans için eski verileri temizle
                if len(self.track_history[track_id]) > 15:
                    self.track_history[track_id].pop(0)
                
                # Sayım kontrolü
                if track_id not in self.counted_ids and self.has_valid_crossing(track_id):
                    self.total_count += 1
                    self.counted_ids.add(track_id)

                # Görselleştirme
                color = self.colors['counted'] if track_id in self.counted_ids else self.colors['uncounted']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def run(self):
        self.initialize_video()
        
        # ROI ve çizgi seçimi için ilk frame
        ret, frame = self.cap.read()
        if ret:
            self.select_roi_interactively(frame)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            self.draw_visuals(processed_frame)
            
            self.writer.write(processed_frame)
            cv2.imshow('AKILLI SAYICI', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()
        print(f"İşlem tamamlandı! Toplam giren müşteri: {self.total_count}")
        print(f"Kaydedilen dosya: {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Müşteri Sayma Sistemi')
    parser.add_argument('video_path', type=str, help='Video dosya yolu')
    args = parser.parse_args()
    
    counter = AdvancedCustomerCounter(args.video_path)
    counter.run()