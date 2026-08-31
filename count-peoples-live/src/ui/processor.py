import cv2
import time
from src.core.base import BaseDetector, BaseReID
from src.tracking.stable_tracker import StableTracker

class Visualizer:
    """Responsável por desenhar na imagem."""
    
    @staticmethod
    def draw_detections(frame, detections, tracker):
        """Desenha boxes, IDs e status."""
        for det in detections:
            x1, y1, x2, y2, conf = det
            # O processamento do tracker deve acontecer ANTES de desenhar
            pass 

    @staticmethod
    def draw_status_bar(frame, fps, total_count):
        """Desenha o painel superior com FPS e contagem."""
        cv2.rectangle(frame, (0, 0), (320, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Unico: {total_count}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class LiveProcessor:
    """Orquestrador principal do processamento de vídeo."""
    
    def __init__(self, detector: BaseDetector, reid: BaseReID, tracker: StableTracker):
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.prev_time = time.time()

    def process_frame(self, frame):
        """Processa um único frame e retorna o frame anotado."""
        # 1. Detecção
        detections = self.detector.detect(frame)
        annotated_frame = frame.copy()
        
        # 2. Tracking e Re-ID
        for det in detections:
            x1, y1, x2, y2, _ = det
            person_crop = frame[max(0, y1):y2, max(0, x1):x2]
            
            if person_crop.size > 0:
                embedding = self.reid.extract(person_crop)
                final_id, is_stable, status = self.tracker.update(embedding)
                
                # 3. Desenho individual
                color = (0, 255, 0) if is_stable else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, status, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 4. Status Bar
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        
        Visualizer.draw_status_bar(annotated_frame, fps, self.tracker.get_total_unique())
        
        return annotated_frame
