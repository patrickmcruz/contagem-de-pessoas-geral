import cv2
import time
import numpy as np
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
        """Desenha o painel superior com FPS e contagem com escala dinâmica."""
        h, w = frame.shape[:2]
        
        # Escalonamento dinâmico baseado na largura do frame (referência 1280px)
        scale = w / 1280.0
        font_scale = 1.0 * scale
        thickness = max(2, int(3 * scale))
        
        # Ajusta retângulo de fundo proporcionalmente
        rect_w = int(480 * scale)
        rect_h = int(150 * scale)
        cv2.rectangle(frame, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
        
        # Desenha textos com coordenadas proporcionais
        y1 = int(60 * scale)
        y2 = int(120 * scale)
        x_offset = int(25 * scale)
        
        cv2.putText(frame, f"FPS: {int(fps)}", (x_offset, y1), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        cv2.putText(frame, f"Total Unico: {total_count}", (x_offset, y2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

class LiveProcessor:
    """Orquestrador principal do processamento de vídeo."""
    
    def __init__(self, detector: BaseDetector, reid: BaseReID, tracker: StableTracker, 
                 blur_threshold=100.0, enable_counting=True, show_status_bar=True):
        self.detector = detector
        self.reid = reid
        self.tracker = tracker
        self.prev_time = time.time()
        self.blur_threshold = blur_threshold
        self.enable_counting = enable_counting
        self.show_status_bar = show_status_bar
        self.sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    def is_blurry(self, image):
        """Verifica se a imagem está borrada usando a Variância de Laplacian."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def sharpen(self, image):
        """Aplica filtro de nitidez."""
        return cv2.filter2D(image, -1, self.sharpen_kernel)

    def process_frame(self, frame):
        """Processa um único frame e retorna o frame anotado."""
        # 1. Detecção e parâmetros de escala
        detections = self.detector.detect(frame)
        annotated_frame = frame.copy()
        
        h, w = frame.shape[:2]
        scale = w / 1280.0
        font_scale = 0.8 * scale
        box_thickness = max(2, int(3 * scale))
        text_thickness = max(2, int(2 * scale))
        
        # 2. Coleta de Crops para processamento em Lote
        valid_detections = []
        crops_to_process = []
        
        for det in detections:
            x1, y1, x2, y2, _ = det
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                if self.is_blurry(crop):
                    crop = self.sharpen(crop)
                
                crops_to_process.append(crop)
                valid_detections.append(det)
        
        # 3. Extração em Lote e Tracking
        embeddings = self.reid.extract_batch(crops_to_process)
        tracking_results = self.tracker.update_batch(embeddings)
        
        # 4. Desenho dos resultados
        for det, res in zip(valid_detections, tracking_results):
            x1, y1, x2, y2, _ = det
            final_id, is_stable, status = res
            
            # Se a contagem estiver desativada, não mostramos os IDs unicos
            label = status if self.enable_counting else "Pessoa"
            
            color = (0, 255, 0) if is_stable or not self.enable_counting else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
            
            text_y = y1 - int(15 * scale)
            if text_y < int(30 * scale): text_y = y1 + int(30 * scale)
            
            cv2.putText(annotated_frame, label, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
        
        # 5. Status Bar
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time
        
        if self.show_status_bar:
            total_unique = self.tracker.get_total_unique() if self.enable_counting else "N/A"
            Visualizer.draw_status_bar(annotated_frame, fps, total_unique)
        
        return annotated_frame
