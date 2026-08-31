import torch
from ultralytics import YOLO
from src.core.base import BaseDetector

class YOLODetector(BaseDetector):
    """Implementação do detector usando YOLOv11."""
    
    def __init__(self, model_path="yolo11n.pt", device=None, conf=0.65):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf = conf
        
    def detect(self, frame):
        """
        Executa a detecção no frame e retorna os resultados formatados.
        Retorna: List de [x1, y1, x2, y2, confidence]
        """
        # classes=[0] filtra apenas por 'pessoa'
        results = self.model(frame, verbose=False, device=self.device, classes=[0], conf=self.conf)
        
        detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, confidence in zip(boxes, confs):
                x1, y1, x2, y2 = box
                detections.append([int(x1), int(y1), int(x2), int(y2), float(confidence)])
                
        return detections
