import torch
import logging
from ultralytics import YOLO
from src.core.base import BaseDetector

logger = logging.getLogger(__name__)

class YOLODetector(BaseDetector):
    """Implementação do detector usando YOLOv11."""
    
    def __init__(self, model_path="yolo11n.pt", device=None, conf=0.65, imgsz=640, 
                 classes=[0], use_half=True):
        # Valida se a GPU está disponível antes de assumir CUDA
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes
        
        # Só permite half se estiver em GPU
        self.use_half = use_half and ("cuda" in self.device)
        
        if "cuda" in self.device:
            logger.info(f"--- [DETECTOR] GPU Ativada para YOLO: {torch.cuda.get_device_name(0)}")
            if self.use_half: logger.info("--- [DETECTOR] Usando Half Precision (FP16)")
        else:
            logger.warning("--- [DETECTOR] GPU NÃO ENCONTRADA para YOLO! Usando CPU.")
        
    def detect(self, frame):
        # Inferência configurável
        results = self.model(frame, verbose=False, device=self.device, 
                             classes=self.classes, conf=self.conf, imgsz=self.imgsz,
                             half=self.use_half)
        
        detections = []
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, confidence in zip(boxes, confs):
                x1, y1, x2, y2 = box
                detections.append([int(x1), int(y1), int(x2), int(y2), float(confidence)])
                
        return detections
