import torch
import logging
from torchvision import models, transforms
from src.core.base import BaseReID

logger = logging.getLogger(__name__)

class MobileNetReID(BaseReID):
    """Extração de assinaturas visuais usando MobileNetV3 Large."""
    
    def __init__(self, device=None, use_half=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega MobileNetV3 Large (mais precisa)
        base_model = models.mobilenet_v3_large(weights='DEFAULT')
        self.model = base_model
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        
        self.use_half = use_half and ("cuda" in self.device)
        
        # Ativa FP16 se estiver em GPU (RTX 4090 voa aqui)
        if self.use_half:
            self.model = self.model.half()
            
        if "cuda" in self.device:
            logger.info(f"--- [REID] GPU Ativada para Processamento: {torch.cuda.get_device_name(0)}")
            if self.use_half: logger.info("--- [REID] Usando Half Precision (FP16)")
        else:
            logger.warning("--- [REID] GPU NÃO ENCONTRADA. Usando CPU (Lento!)")
            
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract(self, person_crop):
        """Extração individual (legado/fallback)."""
        if person_crop.size == 0:
            return None
        
        results = self.extract_batch([person_crop])
        return results[0] if results else None

    def extract_batch(self, crops):
        """Extração em lote para máxima performance na GPU."""
        if not crops:
            return []
            
        tensors = []
        for crop in crops:
            tensors.append(self.preprocess(crop))
            
        input_tensor = torch.stack(tensors).to(self.device)
        
        if self.use_half:
            input_tensor = input_tensor.half()
            
        with torch.no_grad():
            embeddings = self.model(input_tensor).cpu().numpy()
            
        # Retorna lista de embeddings (1, 1280) para cada crop
        return [emb.reshape(1, -1) for emb in embeddings]
