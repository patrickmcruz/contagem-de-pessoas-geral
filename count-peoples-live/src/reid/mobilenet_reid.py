import torch
from torchvision import models, transforms
from src.core.base import BaseReID

class MobileNetReID(BaseReID):
    """Extração de assinaturas visuais usando MobileNetV3."""
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carrega MobileNetV3 Small (leve e rápida)
        base_model = models.mobilenet_v3_small(weights='DEFAULT')
        self.model = base_model
        self.model.classifier = torch.nn.Identity() # Pega apenas as características
        self.model.to(self.device)
        self.model.eval()
        
        # Pipeline de pré-processamento padrão ImageNet
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract(self, person_crop):
        """Transforma o recorte da pessoa em um vetor matemático (embedding)."""
        if person_crop.size == 0:
            return None
            
        input_tensor = self.preprocess(person_crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(input_tensor).cpu().numpy().reshape(1, -1)
        return embedding
