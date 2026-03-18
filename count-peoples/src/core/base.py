from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """Interface para todos os detectores de objetos."""
    
    @abstractmethod
    def detect(self, frame):
        """
        Detecta objetos no frame.
        Retorna uma lista de bounding boxes [[x1, y1, x2, y2, conf], ...]
        """
        pass

class BaseReID(ABC):
    """Interface para extratores de assinaturas visuais (Re-ID)."""
    
    @abstractmethod
    def extract(self, person_crop):
        """
        Extrai o embedding (assinatura) de um recorte de imagem.
        Retorna um vetor numpy (1, N).
        """
        pass
