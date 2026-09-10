"""
Pacote de Modelos do Projeto liuzywen-RGBTCC (BMVC 2022)
=========================================================
Implementa o modelo 'RGB-T Multi-Modal Crowd Counting Based on Transformer'
(Liu et al., BMVC 2022).
"""

from .pvt_v2 import PVTv2Encoder
from .liuzywen_rgbtcc_net import (
    LiuzywenRGBTCCNet,
    CountGuidedMSTTrans,
    ModalGuidedMSDTrans,
    DensityRegressionHead
)
from .models import build_model

__all__ = [
    "PVTv2Encoder",
    "LiuzywenRGBTCCNet",
    "CountGuidedMSTTrans",
    "ModalGuidedMSDTrans",
    "DensityRegressionHead",
    "build_model"
]
