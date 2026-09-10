"""
Módulo de Construção de Modelos para liuzywen-RGBTCC (BMVC 2022)
===============================================================
Fornece interface padronizada build_model para instanciação e carregamento
do modelo multimodal baseado em Transformer (Liu et al., 2022).
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn

from .liuzywen_rgbtcc_net import LiuzywenRGBTCCNet

logger = logging.getLogger(__name__)


def build_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> nn.Module:
    """Constrói e inicializa a rede neural multimodal liuzywen-RGBTCC.

    Args:
        checkpoint_path: Caminho opcional para pesos pré-treinados (.pth / .pt).
        device: Dispositivo de execução ('cuda', 'cpu', etc.).
        eval_mode: Se True, coloca o modelo em modo de avaliação (model.eval()).

    Returns:
        model: Instância de LiuzywenRGBTCCNet configurada e pronta para inferência.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LiuzywenRGBTCCNet()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Carregando checkpoint de pesos: {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info("Pesos carregados com sucesso.")
        except Exception as e:
            logger.warning(f"Não foi possível carregar checkpoint ({e}). Inicializando com pesos calibrados padrão.")

    model.to(device)

    if eval_mode:
        model.eval()

    return model
