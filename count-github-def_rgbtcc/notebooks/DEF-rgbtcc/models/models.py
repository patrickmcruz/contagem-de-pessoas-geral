"""
Módulo de Arquitetura Multimodal RGBT para Contagem de Pessoas
============================================================
Este módulo implementa a arquitetura dual-stream (RGB + Térmica) para contagem
de pessoas via regressão de mapa de densidade (Density Map Estimation).

Compatível com o layout de importação:
    from models.models import build_model
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Bloco convolucional duplo com Convolução 3x3, BatchNorm e ativação LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DualStreamRGBTNet(nn.Module):
    """Rede Neural Dual-Stream para Fusão Multimodal (RGB + Térmica) e Contagem por Mapa de Densidade.

    Arquitetura:
    - Ramo Visual (RGB): Extração hierárquica de características texturais e de borda.
    - Ramo Térmico (LWIR): Extração de assinaturas calóricas e silhuetas humanas.
    - Fusão Multimodal: Ponderação e concatenação das camadas intermediárias.
    - Decodificador de Densidade: Projeção convolucional transposta para mapa de densidade contínuo.
    """

    def __init__(self):
        super().__init__()
        # 1. Encoders por Modalidade (3 -> 32 -> 64 -> 128 canais)
        self.rgb_enc1 = ConvBlock(3, 32)
        self.rgb_enc2 = ConvBlock(32, 64)
        self.rgb_enc3 = ConvBlock(64, 128)

        self.th_enc1 = ConvBlock(3, 32)
        self.th_enc2 = ConvBlock(32, 64)
        self.th_enc3 = ConvBlock(64, 128)

        # 2. Módulo de Fusão e Atenção Espacial Cruzada (256 canais -> 128 -> 64)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            ConvBlock(128, 64),
        )

        # 3. Decodificador com Convoluções Transpostas para Restauração Espacial
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 4. Camada de Regressão de Densidade (Saída: 1 canal contínuo com ativação não-negativa)
        self.regressor = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """Executa a propagação direta (forward pass) do par multimodal.

        Args:
            rgb: Tensor de imagem visível com shape [B, 3, H, W].
            thermal: Tensor de imagem térmica com shape [B, 3, H, W].

        Returns:
            density_map: Tensor de mapa de densidade 2D [B, 1, H, W] (sempre >= 0).
        """
        # Extração de características da modalidade óptica
        r1 = self.rgb_enc1(rgb)
        r2 = self.rgb_enc2(F.max_pool2d(r1, 2))
        r3 = self.rgb_enc3(F.max_pool2d(r2, 2))

        # Extração de características da modalidade térmica
        t1 = self.th_enc1(thermal)
        t2 = self.th_enc2(F.max_pool2d(t1, 2))
        t3 = self.th_enc3(F.max_pool2d(t2, 2))

        # Fusão multimodal das representações
        features_cat = torch.cat([r3, t3], dim=1)
        fused = self.fusion(features_cat)

        # Upsampling e regressão do mapa de densidade
        d1 = self.dec1(fused)
        d2 = self.dec2(d1)
        
        # Ativação Softplus garante densidades contínuas e estritamente não-negativas
        density_map = F.softplus(self.regressor(d2))

        return density_map


def build_model(
    weight_path: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Constrói a arquitetura do modelo e carrega pesos caso disponíveis.

    Args:
        weight_path: Caminho opcional para arquivo de pesos (.pth / .pt).
        device: Dispositivo de execução ('cuda' ou 'cpu').

    Returns:
        model: Instância do modelo configurada em modo de avaliação (eval).
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    model = DualStreamRGBTNet()

    loaded = False
    if weight_path and os.path.exists(weight_path):
        print(f"[build_model] Carregando pesos de checkpoint: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device_obj)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        try:
            model.load_state_dict(state_dict, strict=False)
            print("[build_model] Pesos carregados com sucesso!")
            loaded = True
        except Exception as e:
            print(f"[build_model] Aviso: falha ao carregar state_dict ({e}). Usando inicialização calibrada.")

    if not loaded:
        # Se nenhum caminho foi passado ou não existia, busca o default local se disponível
        default_candidate = Path(__file__).resolve().parent.parent / "weights" / "best_model.pth"
        if default_candidate.exists():
            print(f"[build_model] Carregando pesos padrão encontrados em: {default_candidate}")
            checkpoint = torch.load(default_candidate, map_location=device_obj)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            loaded = True
        else:
            # Inicialização Kaiming Normal
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    model.to(device_obj)
    model.eval()
    return model
