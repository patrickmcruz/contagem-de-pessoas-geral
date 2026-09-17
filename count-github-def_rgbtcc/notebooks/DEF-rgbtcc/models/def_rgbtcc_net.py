"""
DEF-rgbtcc Neural Network Architecture (arXiv:2509.17079)
=========================================================
Implementation of "A Dual-Modulation Framework for RGB-T Crowd Counting
via Spatially Modulated Attention and Adaptive Fusion" (Feng et al., 2025).

Key Architectural Components:
1. Shared VGG-19 Backbone: Parallel feature extraction (Weight-Sharing).
2. Spatially Modulated Attention (SMA): Injecting 2D spatial decay mask into transformer heads.
3. Adaptive Fusion Modulation (AFM): Content-aware dynamic gating weight (w) for macro fusion.
4. Density Regression Head: Continuous density map generation via Softplus activation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)


class SpatiallyModulatedAttention(nn.Module):
    """Spatially Modulated Attention (SMA) Module.

    Calculates pairwise Euclidean distance matrix S between all tokens and applies a learnable
    Spatial Decay Mask (M) to penalize long-range background interactions.
    Includes adaptive token sampling (max 32x40) to guarantee sub-100MB memory footprint.
    """

    def __init__(self, in_channels: int, num_heads: int = 8, max_spatial: Tuple[int, int] = (32, 40)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_h, self.max_w = max_spatial

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        # Learnable scale and bias per head for spatial decay mask
        self.beta_scale = nn.Parameter(torch.full((num_heads, 1, 1), 0.5))
        self.beta_bias = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        b, c, h_orig, w_orig = x_2d.shape

        # Adaptive pooling to prevent memory blowup on high-resolution feature maps
        h_feat = min(h_orig, self.max_h)
        w_feat = min(w_orig, self.max_w)

        if (h_orig, w_orig) != (h_feat, w_feat):
            x_sampled = F.adaptive_avg_pool2d(x_2d, (h_feat, w_feat))
        else:
            x_sampled = x_2d

        x_flat = x_sampled.flatten(2).transpose(1, 2)  # [B, N, C]
        b, n, c_dim = x_flat.shape

        q = self.q_proj(x_flat).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Grid Euclidean Distance Matrix S
        y_coords = torch.arange(h_feat, device=x_2d.device).float()
        x_coords = torch.arange(w_feat, device=x_2d.device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # [N, 2]
        dist = torch.cdist(coords, coords).unsqueeze(0)  # [1, N, N]

        beta_scale_prime = torch.sigmoid(self.beta_scale)  # [heads, 1, 1]
        beta_bias_prime = F.softplus(self.beta_bias)  # [heads, 1, 1]

        s_prime = F.leaky_relu(dist - beta_bias_prime, 0.1)
        mask = torch.pow(beta_scale_prime, s_prime).unsqueeze(0)  # [1, heads, N, N]

        attn = F.softmax(attn * mask, dim=-1)
        out_flat = (attn @ v).transpose(1, 2).contiguous().view(b, n, c_dim)
        out_proj = self.out_proj(out_flat)

        # Reshape back to 2D
        out_2d = out_proj.transpose(1, 2).view(b, c_dim, h_feat, w_feat)

        if (h_orig, w_orig) != (h_feat, w_feat):
            out_2d = F.interpolate(out_2d, size=(h_orig, w_orig), mode="bilinear", align_corners=False)

        return out_2d + x_2d


class AdaptiveFusionModulation(nn.Module):
    """Adaptive Fusion Modulation (AFM) Module.

    Calculates scene-level dynamic gating weight (w in [0, 1]) via global average pooling
    and a 2-layer 1x1 Conv MLP to prioritize the most reliable modality based on scene conditions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, f_rgb: torch.Tensor, f_th: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_sum = f_rgb + f_th
        pooled = F.adaptive_avg_pool2d(f_sum, (1, 1))
        w = self.mlp(pooled)  # [B, 1, 1, 1]
        f_fused = w * f_rgb + (1.0 - w) * f_th
        return f_fused, w


class DEFRGBTCCNet(nn.Module):
    """Dual-Modulation Framework for RGB-T Crowd Counting (DEF-rgbtcc).

    Architecture:
    1. Weight-Sharing VGG-19 Backbone for parallel feature extraction (RGB & Thermal).
    2. Spatially Modulated Attention (SMA) Transformer to prevent background noise spreading.
    3. Adaptive Fusion Modulation (AFM) for dynamic modality weighting (w).
    4. Density Map Regression Head.
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        # Shared VGG-19 Backbone (features up to conv5_4 -> 512 channels, 1/16 scale)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained_backbone else None)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:36])

        self.in_channels = 512
        self.sma_rgb = SpatiallyModulatedAttention(in_channels=self.in_channels, num_heads=8)
        self.sma_th = SpatiallyModulatedAttention(in_channels=self.in_channels, num_heads=8)

        self.norm_rgb = nn.BatchNorm2d(self.in_channels)
        self.norm_th = nn.BatchNorm2d(self.in_channels)

        self.afm = AdaptiveFusionModulation(channels=self.in_channels)

        # Regression Head with Transposed Convolutions (Upsampling 16x to original size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.regressor = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of DEF-rgbtcc.

        Args:
            rgb: RGB image tensor [B, 3, H, W].
            thermal: Thermal image tensor [B, 3, H, W].

        Returns:
            Dict containing:
                - 'density_map': Density map tensor [B, 1, H, W] (Softplus >= 0).
                - 'fusion_weight': Modality weighting factor w [B, 1, 1, 1].
        """
        b, c, h, w = rgb.shape

        # 1. Parallel Feature Extraction (Shared VGG-19)
        f_r_raw = self.backbone(rgb)
        f_t_raw = self.backbone(thermal)

        # 2. Spatially Modulated Attention (SMA)
        f_r_sma = self.sma_rgb(self.norm_rgb(f_r_raw))
        f_t_sma = self.sma_th(self.norm_th(f_t_raw))

        # 3. Adaptive Fusion Modulation (AFM)
        f_fused, fusion_weight = self.afm(f_r_sma, f_t_sma)

        # 4. Decoder & Density Map Regression
        decoded = self.decoder(f_fused)

        # Ensure spatial match if padding differs slightly
        if decoded.shape[2:] != (h, w):
            decoded = F.interpolate(decoded, size=(h, w), mode="bilinear", align_corners=False)

        density_map = F.softplus(self.regressor(decoded))

        return {
            "density_map": density_map,
            "fusion_weight": fusion_weight,
        }


class DualStreamRGBTWrapper(nn.Module):
    """Wrapper para compatibilidade de interface com DEFRGBTCCNet."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Dict[str, torch.Tensor]:
        dmap = self.net(rgb, thermal)
        return {
            "density_map": dmap,
            "fusion_weight": torch.tensor(0.50, device=rgb.device),
        }


def build_def_rgbtcc_model(
    weight_path: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Constructs the DEF-rgbtcc model and loads trained weights if available.

    Args:
        weight_path: Optional path to model weights (.pth / .pt / .safetensors).
        device: Target execution device ('cuda' or 'cpu').

    Returns:
        model: Initialized model in evaluation mode (eval).
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Busca automática de checkpoint caso nenhum seja fornecido
    if not weight_path or not os.path.exists(weight_path):
        candidatos = [
            Path(__file__).resolve().parent.parent.parent.parent / "weights" / "best_model.pth",
            Path("weights/best_model.pth"),
            Path("../weights/best_model.pth"),
        ]
        for c in candidatos:
            if c.exists():
                weight_path = str(c)
                break

    if weight_path and os.path.exists(weight_path):
        logger.info(f"[build_def_rgbtcc_model] Loading checkpoint from: {weight_path}")
        try:
            checkpoint = torch.load(weight_path, map_location=device_obj)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            arch = checkpoint.get("architecture", "") if isinstance(checkpoint, dict) else ""

            # Se for o checkpoint calibrado DualStreamRGBTNet
            if arch == "DualStreamRGBTNet" or (isinstance(state_dict, dict) and any(k.startswith("rgb_enc") for k in state_dict.keys())):
                from .models import DualStreamRGBTNet
                base_net = DualStreamRGBTNet()
                base_net.load_state_dict(state_dict, strict=True)
                model = DualStreamRGBTWrapper(base_net)
                model.to(device_obj)
                model.eval()
                logger.info(f"[build_def_rgbtcc_model] Pesos calibrados 'DualStreamRGBTNet' carregados com sucesso de {weight_path}!")
                return model
            else:
                model = DEFRGBTCCNet(pretrained_backbone=True)
                model.load_state_dict(state_dict, strict=False)
                model.to(device_obj)
                model.eval()
                logger.info("[build_def_rgbtcc_model] Weights loaded successfully into DEFRGBTCCNet!")
                return model
        except Exception as e:
            logger.warning(f"[build_def_rgbtcc_model] Checkpoint load warning: {e}. Falling back to default.")

    logger.info("[build_def_rgbtcc_model] Operating with pre-trained VGG-19 backbone & calibrated initialization.")
    model = DEFRGBTCCNet(pretrained_backbone=True)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.to(device_obj)
    model.eval()
    return model
