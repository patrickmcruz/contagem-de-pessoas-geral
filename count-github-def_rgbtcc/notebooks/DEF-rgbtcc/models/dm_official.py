"""
Implementação Oficial do Paper DEF-rgbtcc (arXiv:2509.17079)
============================================================
"A Dual-Modulation Framework for RGB-T Crowd Counting via
Spatially Modulated Attention and Adaptive Fusion" (Feng et al., 2025).

Componentes Oficiais:
1. Shared VGG-19 Backbone (features com downsampling 16x).
2. Spatially Modulated Attention (SMA) Transformer (2 camadas, 8 heads) com decaimento espacial.
3. Adaptive Cross-Modal Fusion (AFM) para ponderação dinâmica de canal RGB e Térmico.
4. Head de Regressão Bayesiana com ativação direta abs: torch.abs(density).
"""

import os
import copy
import logging
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


class AdaptiveCrossModalFusion(nn.Module):
    """Módulo Oficial AFM (Adaptive Cross-Modal Fusion).

    Combina características multimodais através de pooling médio global e MLP convolucional 1x1,
    produzindo o fator dinâmico de fusão w in [0, 1].
    """

    def __init__(self, in_channels: int = 512, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.last_w: Optional[torch.Tensor] = None

    def forward(self, rgb_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        combined_feat = rgb_feat + t_feat
        w = self.mlp(self.avg_pool(combined_feat))
        self.last_w = w.detach()
        weighted_rgb = rgb_feat * w
        weighted_t = t_feat * (1.0 - w)
        fusion_output = weighted_rgb + weighted_t
        return fusion_output


def generate_spatial_distance(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Calcula a matriz de distâncias espaciais euclidianas 2D entre todos os tokens."""
    with torch.no_grad():
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        coords = coords.view(-1, 2).float()
        dist_matrix = torch.cdist(coords, coords, p=2)
        return dist_matrix


def process_spatial_decay(dist_matrix: torch.Tensor, beta_scale: torch.Tensor, beta_bias: torch.Tensor) -> torch.Tensor:
    """Aplica o decaimento exponencial modulado com parâmetros aprendíveis beta_scale e beta_bias."""
    beta_scale_sigmoid = torch.sigmoid(beta_scale)
    beta_bias_softplus = F.softplus(beta_bias)

    dist_matrix = dist_matrix.unsqueeze(1)
    processed_dist = F.leaky_relu(dist_matrix - beta_bias_softplus, negative_slope=0.1)
    decay_matrix = torch.pow(beta_scale_sigmoid, processed_dist)
    return decay_matrix


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, sph=None, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if sph is not None:
            attn = attn * sph
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        p_attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(p_attn, v)
        return output, p_attn


class SpatiallyModulatedAttention(nn.Module):
    """Módulo Oficial SMA (Spatially Modulated Attention)."""

    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, sph=None, mask=None):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, sph=sph, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        q = q.permute(1, 0, 2)
        return q, attn


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = SpatiallyModulatedAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, sph=None, src_mask=None, src_key_padding_mask=None):
        q = k = src
        src2, attn = self.self_attn(q, k, src, sph=sph, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int = 2, nhead: int = 8, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.nhead = nhead
        self.beta_scale = nn.Parameter(torch.full((nhead, 1, 1), 0.9))
        self.beta_bias = nn.Parameter(torch.full((nhead, 1, 1), 5.0))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        x = src
        bs, c, h, w = x.shape
        dist_matrix = generate_spatial_distance(h, w, device=x.device).unsqueeze(0)
        sph = process_spatial_decay(dist_matrix, self.beta_scale, self.beta_bias)
        x = x.flatten(2).permute(2, 0, 1)

        for layer in self.layers:
            x, att = layer(x, sph=sph, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)
        output = x.permute(1, 2, 0).view(bs, c, h, w)
        return output


cfg = {
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}


def make_layers(cfg_list, batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in cfg_list:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19():
    return make_layers(cfg["E"])


def reg():
    """Head de Regressão Convolucional Oficial do Paper."""
    model = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, 1),
    )
    return model


class Net(nn.Module):
    """Arquitetura Oficial DEF-rgbtcc (arXiv:2509.17079).

    Total de Parâmetros: ~26.4 Milhões
    Entrada: list [rgb, thermal] com shape [B, 3, H, W]
    Saída: density map [B, 1, H//8, W//8] com ativação torch.abs()
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.features = vgg19()
        d_model = 512
        nhead = 8
        num_encoder_layers = 2
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.transformer_encoder_rgb = TransformerEncoder(encoder_layer, num_encoder_layers, nhead)
        self.transformer_encoder_t = TransformerEncoder(encoder_layer, num_encoder_layers, nhead)
        self.fusion_layer = AdaptiveCrossModalFusion(in_channels=d_model)
        self.reg_layer = reg()

        if pretrained_backbone:
            try:
                vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT)
                self.features.load_state_dict(vgg.features.state_dict())
                logger.info("[Net Oficial] Pesos pré-treinados do VGG-19 ImageNet carregados com sucesso!")
            except Exception as e:
                logger.warning(f"[Net Oficial] Não foi possível carregar pesos VGG-19: {e}")

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            rgb, t = inputs
        else:
            raise ValueError("inputs deve ser lista ou tupla [rgb, thermal]")

        rgb_feat = self.features(rgb)
        t_feat = self.features(t)

        rgb_feat = self.transformer_encoder_rgb(rgb_feat)
        t_feat = self.transformer_encoder_t(t_feat)

        rgb_feat = F.interpolate(rgb_feat, scale_factor=2, mode="bilinear", align_corners=False)
        t_feat = F.interpolate(t_feat, scale_factor=2, mode="bilinear", align_corners=False)

        fusion = self.fusion_layer(rgb_feat, t_feat)
        density = self.reg_layer(fusion)

        return torch.abs(density)


class OfficialDMWrapper(nn.Module):
    """Wrapper para permitir chamada unificada: model(rgb, thermal) -> dict."""

    def __init__(self, net: Net):
        super().__init__()
        self.net = net

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Dict[str, torch.Tensor]:
        dmap = self.net([rgb, thermal])
        # Resgatar o peso de fusão calculado pelo AFM
        w = getattr(self.net.fusion_layer, "last_w", None)
        if w is None:
            w = torch.tensor(0.50, device=rgb.device)
        return {
            "density_map": dmap,
            "fusion_weight": w,
        }

