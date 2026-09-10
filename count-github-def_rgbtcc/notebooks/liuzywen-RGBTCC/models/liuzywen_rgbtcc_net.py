"""
Liuzywen-RGBTCC Neural Network Architecture (BMVC 2022)
======================================================
Official Paper:
  "RGB-T Multi-Modal Crowd Counting Based on Transformer"
  Zhengyi Liu, Wei Wu, Yacheng Tan, Guanghui Zhang
  BMVC 2022 / arXiv:2301.03033v1.

Core Architectural Innovations:
1. Dual-Stream PVTv2 Encoders:
   - Visual Stream: F_r = {F_r^1, F_r^2, F_r^3, F_r^4}
   - Thermal Stream: F_t = {F_t^1, F_t^2, F_t^3, F_t^4}
2. Count-Guided Multi-Scale Token Transformer (MSTTrans):
   - Learnable count token F_count ∈ R^{1 x C} guiding two-modal interaction.
   - Parallel three-scale token sequences: initial (N^2), middle (N), large (1).
   - Multi-Head Self-Attention (MHSA) + length restoration FC + channel MLP.
3. Modal-Guided Count Enhancement (MSDTrans):
   - Thermal feature G_t + count token G_count as Query (Q).
   - Multi-scale visual features {G_r, F_r^3, F_r^2, F_r^1} as Key & Value (K, V).
   - Multi-scale deformable cross-attention enhancing thermal counting cues.
4. Density Regression Head (RH):
   - Two 3x3 convs + 1x1 conv + non-negative activation yielding spatial density map D(x, y).
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvt_v2 import PVTv2Encoder


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (MHSA) with LayerNorm and residual connection."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return x + out


class CountGuidedMSTTrans(nn.Module):
    """Count-Guided Multi-Scale Token Transformer (MSTTrans) - Section 3.1.

    Fuses optical and thermal high-layer features under the guidance of a global learnable count token.
    Uses three parallel token sequences across scales (initial N^2, middle N, and large 1).
    """

    def __init__(self, channels: int = 512, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # Learnable global count token F_count ∈ R^{1 x C}
        self.f_count = nn.Parameter(torch.zeros(1, 1, channels))
        nn.init.trunc_normal_(self.f_count, std=0.02)

        # 3 parallel branches of 2 MHSA layers each
        self.branch1 = nn.Sequential(
            MultiHeadSelfAttention(channels, num_heads=num_heads),
            MultiHeadSelfAttention(channels, num_heads=num_heads),
        )
        self.branch2 = nn.Sequential(
            MultiHeadSelfAttention(channels, num_heads=num_heads),
            MultiHeadSelfAttention(channels, num_heads=num_heads),
        )
        self.branch3 = nn.Sequential(
            MultiHeadSelfAttention(channels, num_heads=num_heads),
            MultiHeadSelfAttention(channels, num_heads=num_heads),
        )

        # Channel-wise MLPs for residual combination
        self.mlp2 = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # Final multi-scale fusion MLP: Concat(f1', g2', g3') [dim: 3*C -> C]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(
        self, f_r4: torch.Tensor, f_t4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Forward pass of MSTTrans.

        Args:
            f_r4: High-layer visual feature [B, C, H, W]
            f_t4: High-layer thermal feature [B, C, H, W]

        Returns:
            G_r: Fused visual feature [B, C, H, W]
            G_t: Fused thermal feature [B, C, H, W]
            G_count: Global count feature [B, 1, C]
            h, w: Spatial feature dimensions
        """
        b, c, h, w = f_r4.shape
        n2 = h * w  # Total number of spatial tokens N^2
        n = max(1, int(math.sqrt(n2)))  # Middle-scale tokens N

        # Flatten spatial dimensions: [B, N^2, C]
        tokens_r = f_r4.flatten(2).transpose(1, 2)
        tokens_t = f_t4.flatten(2).transpose(1, 2)

        # Expand learnable count token for batch: [B, 1, C]
        count_token = self.f_count.expand(b, -1, -1)

        # 1. Scale 1 (Initial scale): f1 = [F_r^4, F_t^4, F_count] -> Shape [B, 2*N^2 + 1, C]
        f1 = torch.cat([tokens_r, tokens_t, count_token], dim=1)
        f1_prime = self.branch1(f1)

        # 2. Scale 2 (Middle scale): merge N^2 -> N
        # Adaptive pool over spatial feature map to get N tokens
        pool_h = max(1, int(math.sqrt(n)))
        pool_w = max(1, n // pool_h)
        actual_n = pool_h * pool_w

        r_mid = F.adaptive_avg_pool2d(f_r4, (pool_h, pool_w)).flatten(2).transpose(1, 2)
        t_mid = F.adaptive_avg_pool2d(f_t4, (pool_h, pool_w)).flatten(2).transpose(1, 2)
        f2 = torch.cat([r_mid, t_mid, count_token], dim=1)  # Shape [B, 2*actual_n + 1, C]
        f2_prime = self.branch2(f2)

        # Restore middle-scale sequence length back to (2*N^2 + 1)
        # Interpolate spatial portions and preserve count token
        r_mid_out = f2_prime[:, :actual_n, :].transpose(1, 2).reshape(b, c, pool_h, pool_w)
        t_mid_out = f2_prime[:, actual_n:2*actual_n, :].transpose(1, 2).reshape(b, c, pool_h, pool_w)
        c_mid_out = f2_prime[:, 2*actual_n:, :]

        r_res2 = F.interpolate(r_mid_out, size=(h, w), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        t_res2 = F.interpolate(t_mid_out, size=(h, w), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        g2 = torch.cat([r_res2, t_res2, c_mid_out], dim=1)  # Shape [B, 2*N^2 + 1, C]
        g2_prime = self.mlp2(torch.cat([g2, f1], dim=-1))

        # 3. Scale 3 (Large scale): merge N^2 -> 1
        r_large = F.adaptive_avg_pool2d(f_r4, (1, 1)).flatten(2).transpose(1, 2)  # [B, 1, C]
        t_large = F.adaptive_avg_pool2d(f_t4, (1, 1)).flatten(2).transpose(1, 2)  # [B, 1, C]
        f3 = torch.cat([r_large, t_large, count_token], dim=1)  # Shape [B, 3, C]
        f3_prime = self.branch3(f3)

        # Restore large-scale sequence length: broadcast 1 token across N^2
        r_large_out = f3_prime[:, 0:1, :].expand(-1, n2, -1)
        t_large_out = f3_prime[:, 1:2, :].expand(-1, n2, -1)
        c_large_out = f3_prime[:, 2:3, :]
        g3 = torch.cat([r_large_out, t_large_out, c_large_out], dim=1)  # Shape [B, 2*N^2 + 1, C]
        g3_prime = self.mlp3(torch.cat([g3, f1], dim=-1))

        # 4. Multi-scale aggregation: G = [G_r, G_t, G_count] = MLP(Concat(f1', g2', g3'))
        g_all = self.fusion_mlp(torch.cat([f1_prime, g2_prime, g3_prime], dim=-1))

        # Split into optimized color, thermal, and count features
        g_r_tokens = g_all[:, :n2, :]
        g_t_tokens = g_all[:, n2:2*n2, :]
        g_count = g_all[:, 2*n2:, :]

        # Reshape back to 2D spatial feature maps [B, C, H, W]
        g_r = g_r_tokens.transpose(1, 2).reshape(b, c, h, w)
        g_t = g_t_tokens.transpose(1, 2).reshape(b, c, h, w)

        return g_r, g_t, g_count, h, w


class ModalGuidedMSDTrans(nn.Module):
    """Modal-Guided Counting Enhancement (MSDTrans) - Section 3.2.

    Uses enhanced thermal feature G_t and count token G_count as Query (Q).
    Uses multi-scale visual features {G_r, F_r^3, F_r^2, F_r^1} as Key & Value (K, V).
    Cross-attention enhances thermal crowd cues under visual geometric guidance.
    """

    def __init__(self, embed_dims: List[int] = [64, 128, 320, 512], num_heads: int = 8):
        super().__init__()
        c_top = embed_dims[3]  # 512
        self.num_heads = num_heads
        self.head_dim = c_top // num_heads
        self.scale = self.head_dim ** -0.5

        # Feature adapters to project low-level optical features to common channel dimension
        self.proj_f1 = nn.Conv2d(embed_dims[0], c_top, kernel_size=1)
        self.proj_f2 = nn.Conv2d(embed_dims[1], c_top, kernel_size=1)
        self.proj_f3 = nn.Conv2d(embed_dims[2], c_top, kernel_size=1)

        # Cross-attention projections
        self.q_proj = nn.Linear(c_top, c_top)
        self.k_proj = nn.Linear(c_top, c_top)
        self.v_proj = nn.Linear(c_top, c_top)
        self.out_proj = nn.Linear(c_top, c_top)
        self.norm = nn.LayerNorm(c_top)

    def forward(
        self,
        g_t: torch.Tensor,
        g_count: torch.Tensor,
        g_r: torch.Tensor,
        f_r_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of MSDTrans.

        Args:
            g_t: Enhanced thermal feature [B, C, H, W]
            g_count: Global count token [B, 1, C]
            g_r: Enhanced visual feature [B, C, H, W]
            f_r_list: Multi-scale optical features [F_r^1, F_r^2, F_r^3, F_r^4]

        Returns:
            O_t: Modal-guided enhanced thermal feature map [B, C, H, W]
            O_count: Refined global count token [B, 1, C]
        """
        b, c, h, w = g_t.shape
        n = h * w

        # Build Query: [G_t, G_count] -> Shape [B, N + 1, C]
        q_tokens = torch.cat([g_t.flatten(2).transpose(1, 2), g_count], dim=1)

        # Build Multi-scale Key & Value from optical hierarchy:
        # Resize low-level optical features to top spatial resolution (H, W) and project
        f1_proj = F.adaptive_avg_pool2d(self.proj_f1(f_r_list[0]), (h, w)).flatten(2).transpose(1, 2)
        f2_proj = F.adaptive_avg_pool2d(self.proj_f2(f_r_list[1]), (h, w)).flatten(2).transpose(1, 2)
        f3_proj = F.adaptive_avg_pool2d(self.proj_f3(f_r_list[2]), (h, w)).flatten(2).transpose(1, 2)
        gr_proj = g_r.flatten(2).transpose(1, 2)

        # Multi-scale optical token sequence: Shape [B, 4*N, C]
        kv_tokens = torch.cat([gr_proj, f3_proj, f2_proj, f1_proj], dim=1)

        # Multi-Head Cross Attention
        q = self.q_proj(q_tokens).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_tokens).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_tokens).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, -1, c)
        out = self.norm(q_tokens + self.out_proj(out))

        # Unpack O_t and O_count
        o_t_tokens = out[:, :n, :]
        o_count = out[:, n:, :]

        o_t = o_t_tokens.transpose(1, 2).reshape(b, c, h, w)
        return o_t, o_count


class DensityRegressionHead(nn.Module):
    """Density Map Regression Head (RH) - Section 3.3.

    Consists of two 3x3 convolutions with upsampling and one 1x1 convolution
    followed by non-negative Softplus activation to generate continuous density map D(x, y).
    """

    def __init__(self, in_channels: int = 512, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Softplus(beta=1.0)
        )

        # Global count regressor from O_count token
        self.token_count_head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, o_t: torch.Tensor, o_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        density_map = self.head(o_t)
        token_count = self.token_count_head(o_count.squeeze(1))
        return density_map, token_count


class LiuzywenRGBTCCNet(nn.Module):
    """Complete Neural Network Architecture for liuzywen-RGBTCC (BMVC 2022).

    Integrates:
      1. Dual-Stream PVTv2 Hierarchical Encoders
      2. Count-Guided Multi-Scale Token Transformer (MSTTrans)
      3. Modal-Guided Multi-Scale Deformable Enhancement (MSDTrans)
      4. Density Map Regression Head (RH)
    """

    def __init__(
        self,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2]
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # 1. Dual Encoders
        self.rgb_encoder = PVTv2Encoder(
            in_channels=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths
        )
        self.thermal_encoder = PVTv2Encoder(
            in_channels=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths
        )

        # 2. MSTTrans Fusion
        self.mst_trans = CountGuidedMSTTrans(
            channels=embed_dims[3],
            num_heads=num_heads[3]
        )

        # 3. MSDTrans Count Enhancement
        self.msd_trans = ModalGuidedMSDTrans(
            embed_dims=embed_dims,
            num_heads=num_heads[3]
        )

        # 4. Regression Head
        self.regression_head = DensityRegressionHead(
            in_channels=embed_dims[3],
            hidden_dim=64
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multimodal crowd counting.

        Args:
            rgb: Optical image tensor [B, 3, H, W]
            thermal: Thermal image tensor [B, 3, H, W]

        Returns:
            Dict containing:
              - 'density_map': Spatial density map [B, 1, H_out, W_out]
              - 'count': Integrated crowd count (sum of density map) [B]
              - 'token_count': Direct scalar count predicted by O_count [B]
        """
        # 1. Hierarchical Feature Extraction
        f_r = self.rgb_encoder(rgb)         # [F_r^1, F_r^2, F_r^3, F_r^4]
        f_t = self.thermal_encoder(thermal) # [F_t^1, F_t^2, F_t^3, F_t^4]

        # 2. Count-guided Multi-Scale Token Fusion (MSTTrans)
        g_r, g_t, g_count, _, _ = self.mst_trans(f_r[3], f_t[3])

        # 3. Modal-guided Count Enhancement (MSDTrans)
        o_t, o_count = self.msd_trans(g_t, g_count, g_r, f_r)

        # 4. Density Regression
        density_map, token_count = self.regression_head(o_t, o_count)

        # Total count is spatial integration of density map
        count_from_density = torch.sum(density_map, dim=(1, 2, 3))

        return {
            "density_map": density_map,
            "count": count_from_density,
            "token_count": token_count.squeeze(-1)
        }
