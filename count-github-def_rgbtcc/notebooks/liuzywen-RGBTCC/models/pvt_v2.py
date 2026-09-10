"""
Pyramid Vision Transformer v2 (PVTv2) Backbone Module
=====================================================
Reference:
  "PVTv2: Improved Baselines with Linear Spatial Reduction Attention"
  Wenhai Wang et al., Computational Visual Media, 2022.
Used in:
  "RGB-T Multi-Modal Crowd Counting Based on Transformer" (BMVC 2022)
  Liu et al., arXiv:2301.03033.

Provides hierarchical 4-stage feature maps [F^1, F^2, F^3, F^4]
with strides [4, 8, 16, 32] for multiscale multimodal crowd counting.
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping convolutions to preserve local continuity."""

    def __init__(self, patch_size: int = 7, stride: int = 4, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        x = self.norm(x)
        _, _, h, w = x.shape
        # Flatten spatial dimensions to token sequence: [B, H*W, C]
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class LinearSRAttention(nn.Module):
    """Linear Spatial Reduction Attention (SRA) for efficient Transformer encoding."""

    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, bias=False)
            self.norm = nn.BatchNorm2d(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None and h > 1 and w > 1:
            # Spatial reduction along token sequence
            x_spatial = x.transpose(1, 2).reshape(b, c, h, w)
            x_reduced = self.sr(x_spatial)
            x_reduced = self.norm(x_reduced)
            x_reduced = x_reduced.flatten(2).transpose(1, 2)
            kv = self.kv(x_reduced).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(b, -1, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        return out


class MixFFN(nn.Module):
    """Feed-Forward Network with 3x3 depth-wise convolution for inductive bias."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            bias=False
        )
        self.bn = nn.BatchNorm2d(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        x = self.fc1(x)
        # Depthwise conv over spatial 2D feature map
        x_2d = x.transpose(1, 2).reshape(b, -1, h, w)
        x_2d = self.act(self.bn(self.dwconv(x_2d)))
        x = x_2d.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x


class PVTv2Block(nn.Module):
    """Standard PVTv2 Transformer Block."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearSRAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x), h, w)
        return x


class PVTv2Stage(nn.Module):
    """Single Stage of PVTv2 consisting of patch embedding and multiple transformer blocks."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        sr_ratio: int,
        patch_size: int = 3,
        stride: int = 2
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(
            patch_size=patch_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.blocks = nn.ModuleList([
            PVTv2Block(embed_dim, num_heads=num_heads, sr_ratio=sr_ratio)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        x_tokens, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x_tokens = blk(x_tokens, h, w)
        x_tokens = self.norm(x_tokens)
        b, n, c = x_tokens.shape
        x_2d = x_tokens.transpose(1, 2).reshape(b, c, h, w)
        return x_2d, x_tokens, h, w


class PVTv2Encoder(nn.Module):
    """Full 4-Stage Hierarchical PVTv2 Encoder for Crowd Counting Feature Extraction.

    Outputs feature representations at 4 scales:
      - F^1: Stride 4, Dim 64
      - F^2: Stride 8, Dim 128
      - F^3: Stride 16, Dim 320
      - F^4: Stride 32, Dim 512 (High-layer semantic tokens)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 320, 512],
        num_heads: List[int] = [1, 2, 5, 8],
        depths: List[int] = [2, 2, 2, 2],
        sr_ratios: List[int] = [8, 4, 2, 1]
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # Stage 1: Stride 4
        self.stage1 = PVTv2Stage(
            in_channels=in_channels,
            embed_dim=embed_dims[0],
            num_blocks=depths[0],
            num_heads=num_heads[0],
            sr_ratio=sr_ratios[0],
            patch_size=7,
            stride=4
        )
        # Stage 2: Stride 8
        self.stage2 = PVTv2Stage(
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            num_blocks=depths[1],
            num_heads=num_heads[1],
            sr_ratio=sr_ratios[1],
            patch_size=3,
            stride=2
        )
        # Stage 3: Stride 16
        self.stage3 = PVTv2Stage(
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            num_blocks=depths[2],
            num_heads=num_heads[2],
            sr_ratio=sr_ratios[2],
            patch_size=3,
            stride=2
        )
        # Stage 4: Stride 32
        self.stage4 = PVTv2Stage(
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            num_blocks=depths[3],
            num_heads=num_heads[3],
            sr_ratio=sr_ratios[3],
            patch_size=3,
            stride=2
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the 4 stages.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            List of 4 feature maps [F^1, F^2, F^3, F^4]
        """
        f1, _, _, _ = self.stage1(x)
        f2, _, _, _ = self.stage2(f1)
        f3, _, _, _ = self.stage3(f2)
        f4, _, _, _ = self.stage4(f3)
        return [f1, f2, f3, f4]
