"""
PyTorch implementation of core layers for the Video VAE and DiT.
Ported from JAX/Flax: /projects/video-VAE/diffusion/layers.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Converts video into patch tokens with normalization and linear projection."""

    def __init__(self, height: int, width: int, channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        dim = patch_size * patch_size * channels
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) video tensor
        Returns:
            (B, T, hw, patch_dim) patch tokens
        """
        x = rearrange(x, "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
                       p1=self.patch_size, p2=self.patch_size)
        x = self.norm(x)
        x = self.linear(x)
        return x


class PatchUnEmbedding(nn.Module):
    """Converts patch tokens back to spatial video with upsampling."""

    def __init__(self, height: int, width: int, channels: int,
                 patch_size: int, upsample_rate: int):
        super().__init__()
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.upsample_rate = upsample_rate
        dim = patch_size * patch_size * channels
        self.linear = nn.Linear(dim, dim)
        self.upsample = nn.Linear(dim, dim * upsample_rate)
        self.downsample = nn.Linear(channels * upsample_rate, channels)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, hw, patch_dim) patch tokens
        Returns:
            convolutional_upsampled_features: (B, T, H, W, C*upsample_rate)
            x: (B, T, H, W, C) downsampled features
        """
        x = self.linear(x)
        x = self.upsample(x)
        convolutional_upsampled_features = rearrange(
            x, "b t (h w) (p1 p2 c u) -> b t (h p1) (w p2) (c u)",
            p1=self.patch_size, p2=self.patch_size,
            h=self.height // self.patch_size, w=self.width // self.patch_size,
            u=self.upsample_rate
        )
        x = self.downsample(convolutional_upsampled_features)
        return convolutional_upsampled_features, x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with NTK scaling."""

    def __init__(self, head_dim: int, max_len: int = 8192,
                 alpha: float = 1.0, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len

        ntk_base = base * (alpha ** (head_dim / (head_dim - 2)))
        inv_freq = 1.0 / (ntk_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Shape: [1, max_len, 1, head_dim]
        self.register_buffer("cos_cached", torch.cos(emb)[None, :, None, :])
        self.register_buffer("sin_cached", torch.sin(emb)[None, :, None, :])

    def rotate_queries_and_keys(self, q: torch.Tensor, k: torch.Tensor):
        """
        Args:
            q, k: [Batch, Seq_Len, Num_Heads, Head_Dim]
        Returns:
            Rotated q, k with same shape
        """
        seq_len = q.shape[1]
        cos_slice = self.cos_cached[:, :seq_len, :, :].to(q.dtype)
        sin_slice = self.sin_cached[:, :seq_len, :, :].to(q.dtype)
        q_rot = (q * cos_slice) + (rotate_half(q) * sin_slice)
        k_rot = (k * cos_slice) + (rotate_half(k) * sin_slice)
        return q_rot, k_rot


class Attention(nn.Module):
    """Multi-head attention with RoPE, Q/K normalization, and pre-norm."""

    def __init__(self, in_features: int, num_heads: int, qkv_features: int,
                 max_len: int, use_qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = qkv_features // num_heads
        self.head_dim = head_dim
        self.qkv_features = qkv_features

        self.input_norm = nn.LayerNorm(in_features, eps=1e-6)
        self.qkv_projection = nn.Linear(in_features, qkv_features * 3)
        # Output projection with small initialization (matches JAX variance_scaling 1e-2)
        self.out_projection = nn.Linear(qkv_features, in_features)
        self.ROPE = RotaryEmbedding(head_dim=head_dim, max_len=max_len)
        self.q_norm = nn.LayerNorm(head_dim, bias=False, eps=1e-6)
        self.k_norm = nn.LayerNorm(head_dim, bias=False, eps=1e-6)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, seq, dim)
            mask: (B, 1, 1, seq) boolean mask (True = attend, False = ignore)
        Returns:
            (B, seq, dim)
        """
        x = self.input_norm(x)
        qkv = self.qkv_projection(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b seq (head dim) -> b seq head dim", head=self.num_heads)
        k = rearrange(k, "b seq (head dim) -> b seq head dim", head=self.num_heads)
        v = rearrange(v, "b seq (head dim) -> b seq head dim", head=self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.ROPE.rotate_queries_and_keys(q, k)

        # Transpose for attention: (B, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build attention mask for SDPA
        attn_mask = None
        if mask is not None:
            # mask: (B, 1, 1, seq) -> broadcast to (B, heads, seq_q, seq_k)
            attn_mask = mask  # PyTorch SDPA handles broadcasting

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_output = attn_output.transpose(1, 2)  # (B, seq, heads, dim)
        attn_output = rearrange(attn_output, "b seq head dim -> b seq (head dim)")
        attn_output = self.out_projection(attn_output)
        return attn_output


class MLP(nn.Module):
    """Feed-forward network with pre-norm and SiLU activation."""

    def __init__(self, in_features: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features, eps=1e-6)
        self.linear1 = nn.Linear(in_features, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        return x


class FactoredAttention(nn.Module):
    """Factored spatiotemporal attention block: temporal attn+MLP then spatial attn+MLP."""

    def __init__(self, mlp_dim: int, in_features: int, num_heads: int,
                 qkv_features: int, max_temporal_len: int, max_spatial_len: int):
        super().__init__()
        self.SpatialAttention = Attention(in_features, num_heads, qkv_features,
                                          max_spatial_len, True)
        self.SpatialMLP = MLP(in_features, mlp_dim)
        self.TemporalAttention = Attention(in_features, num_heads, qkv_features,
                                           max_temporal_len, False)
        self.TemporalMLP = MLP(in_features, mlp_dim)

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, HW, C) input tensor
            temporal_mask: (B, 1, 1, T) boolean mask
        Returns:
            (B, T, HW, C) output tensor
        """
        b, t, hw, c = x.shape

        # Temporal attention
        temporal_x = rearrange(x, "b t hw c -> (b hw) t c")
        temporal_mask_expanded = repeat(temporal_mask, "b 1 1 t -> (b hw) 1 1 t", hw=hw)
        temporal_attn_output = self.TemporalAttention(temporal_x, mask=temporal_mask_expanded)
        temporal_x = temporal_x + temporal_attn_output
        temporal_x = temporal_x + self.TemporalMLP(temporal_x)
        original_shape_x = rearrange(temporal_x, "(b hw) t c -> b t hw c", b=b, hw=hw)

        # Spatial attention
        spatial_x = rearrange(original_shape_x, "b t hw c -> (b t) hw c")
        spatial_attn_output = self.SpatialAttention(spatial_x)  # No mask for spatial
        spatial_x = spatial_x + spatial_attn_output
        spatial_x = spatial_x + self.SpatialMLP(spatial_x)

        original_shape_x = rearrange(spatial_x, "(b t) hw c -> b t hw c", b=b, t=t)
        return original_shape_x


class GumbelSigmoidSTE(nn.Module):
    """Gumbel-Sigmoid with Straight-Through Estimator for differentiable discrete selection."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, train: bool = True) -> torch.Tensor:
        if train:
            eps = 1e-20
            u = torch.rand_like(logits).clamp(eps, 1.0 - eps)
            logistic_noise = torch.log(u / (1 - u))
            y = torch.sigmoid((logits + logistic_noise) / self.temperature)
            # Straight-through: round in forward, pass gradient through
            return y.round() + (y - y.detach())
        else:
            return torch.round(torch.sigmoid(logits / self.temperature))
