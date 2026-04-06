"""
PyTorch implementation of the Video Diffusion Transformer (DiT).
Ported from JAX/Flax: /projects/video-VAE/diffusion/diffusion_model.py

The DiT takes compressed VAE latents and denoises them using flow matching.
It also predicts frame spacing (adjacent differences) that determine where
each generated latent frame maps in the final output video.
"""

import torch
import torch.nn as nn
from einops import rearrange

from layers import FactoredAttention


class VideoDiT(nn.Module):
    """Video Diffusion Transformer for denoising compressed latent representations."""

    def __init__(self, hw: int = 256, residual_dim: int = 1024,
                 compressed_channel_dim: int = 96, depth: int = 24,
                 mlp_dim: int = 2048, num_heads: int = 8,
                 qkv_features: int = 1024, max_temporal_len: int = 64):
        super().__init__()

        # Timestep projection (initialized to zeros)
        self.timestep_proj = nn.Linear(1, residual_dim)
        nn.init.zeros_(self.timestep_proj.weight)
        nn.init.zeros_(self.timestep_proj.bias)

        self.up_proj = nn.Linear(compressed_channel_dim, residual_dim)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(FactoredAttention(
                mlp_dim=mlp_dim,
                in_features=residual_dim,
                num_heads=num_heads,
                qkv_features=qkv_features,
                max_temporal_len=max_temporal_len,
                max_spatial_len=hw,
            ))

        self.down_proj = nn.Linear(residual_dim, compressed_channel_dim)
        self.spacing_pred1 = nn.Linear(residual_dim, 1)
        self.spacing_pred2 = nn.Linear(hw, 1)

    def forward(self, compressed: torch.Tensor, compression_mask: torch.Tensor,
                time: torch.Tensor):
        """
        Args:
            compressed: (B, T, hw, compressed_channel_dim)
            compression_mask: (B, T) boolean
            time: (B, 1) timestep in [0, 1]
        Returns:
            latent_prediction: (B, T, hw, compressed_channel_dim)
            spacing_prediction: (B, T) predicted frame gaps (adjacent differences)
        """
        compression_mask = rearrange(compression_mask, "b t -> b 1 1 t")
        timestep_weights = rearrange(self.timestep_proj(time), "b d -> b 1 1 d")
        x = self.up_proj(compressed) + timestep_weights

        for layer in self.layers:
            x = layer(x, compression_mask)

        latent_prediction = self.down_proj(x)
        spacing_reduce1 = rearrange(self.spacing_pred1(x), "b t hw 1 -> b t hw")
        spacing_reduce2 = rearrange(self.spacing_pred2(spacing_reduce1), "b t 1 -> b t")
        return latent_prediction, spacing_reduce2


def sample(dit: VideoDiT, noise: torch.Tensor, compression_mask: torch.Tensor,
           num_steps: int = 100):
    """
    Generate samples using Euler integration of the learned velocity field.

    Args:
        dit: VideoDiT model
        noise: (B, T, hw, d) initial noise
        compression_mask: (B, T) boolean
        num_steps: number of Euler integration steps
    Returns:
        x: (B, T, hw, d) denoised latent
        selection_prediction: (B, T) predicted frame gaps (adjacent differences)
    """
    dt = 1.0 / num_steps
    x = noise
    selection_prediction = None

    for i in range(num_steps):
        t = torch.full((noise.shape[0], 1), i / num_steps,
                       device=noise.device, dtype=noise.dtype)
        velocity, selection_prediction = dit(x, compression_mask, t)
        x = x + velocity.to(x.dtype) * dt

    return x, selection_prediction


def gaps_to_positions(gaps: torch.Tensor, mask: torch.Tensor):
    """
    Convert predicted frame gaps (adjacent differences) to absolute frame positions.

    The DiT predicts gaps between consecutive selected frames:
      gaps[0] = position of the first selected frame (>= 0)
      gaps[i] = distance from frame i-1 to frame i (>= 1 for i > 0)

    cumsum(gaps) gives absolute positions: [1, 2, 1] -> [1, 3, 4]

    Args:
        gaps: (B, T) raw float predictions from DiT
        mask: (B, T) boolean indicating valid latent frames
    Returns:
        positions: (B, T) integer absolute frame positions
        total_frames: (B,) integer total output video length per batch element
    """
    # Round to integers and enforce constraints
    int_gaps = gaps.float().round().long()
    # First gap: position of first frame (>= 0)
    int_gaps[:, 0] = int_gaps[:, 0].clamp(min=0)
    # Subsequent gaps: at least 1 apart
    if int_gaps.shape[1] > 1:
        int_gaps[:, 1:] = int_gaps[:, 1:].clamp(min=1)

    # Zero out invalid positions
    int_gaps = int_gaps * mask.long()

    # Cumsum to get absolute positions
    positions = torch.cumsum(int_gaps, dim=1)

    # Total frames = position of last valid frame + 1
    # Find last valid position per batch element
    valid_positions = positions * mask.long()
    total_frames = valid_positions.max(dim=1).values + 1

    return positions, total_frames
