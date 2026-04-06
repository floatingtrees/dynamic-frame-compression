"""
PyTorch implementation of 3D UNet for video processing.
Ported from JAX/Flax: /projects/video-VAE/diffusion/unet.py

NOTE: JAX uses channels-last (B, T, H, W, C) convention throughout.
We maintain this convention in PyTorch to match the JAX implementation exactly,
using permutations to/from channels-first for Conv3d ops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D convolution block with GroupNorm and SiLU activation.

    Input/Output: (B, T, H, W, C) - channels last to match JAX.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, temporal_kernel: int = 3):
        super().__init__()
        padding_t = temporal_kernel // 2
        padding_s = kernel_size // 2
        # Conv3d expects (B, C, T, H, W)
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, kernel_size, kernel_size),
            padding=(padding_t, padding_s, padding_s),
        )
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
            eps=1e-6,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C) -> (B, T, H, W, C)"""
        # channels last -> channels first
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        # channels first -> channels last
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        return x


class DownBlock3D(nn.Module):
    """3D downsampling block with two conv layers and spatial-only pooling."""

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 3):
        super().__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, kernel_size=3,
                                 temporal_kernel=temporal_kernel)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size=3,
                                 temporal_kernel=temporal_kernel)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, H, W, C) -> (pooled, skip)
        pooled: (B, T, H/2, W/2, C)
        skip: (B, T, H, W, C)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        # Spatial-only max pooling: (1, 2, 2) window/stride
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H/2, W/2, C)
        return x, skip


class UpBlock3D(nn.Module):
    """3D upsampling block with transposed conv and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 3):
        super().__init__()
        # ConvTranspose3d for spatial upsampling (preserve temporal)
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        # After concatenation with skip: out_channels * 2
        self.conv1 = ConvBlock3D(out_channels * 2, out_channels, kernel_size=3,
                                 temporal_kernel=temporal_kernel)
        self.conv2 = ConvBlock3D(out_channels, out_channels, kernel_size=3,
                                 temporal_kernel=temporal_kernel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, H, W, C)
        skip: (B, T, H*2, W*2, C_skip)
        Returns: (B, T, H*2, W*2, C_out)
        """
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H*2, W*2, C)
        # Concatenate with skip connection along channel dim
        x = torch.cat([x, skip], dim=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):
    """
    3D UNet for video processing with temporal convolutions.

    Input: (B, T, H, W, channels)
    Output: (B, T, H, W, out_features)
    """

    def __init__(self, channels: int, base_features: int = 16,
                 num_levels: int = 3, out_features: int = 3,
                 temporal_kernel: int = 3):
        super().__init__()
        self.num_levels = num_levels

        # Initial mixer
        padding_t = temporal_kernel // 2
        self.patch_mixer = nn.Conv3d(
            channels, channels,
            kernel_size=(temporal_kernel, 7, 7),
            padding=(padding_t, 3, 3),
        )

        # Encoder path
        self.encoders = nn.ModuleList()
        in_ch = channels
        for i in range(num_levels):
            out_ch = base_features * (2 ** i)
            self.encoders.append(DownBlock3D(in_ch, out_ch,
                                             temporal_kernel=temporal_kernel))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_features * (2 ** num_levels)
        self.bottleneck1 = ConvBlock3D(in_ch, bottleneck_ch, kernel_size=3,
                                       temporal_kernel=temporal_kernel)
        self.bottleneck2 = ConvBlock3D(bottleneck_ch, bottleneck_ch, kernel_size=3,
                                       temporal_kernel=temporal_kernel)

        # Decoder path (reverse of encoder)
        self.decoders = nn.ModuleList()
        in_ch = bottleneck_ch
        for i in range(num_levels - 1, -1, -1):
            out_ch = base_features * (2 ** i)
            self.decoders.append(UpBlock3D(in_ch, out_ch,
                                           temporal_kernel=temporal_kernel))
            in_ch = out_ch

        # Final conv with zero initialization
        self.final_conv = nn.Conv3d(
            base_features, out_features,
            kernel_size=(1, 1, 1),
        )
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, C) -> (B, T, H, W, out_features)"""
        # Initial mixing
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.patch_mixer(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)

        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)

        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Final projection
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        x = self.final_conv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, out_features)
        return x
