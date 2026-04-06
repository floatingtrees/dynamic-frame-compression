"""
PyTorch implementation of the Video VAE (autoencoder).
Ported from JAX/Flax: /projects/video-VAE/diffusion/autoencoder.py

Provides:
    - Encoder: Patch embed -> FactoredAttention layers -> spatial compression + variance + selection
    - Decoder: Spatial decompress -> FactoredAttention layers -> unpatch + UNet refinement
    - VideoVAE: Full encode-decode pipeline with compress/decompress methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers import (PatchEmbedding, PatchUnEmbedding, FactoredAttention,
                    GumbelSigmoidSTE)
from unet import UNet


class Encoder(nn.Module):
    def __init__(self, height: int, width: int, channels: int, patch_size: int,
                 depth: int, mlp_dim: int, num_heads: int, qkv_features: int,
                 max_temporal_len: int, spatial_compression_rate: int):
        super().__init__()
        max_spatial_len = (height // patch_size) * (width // patch_size)
        self.last_dim = channels * patch_size * patch_size

        self.patch_embedding = PatchEmbedding(height, width, channels, patch_size)
        self.spatial_compression = nn.Linear(
            self.last_dim, self.last_dim // spatial_compression_rate)
        self.variance_estimator = nn.Linear(
            self.last_dim, self.last_dim // spatial_compression_rate)
        self.selection_layer1 = nn.Linear(
            self.last_dim // spatial_compression_rate, 1)
        self.selection_layer2 = nn.Linear(max_spatial_len, 1)
        self.gumbel_sigmoid = GumbelSigmoidSTE(temperature=1.0)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(FactoredAttention(
                mlp_dim=mlp_dim,
                in_features=self.last_dim,
                num_heads=num_heads,
                qkv_features=qkv_features,
                max_temporal_len=max_temporal_len,
                max_spatial_len=max_spatial_len,
            ))

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                train: bool = True):
        """
        Args:
            x: (B, T, H, W, C)
            mask: (B, 1, 1, T)
        Returns:
            mean: (B, T, hw, compressed_dim)
            variance: (B, T, hw, compressed_dim)
            selection: (B, T, 1)
        """
        x = self.patch_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        mean = self.spatial_compression(x)
        variance = F.softplus(self.variance_estimator(x).float())
        variance = (variance + 1e-6).to(mean.dtype)
        selection_intermediate = self.selection_layer1(mean)
        selection_intermediate = rearrange(selection_intermediate, "b t hw 1 -> b t hw")
        selection = torch.sigmoid(self.selection_layer2(selection_intermediate) + 1)
        return mean, variance, selection


class Decoder(nn.Module):
    def __init__(self, height: int, width: int, channels: int, patch_size: int,
                 depth: int, mlp_dim: int, num_heads: int, qkv_features: int,
                 max_temporal_len: int, spatial_compression_rate: int,
                 unembedding_upsample_rate: int):
        super().__init__()
        self.last_dim = channels * patch_size * patch_size
        self.patch_unembedding = PatchUnEmbedding(
            height, width, channels, patch_size, unembedding_upsample_rate)
        self.spatial_decompression = nn.Linear(
            self.last_dim // spatial_compression_rate, self.last_dim)

        max_spatial_len = (height // patch_size) * (width // patch_size)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(FactoredAttention(
                mlp_dim=mlp_dim,
                in_features=self.last_dim,
                num_heads=num_heads,
                qkv_features=qkv_features,
                max_temporal_len=max_temporal_len,
                max_spatial_len=max_spatial_len,
            ))
        self.unet = UNet(
            channels=channels * unembedding_upsample_rate,
            base_features=16, num_levels=3, out_features=channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                train: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, T, hw, compressed_dim)
            mask: (B, 1, 1, T)
        Returns:
            (B, T, H, W, C) reconstructed video
        """
        x = self.spatial_decompression(x)
        for layer in self.layers:
            x = layer(x, mask)
        convolutional_upsampled_features, x = self.patch_unembedding(x)
        unet_output = self.unet(convolutional_upsampled_features)
        x = x + unet_output
        return x


class VideoVAE(nn.Module):
    def __init__(self, height: int = 256, width: int = 256, channels: int = 3,
                 patch_size: int = 16, encoder_depth: int = 9,
                 decoder_depth: int = 12, mlp_dim: int = 1536,
                 num_heads: int = 8, qkv_features: int = 512,
                 max_temporal_len: int = 64, spatial_compression_rate: int = 8,
                 unembedding_upsample_rate: int = 4):
        super().__init__()
        self.encoder = Encoder(
            height, width, channels, patch_size, encoder_depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate)
        self.decoder = Decoder(
            height, width, channels, patch_size, decoder_depth,
            mlp_dim, num_heads, qkv_features, max_temporal_len,
            spatial_compression_rate, unembedding_upsample_rate)

        compressed_dim = channels * patch_size * patch_size // spatial_compression_rate
        self.fill_token = nn.Parameter(
            torch.randn(1, 1, 1, compressed_dim) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                train: bool = True, p: int = 2):
        """
        Full forward pass with selection sampling.
        Args:
            x: (B, T, H, W, C)
            mask: (B, 1, 1, T)
            train: whether in training mode
            p: number of selection samples per batch element
        """
        mean, variance, selection = self.encoder(x, mask, train=train)

        if train:
            noise = torch.randn_like(variance)
            std = torch.sqrt(variance)
            sampled_latent = mean + noise * std
        else:
            sampled_latent = mean

        selection = repeat(selection, "b t 1 -> (b p) t 1 1", p=p)
        sampled_latent = repeat(sampled_latent, "b ... -> (b p) ...", p=p)
        mean = repeat(mean, "b ... -> (b p) ...", p=p)
        variance = repeat(variance, "b ... -> (b p) ...", p=p)
        mask = repeat(mask, "b ... -> (b p) ...", p=p)

        selection_mask = torch.bernoulli(selection.expand_as(
            sampled_latent[:, :, :1, :1].expand(-1, -1, 1, 1)
        ).squeeze(-1).squeeze(-1).unsqueeze(-1).unsqueeze(-1))
        # Simpler: just use selection directly
        selection_mask = torch.bernoulli(
            selection.expand(-1, -1, -1, -1)
        ).to(sampled_latent.dtype)

        compressed_representation = (
            self.fill_token * (1 - selection_mask) +
            sampled_latent * selection_mask
        )
        reconstruction = self.decoder(compressed_representation, mask, train=train)
        return reconstruction, compressed_representation, selection, selection_mask, variance, mean

    def compress(self, x: torch.Tensor, mask: torch.Tensor,
                 train: bool = False):
        """
        Encode video to compressed latent representation.
        Args:
            x: (B, T, H, W, C)
            mask: (B, 1, 1, T)
        Returns:
            compressed: (B, T, hw, compressed_dim) left-packed
            selection_indices: (B, T) adjacent differences
            compression_mask: (B, T) bool mask of valid positions
        """
        mean, variance, selection_probs = self.encoder(x, mask, train=train)

        if train:
            noise = torch.randn_like(variance)
            std = torch.sqrt(variance)
            sampled_latent = mean + noise * std
            selection_mask = torch.bernoulli(selection_probs)
        else:
            sampled_latent = mean
            selection_mask = (selection_probs > 0.5).float()

        selection_mask = rearrange(selection_mask, "b t 1 -> b t")

        # Convert to indices and left-pack
        compressed_list = []
        indices_list = []
        mask_list = []

        b, t, hw, d = sampled_latent.shape
        for bi in range(b):
            sel = selection_mask[bi]  # (T,)
            # Find indices where selection is 1
            active_indices = torch.where(sel > 0.5)[0]
            dynamic_len = active_indices.shape[0]

            # Gather selected frames
            gathered = torch.zeros(t, hw, d, device=x.device, dtype=sampled_latent.dtype)
            valid_mask = torch.zeros(t, device=x.device, dtype=torch.bool)

            if dynamic_len > 0:
                gathered[:dynamic_len] = sampled_latent[bi, active_indices]
                valid_mask[:dynamic_len] = True

            # Compute adjacent differences of indices
            adj_indices = torch.zeros(t, device=x.device, dtype=torch.long)
            if dynamic_len > 0:
                padded_indices = torch.zeros(dynamic_len, device=x.device, dtype=torch.long)
                padded_indices[:] = active_indices
                adj_diff = torch.diff(padded_indices, prepend=torch.tensor([0], device=x.device))
                adj_indices[:dynamic_len] = adj_diff

            compressed_list.append(gathered)
            indices_list.append(adj_indices)
            mask_list.append(valid_mask)

        compressed = torch.stack(compressed_list)
        selection_indices = torch.stack(indices_list)
        compression_mask = torch.stack(mask_list)

        return compressed, selection_indices, compression_mask

    def decompress(self, compressed: torch.Tensor, attention_mask: torch.Tensor,
                   selection_indices: torch.Tensor, compression_mask: torch.Tensor,
                   train: bool = False, output_length: int = None):
        """
        Decode compressed latent back to video.

        Scatters compressed latent frames to their correct temporal positions
        (determined by cumsum of selection_indices) and fills gaps with the
        learned fill_token before passing through the decoder.

        Args:
            compressed: (B, T_compressed, hw, d) left-packed latent frames
            attention_mask: (B, 1, 1, T_out) mask for decoder attention
            selection_indices: (B, T_compressed) adjacent differences (frame gaps)
            compression_mask: (B, T_compressed) bool mask of valid entries
            train: whether in training mode
            output_length: explicit output temporal length. If None, inferred
                           from cumsum of selection_indices (last valid position + 1).
        Returns:
            reconstruction: (B, T_out, H, W, C)
        """
        b, t_comp, hw, d = compressed.shape
        fill = self.fill_token.view(1, d)  # (1, d) for broadcasting over hw
        device = compressed.device

        # Convert adjacent differences to absolute positions
        abs_indices = torch.cumsum(selection_indices.long(), dim=1)

        # Determine output length
        if output_length is None:
            # Infer from the last valid position
            valid_pos = abs_indices * compression_mask.long()
            output_length = int(valid_pos.max().item()) + 1

        t_out = output_length

        result_list = []
        for bi in range(b):
            # Start with fill token everywhere: (t_out, hw, d)
            result = fill.expand(hw, d).unsqueeze(0).expand(t_out, hw, d).clone()
            mask_b = compression_mask[bi]
            indices_b = abs_indices[bi]

            # Scatter valid compressed frames to their positions
            for ti in range(t_comp):
                if mask_b[ti]:
                    idx = int(indices_b[ti].item())
                    if 0 <= idx < t_out:
                        result[idx] = compressed[bi, ti]

            result_list.append(result)

        full_representation = torch.stack(result_list)

        # Build attention mask for the output length
        attn_mask = torch.ones(b, 1, 1, t_out, dtype=torch.bool, device=device)
        reconstruction = self.decoder(full_representation, attn_mask, train=train)
        return reconstruction
