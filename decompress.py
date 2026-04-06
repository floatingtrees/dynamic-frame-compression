"""
Decompress a video from compressed latent representation using the VAE decoder.

Usage:
    python decompress.py --input compressed.pt --output reconstructed.mp4
"""

import argparse
import os
import numpy as np
import torch
import imageio

from autoencoder import VideoVAE


def load_vae(checkpoint_path: str, device: str = "cuda") -> VideoVAE:
    model = VideoVAE()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def save_video(frames: np.ndarray, output_path: str, fps: float = 30.0):
    """Save video frames to an mp4 file."""
    frames = np.clip(frames, 0.0, 1.0)
    frames_uint8 = (frames * 255).astype(np.uint8)
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                quality=8, pixelformat='yuv420p')
    for i in range(frames_uint8.shape[0]):
        writer.append_data(frames_uint8[i])
    writer.close()
    print(f"Saved video to {output_path} ({frames_uint8.shape[0]} frames, {fps} fps)")


@torch.no_grad()
def decompress_video(vae: VideoVAE, compressed_data: dict,
                     device: str = "cuda") -> np.ndarray:
    """
    Decompress a video from compressed latent representation.

    Args:
        vae: Loaded VAE model
        compressed_data: dict from compress.py containing compressed, selection_indices,
                         compression_mask, original_num_frames

    Returns:
        video: (T, H, W, 3) float32 array in [0, 1]
    """
    compressed = compressed_data["compressed"].to(device)
    selection_indices = compressed_data["selection_indices"].to(device)
    compression_mask = compressed_data["compression_mask"].to(device)
    t = compressed_data["original_num_frames"]

    attention_mask = torch.ones(1, 1, 1, t, dtype=torch.bool, device=device)

    video = vae.decompress(compressed, attention_mask, selection_indices,
                           compression_mask, train=False)

    # (1, T, H, W, C) -> (T, H, W, C)
    video_np = video[0].cpu().float().numpy()
    # Convert from model range to [0, 1]
    video_np = (video_np + 1.0) / 2.0
    return video_np


def main():
    parser = argparse.ArgumentParser(description="Decompress video from latent")
    parser.add_argument("--input", type=str, required=True,
                        help="Input compressed representation path (.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output video path (.mp4)")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_pytorch.pt",
                        help="Path to VAE weights")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output video frame rate")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae = load_vae(args.vae_checkpoint, args.device)

    print(f"Loading compressed data from {args.input}...")
    compressed_data = torch.load(args.input, map_location="cpu", weights_only=True)
    print(f"  Compressed shape: {compressed_data['compressed'].shape}")
    print(f"  Original frames: {compressed_data['original_num_frames']}")
    print(f"  Kept frames: {compressed_data['num_kept_frames']}")

    print("Decompressing...")
    video = decompress_video(vae, compressed_data, args.device)
    print(f"  Reconstructed video shape: {video.shape}")

    save_video(video, args.output, fps=args.fps)


if __name__ == "__main__":
    main()
