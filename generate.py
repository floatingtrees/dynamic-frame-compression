"""
Generate video from noise using the DiT + VAE pipeline.

Usage:
    python generate.py --num_frames 16 --num_steps 100 --output generated_video.mp4
    python generate.py --num_frames 8 --num_steps 50 --seed 42 --output my_video.mp4
"""

import argparse
import os
import numpy as np
import torch
import imageio

from autoencoder import VideoVAE
from diffusion_model import VideoDiT, sample


def load_vae(checkpoint_path: str, device: str = "cuda") -> VideoVAE:
    """Load the VAE model with converted PyTorch weights in bfloat16."""
    model = VideoVAE()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_dit(checkpoint_path: str, device: str = "cuda") -> VideoDiT:
    """Load the DiT model with converted PyTorch weights in bfloat16."""
    model = VideoDiT(depth=30)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def save_video(frames: np.ndarray, output_path: str, fps: float = 30.0):
    """
    Save video frames to an mp4 file.

    Args:
        frames: (T, H, W, C) float array in [0, 1] or arbitrary range (will be clipped)
        output_path: path to save the mp4 file
        fps: frames per second
    """
    # Clip to [0, 1] and convert to uint8
    frames = np.clip(frames, 0.0, 1.0)
    frames_uint8 = (frames * 255).astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                quality=8, pixelformat='yuv420p')
    for i in range(frames_uint8.shape[0]):
        writer.append_data(frames_uint8[i])
    writer.close()
    print(f"Saved video to {output_path} ({frames_uint8.shape[0]} frames, {fps} fps)")


def save_frames(frames: np.ndarray, output_dir: str):
    """Save individual frames as PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    frames = np.clip(frames, 0.0, 1.0)
    frames_uint8 = (frames * 255).astype(np.uint8)
    for i in range(frames_uint8.shape[0]):
        path = os.path.join(output_dir, f"frame_{i:04d}.png")
        imageio.imwrite(path, frames_uint8[i])
    print(f"Saved {frames_uint8.shape[0]} frames to {output_dir}/")


@torch.no_grad()
def generate(dit: VideoDiT, vae: VideoVAE, num_frames: int = 16,
             num_steps: int = 100, seed: int = 42,
             device: str = "cuda") -> np.ndarray:
    """
    Generate a video using DiT sampling + VAE decoding.

    Args:
        dit: Loaded DiT model
        vae: Loaded VAE model
        num_frames: Number of frames to generate
        num_steps: Number of Euler integration steps for sampling
        seed: Random seed
        device: Device to run on

    Returns:
        video: (num_frames, 256, 256, 3) float array
    """
    torch.manual_seed(seed)

    # hw = 256 spatial patches (16x16 patches from 256x256)
    hw = 256
    compressed_dim = 96  # channels per spatial-compression rate

    # Sample noise in bfloat16
    noise = torch.randn(1, num_frames, hw, compressed_dim, device=device, dtype=torch.bfloat16)
    compression_mask = torch.ones(1, num_frames, dtype=torch.bool, device=device)

    print(f"Sampling with DiT ({num_steps} steps, bf16)...")
    latent, selection_pred = sample(dit, noise, compression_mask, num_steps=num_steps)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Selection prediction: {selection_pred}")

    print("Decoding with VAE...")
    attention_mask = torch.ones(1, 1, 1, num_frames, dtype=torch.bool, device=device)
    video = vae.decoder(latent, attention_mask, train=False)
    print(f"  Video shape: {video.shape}")

    # Convert to numpy: (1, T, H, W, C) -> (T, H, W, C)
    video_np = video[0].cpu().float().numpy()

    # Normalize to [0, 1] range
    # The model outputs values in roughly [-1, 1] range
    # We shift and scale to [0, 1]
    video_np = (video_np + 1.0) / 2.0

    print(f"  Output range: [{video_np.min():.3f}, {video_np.max():.3f}]")
    return video_np


def main():
    parser = argparse.ArgumentParser(description="Generate video with DiT + VAE")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_pytorch.pt",
                        help="Path to converted VAE PyTorch weights")
    parser.add_argument("--dit_checkpoint", type=str, default="dit_pytorch.pt",
                        help="Path to converted DiT PyTorch weights")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames to generate")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of Euler integration steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="generated_video.mp4",
                        help="Output video path")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="Directory to save individual frames as PNGs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    print(f"Loading VAE from {args.vae_checkpoint}...")
    vae = load_vae(args.vae_checkpoint, args.device)

    print(f"Loading DiT from {args.dit_checkpoint}...")
    dit = load_dit(args.dit_checkpoint, args.device)

    video = generate(dit, vae, num_frames=args.num_frames,
                     num_steps=args.num_steps, seed=args.seed,
                     device=args.device)

    save_video(video, args.output)

    if args.save_frames:
        save_frames(video, args.save_frames)


if __name__ == "__main__":
    main()
