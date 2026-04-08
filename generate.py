"""
Generate video from noise using the DiT + VAE pipeline.

The DiT generates compressed latent frames along with predicted frame gaps
(adjacent differences). The gaps determine where each latent frame maps in
the output video, and the VAE decoder fills in the gaps with a learned fill
token. The total output length = sum of predicted gaps.

Usage:
    python generate.py --num_latent_frames 16 --num_steps 100 --output generated_video.mp4
    python generate.py --num_latent_frames 8 --num_steps 50 --seed 42 --output my_video.mp4
"""

import argparse
import os
import numpy as np
import torch
import imageio

from diffusion_model import sample, gaps_to_positions
from model_loader import load_vae, load_dit


def save_video(frames: np.ndarray, output_path: str, fps: float = 30.0):
    """Save (T, H, W, C) float frames as mp4."""
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
def generate(dit, vae, num_latent_frames: int = 16,
             num_steps: int = 100, seed: int = 42,
             device: str = "cuda") -> np.ndarray:
    """
    Generate a video using DiT sampling + VAE decoding with frame gap prediction.

    The DiT generates `num_latent_frames` compressed latent frames. It also
    predicts frame gaps (adjacent differences) that specify where each latent
    frame should be placed in the output video. The total output video length
    is determined by the sum of these gaps.

    Args:
        dit: Loaded DiT model
        vae: Loaded VAE model
        num_latent_frames: Number of latent frames to generate (compressed)
        num_steps: Number of Euler integration steps for sampling
        seed: Random seed
        device: Device to run on

    Returns:
        video: (total_frames, 256, 256, 3) float array in [0,1]
    """
    torch.manual_seed(seed)

    hw = 256        # spatial patches (16x16 patches from 256x256)
    compressed_dim = 96

    # Sample noise for compressed latent frames
    noise = torch.randn(1, num_latent_frames, hw, compressed_dim,
                         device=device, dtype=torch.bfloat16)
    compression_mask = torch.ones(1, num_latent_frames, dtype=torch.bool, device=device)

    print(f"Sampling with DiT ({num_steps} steps, {num_latent_frames} latent frames, bf16)...")
    latent, gap_pred = sample(dit, noise, compression_mask, num_steps=num_steps)

    # Convert predicted gaps to absolute positions and total output length
    positions, total_frames = gaps_to_positions(gap_pred, compression_mask)
    total_frames_int = int(total_frames[0].item())

    # Build adjacent differences from positions for decompress()
    gaps_int = gap_pred.float().round().long()
    gaps_int[:, 0] = gaps_int[:, 0].clamp(min=0)
    if gaps_int.shape[1] > 1:
        gaps_int[:, 1:] = gaps_int[:, 1:].clamp(min=1)
    gaps_int = gaps_int * compression_mask.long()

    print(f"  Predicted gaps: {gaps_int[0].tolist()}")
    print(f"  Frame positions: {positions[0].tolist()}")
    print(f"  Output video length: {total_frames_int} frames "
          f"(from {num_latent_frames} latent frames)")

    print("Decoding with VAE (scattering to positions, filling gaps)...")
    video = vae.decompress(
        latent, None,  # attention_mask built internally
        gaps_int, compression_mask,
        train=False, output_length=total_frames_int)

    print(f"  Video shape: {video.shape}")

    # Model outputs [0, 1] directly
    video_np = video[0].cpu().float().numpy()
    print(f"  Output range: [{video_np.min():.3f}, {video_np.max():.3f}]")
    return video_np


def main():
    parser = argparse.ArgumentParser(description="Generate video with DiT + VAE")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_pytorch.pt",
                        help="Path to converted VAE PyTorch weights")
    parser.add_argument("--dit_checkpoint", type=str, default="dit_pytorch.pt",
                        help="Path to converted DiT PyTorch weights")
    parser.add_argument("--num_latent_frames", type=int, default=16,
                        help="Number of compressed latent frames to generate")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of Euler integration steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="generated_video.mp4",
                        help="Output video path")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="Directory to save individual frames as PNGs")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Output video frame rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    vae = load_vae(args.vae_checkpoint, args.device)
    dit = load_dit(args.dit_checkpoint, args.device)

    video = generate(dit, vae, num_latent_frames=args.num_latent_frames,
                     num_steps=args.num_steps, seed=args.seed,
                     device=args.device)

    save_video(video, args.output, fps=args.fps)

    if args.save_frames:
        save_frames(video, args.save_frames)


if __name__ == "__main__":
    main()
