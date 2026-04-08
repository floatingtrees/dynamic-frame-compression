"""
Compress a video using the VAE encoder.

Usage:
    python compress.py --input video.mp4 --output compressed.pt
    python compress.py --input video.mp4 --output compressed.pt --max_frames 32
"""

import argparse
import os
import numpy as np
import torch
import imageio

from model_loader import load_vae


def load_video(path: str, max_frames: int = 32, resize: tuple = (256, 256)) -> np.ndarray:
    """Load a video file and preprocess it.

    Args:
        path: Path to video file
        max_frames: Maximum number of frames to load
        resize: (height, width) to resize frames to

    Returns:
        video: (1, T, H, W, 3) float32 array normalized to roughly [-1, 1]
    """
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        # Resize if needed
        if frame.shape[0] != resize[0] or frame.shape[1] != resize[1]:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((resize[1], resize[0]), Image.LANCZOS)
            frame = np.array(img)
        frames.append(frame)
    reader.close()

    video = np.stack(frames, axis=0).astype(np.float32)
    # Normalize from [0, 255] to [0, 1] (model expects [0,1])
    video = video / 255.0
    # Add batch dimension
    video = video[np.newaxis]  # (1, T, H, W, C)
    return video


@torch.no_grad()
def compress_video(vae: VideoVAE, video: np.ndarray,
                   device: str = "cuda") -> dict:
    """
    Compress a video using the VAE encoder.

    Args:
        vae: Loaded VAE model
        video: (1, T, H, W, 3) float32 array

    Returns:
        dict with compressed, selection_indices, compression_mask
    """
    video_t = torch.tensor(video, device=device, dtype=torch.bfloat16)
    t = video.shape[1]
    mask = torch.ones(1, 1, 1, t, dtype=torch.bool, device=device)

    compressed, selection_indices, compression_mask = vae.compress(video_t, mask, train=False)

    return {
        "compressed": compressed.cpu(),
        "selection_indices": selection_indices.cpu(),
        "compression_mask": compression_mask.cpu(),
        "original_num_frames": t,
        "num_kept_frames": int(compression_mask.sum().item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compress a video with VAE")
    parser.add_argument("--input", type=str, required=True,
                        help="Input video path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output compressed representation path (.pt)")
    parser.add_argument("--vae_checkpoint", type=str, default="vae_pytorch.pt",
                        help="Path to VAE weights")
    parser.add_argument("--max_frames", type=int, default=32,
                        help="Maximum frames to process")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    vae = load_vae(args.vae_checkpoint, args.device)

    print(f"Loading video from {args.input}...")
    video = load_video(args.input, max_frames=args.max_frames)
    print(f"  Video shape: {video.shape}")

    print("Compressing...")
    result = compress_video(vae, video, args.device)

    print(f"  Compressed shape: {result['compressed'].shape}")
    print(f"  Kept {result['num_kept_frames']}/{result['original_num_frames']} frames")

    # Compute compression ratio
    original_size = np.prod(video.shape) * 4  # float32 bytes
    compressed_size = (result["compressed"].numel() * 4 +
                       result["selection_indices"].numel() * 8 +
                       result["compression_mask"].numel())
    ratio = original_size / compressed_size
    print(f"  Compression ratio: {ratio:.1f}x")

    torch.save(result, args.output)
    print(f"Saved compressed representation to {args.output}")


if __name__ == "__main__":
    main()
