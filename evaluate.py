"""
Evaluation script for README documentation.
- Autoencoder reconstruction quality on 600+ real videos
- Standard Bernoulli compression stats (average frame keep rate)
- Compression at different frame budgets on diverse videos
- Video generation from DiT with frame-gap prediction (4x4 grid)
"""

import os
import glob
import random
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from diffusion_model import sample, gaps_to_positions
from model_loader import load_vae, load_dit


DEVICE = "cuda"
DOC_DIR = "docs"


def to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.float().cpu().numpy()
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def get_font(size=14):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def load_video(path, max_frames=32, resize=(256, 256), start_frame=0):
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if i < start_frame:
            continue
        if len(frames) >= max_frames:
            break
        if frame.shape[0] != resize[0] or frame.shape[1] != resize[1]:
            img = Image.fromarray(frame).resize((resize[1], resize[0]), Image.LANCZOS)
            frame = np.array(img)
        frames.append(frame)
    reader.close()
    if len(frames) == 0:
        return None
    return np.stack(frames).astype(np.float32) / 255.0


def save_gif(frames_01, path, fps=12):
    frames_u8 = [to_uint8(f) for f in frames_01]
    imageio.mimwrite(path, frames_u8, format="GIF", duration=1000 / fps, loop=0)


def shrink_gif(path, scale="iw*3/4", max_colors=64):
    tmp = path + ".tmp.gif"
    os.system(f'ffmpeg -y -i "{path}" -vf "fps=8,scale={scale}:-1:flags=lanczos,'
              f'split[s0][s1];[s0]palettegen=max_colors={max_colors}[p];'
              f'[s1][p]paletteuse=dither=bayer:bayer_scale=3" -loop 0 "{tmp}" 2>/dev/null')
    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)


def hstack_videos(videos_01, pad=2):
    T = min(v.shape[0] for v in videos_01)
    out = []
    for t in range(T):
        row = [to_uint8(v[t]) for v in videos_01]
        sep = np.full((row[0].shape[0], pad, 3), 255, dtype=np.uint8)
        parts = []
        for i, r in enumerate(row):
            if i > 0:
                parts.append(sep)
            parts.append(r)
        out.append(np.concatenate(parts, axis=1))
    return np.stack(out).astype(np.float32) / 255.0


def vstack_videos(videos_01, pad=2):
    T = min(v.shape[0] for v in videos_01)
    out = []
    for t in range(T):
        col = [to_uint8(v[t]) for v in videos_01]
        sep = np.full((pad, col[0].shape[1], 3), 255, dtype=np.uint8)
        parts = []
        for i, c in enumerate(col):
            if i > 0:
                parts.append(sep)
            parts.append(c)
        out.append(np.concatenate(parts, axis=0))
    return np.stack(out).astype(np.float32) / 255.0


def label_frame(frame_u8, text, font):
    img = Image.fromarray(frame_u8)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 16)], fill=(0, 0, 0))
    draw.text((3, 0), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def label_video(frames_01, text, font):
    return np.stack([label_frame(to_uint8(f), text, font).astype(np.float32) / 255.0
                     for f in frames_01])


# ──────────────────────────────────────────────────────────────
# 1. MSE + standard compression stats over 600+ videos
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mse(vae, video_dir, num_videos=600, max_frames=32):
    print(f"\n=== Evaluating autoencoder on {num_videos} videos ===")
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))[:num_videos]

    all_mse = []
    all_keep_rates = []
    all_bernoulli_mse = []

    for idx, path in enumerate(paths):
        video_01 = load_video(path, max_frames=max_frames)
        if video_01 is None or video_01.shape[0] < 2:
            continue
        T = video_01.shape[0]
        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

        mean, _, sel_probs = vae.encoder(video_t, mask4d, train=False)
        fill = vae.fill_token.to(torch.bfloat16)

        # All-frames reconstruction MSE
        recon = vae.decoder(mean, mask4d, train=False)
        recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)
        mse = float(np.mean((video_01 - recon_np) ** 2))
        all_mse.append(mse)

        # Standard Bernoulli selection (matching training behavior)
        sel = sel_probs.squeeze(-1).squeeze(0).float().cpu().numpy()  # (T,)
        # Bernoulli sample (use fixed seed per video for reproducibility)
        torch.manual_seed(idx)
        sel_mask = torch.bernoulli(sel_probs.squeeze(-1)).to(torch.bfloat16)  # (1, T)
        keep_rate = float(sel_mask.sum().item()) / T
        all_keep_rates.append(keep_rate)

        # Bernoulli reconstruction
        sel_mask_4d = sel_mask.unsqueeze(-1).unsqueeze(-1)  # (1, T, 1, 1)
        compressed = mean * sel_mask_4d + fill * (1 - sel_mask_4d)
        recon_b = vae.decoder(compressed, mask4d, train=False)
        recon_b_np = np.clip(recon_b[0].float().cpu().numpy(), 0, 1)
        mse_b = float(np.mean((video_01 - recon_b_np) ** 2))
        all_bernoulli_mse.append(mse_b)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(paths)}] all-frames MSE={np.mean(all_mse):.6f}, "
                  f"Bernoulli MSE={np.mean(all_bernoulli_mse):.6f}, "
                  f"keep rate={np.mean(all_keep_rates):.3f}")

    n = len(all_mse)
    mean_mse = float(np.mean(all_mse))
    mean_bernoulli_mse = float(np.mean(all_bernoulli_mse))
    mean_keep = float(np.mean(all_keep_rates))
    std_keep = float(np.std(all_keep_rates))
    print(f"\n  Results over {n} videos:")
    print(f"    All-frames MSE:    {mean_mse:.6f}")
    print(f"    Bernoulli MSE:     {mean_bernoulli_mse:.6f}")
    print(f"    Mean keep rate:    {mean_keep:.3f} ± {std_keep:.3f} "
          f"({mean_keep*32:.1f}/{32} frames avg)")
    print(f"    Temporal compress:  {1/mean_keep:.1f}x average")

    return {
        "n": n,
        "all_frames_mse": mean_mse,
        "bernoulli_mse": mean_bernoulli_mse,
        "keep_rate": mean_keep,
        "keep_rate_std": std_keep,
    }


# ──────────────────────────────────────────────────────────────
# 2. Reconstruction demos
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def reconstruction_demos(vae, video_paths):
    print(f"\n=== Reconstruction demos ({len(video_paths)} videos) ===")
    font = get_font(11)

    for i, (path, start, label) in enumerate(video_paths):
        video_01 = load_video(path, max_frames=32, start_frame=start)
        if video_01 is None:
            continue
        T = video_01.shape[0]
        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)
        mean, _, _ = vae.encoder(video_t, mask4d, train=False)
        recon = vae.decoder(mean, mask4d, train=False)
        recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)
        mse = float(np.mean((video_01 - recon_np) ** 2))

        labeled = hstack_videos([
            label_video(video_01, "Original", font),
            label_video(recon_np, f"Reconstruction (MSE={mse:.4f})", font),
        ])
        out_path = os.path.join(DOC_DIR, f"recon_video{i}.gif")
        save_gif(labeled, out_path, fps=12)
        shrink_gif(out_path)
        print(f"  recon_video{i}.gif: {label}, MSE={mse:.6f}")


# ──────────────────────────────────────────────────────────────
# 3. Compression demos with standard Bernoulli mode
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compression_demos(vae, video_paths, max_frames=32):
    print(f"\n=== Compression demos ({len(video_paths)} videos, {max_frames} frames) ===")
    font = get_font(10)

    for vi, (path, start, label) in enumerate(video_paths):
        video_01 = load_video(path, max_frames=max_frames, start_frame=start)
        if video_01 is None:
            continue
        T = video_01.shape[0]
        half = T // 2
        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

        mean, _, selection_probs = vae.encoder(video_t, mask4d, train=False)
        latent = mean
        fill = vae.fill_token.to(torch.bfloat16)
        sel = selection_probs.squeeze(-1).squeeze(0).float().cpu().numpy()

        # Half frames baseline (top-half by selection score)
        topk_half = np.sort(np.argsort(sel)[-half:])
        sm_half = torch.zeros(1, T, 1, 1, device=DEVICE, dtype=torch.bfloat16)
        sm_half[0, topk_half, 0, 0] = 1.0
        recon_half = vae.decoder(latent * sm_half + fill * (1 - sm_half), mask4d, train=False)
        recon_half_np = np.clip(recon_half[0].float().cpu().numpy(), 0, 1)
        mse_half = float(np.mean((video_01 - recon_half_np) ** 2))

        parts = [
            label_video(video_01, "Original", font),
            label_video(recon_half_np, f"Top-{half} MSE={mse_half:.4f}", font),
        ]

        # Argmax budgets
        for budget in [8, 4, 1]:
            topk = np.sort(np.argsort(sel)[-budget:])
            sm = torch.zeros(1, T, 1, 1, device=DEVICE, dtype=torch.bfloat16)
            sm[0, topk, 0, 0] = 1.0
            r = vae.decoder(latent * sm + fill * (1 - sm), mask4d, train=False)
            r_np = np.clip(r[0].float().cpu().numpy(), 0, 1)
            mse = float(np.mean((video_01 - r_np) ** 2))
            parts.append(label_video(r_np, f"Top-{budget} MSE={mse:.4f}", font))

        composite = hstack_videos(parts, pad=2)
        out_path = os.path.join(DOC_DIR, f"compress_video{vi}.gif")
        save_gif(composite, out_path, fps=12)
        shrink_gif(out_path)
        print(f"  compress_video{vi}.gif: {label}, half={half}/{T}")


# ──────────────────────────────────────────────────────────────
# 4. Generation: 4x4 grid
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_grid(dit, vae, num_latent=8):
    print(f"\n=== Generating 5x5 grid ({num_latent} latent frames per video) ===")
    font = get_font(10)
    seeds = [0, 7, 13, 42, 55, 77, 99, 123, 200, 256,
             333, 404, 512, 600, 700, 777, 888, 999, 1024, 1111,
             1337, 1500, 1776, 2000, 2025]

    rows = []
    row_vids = []

    for seed in seeds:
        torch.manual_seed(seed)
        noise = torch.randn(1, num_latent, 256, 96, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(1, num_latent, dtype=torch.bool, device=DEVICE)
        latent, gap_pred = sample(dit, noise, mask, num_steps=100)
        _, total = gaps_to_positions(gap_pred, mask)
        t_out = int(total[0].item())
        gaps_int = gap_pred.float().round().long()
        gaps_int[:, 0] = gaps_int[:, 0].clamp(min=0)
        if gaps_int.shape[1] > 1:
            gaps_int[:, 1:] = gaps_int[:, 1:].clamp(min=1)
        gaps_int = gaps_int * mask.long()

        video = vae.decompress(latent, None, gaps_int, mask,
                               train=False, output_length=t_out)
        v_np = np.clip(video[0].float().cpu().numpy(), 0, 1)
        v_labeled = label_video(v_np, f"s={seed} ({t_out}f)", font)
        row_vids.append(v_labeled)
        print(f"  seed={seed:4d}: {t_out} frames")

        if len(row_vids) == 5:
            rows.append(hstack_videos(row_vids, pad=2))
            row_vids = []

    composite = vstack_videos(rows, pad=2)
    out_path = os.path.join(DOC_DIR, "generated_grid.gif")
    save_gif(composite, out_path, fps=12)
    shrink_gif(out_path, scale="iw*3/5", max_colors=64)
    print(f"  Saved generated_grid.gif")


# ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(DOC_DIR, exist_ok=True)

    vae = load_vae()
    dit = load_dit()

    # --- VAE ---
    stats = evaluate_mse(vae, "/mnt/t9/videos/videos1", num_videos=600, max_frames=32)

    recon_paths = [
        ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-aerial-panorama-of-a-coast-and-its-reliefs-36615_000.mp4", 0, "aerial coast"),
        ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-a-man-doing-jumping-tricks-at-the-beach-1222_001.mp4", 0, "beach jumping"),
        ("/projects/video-VAE/inference/test_videos/videos0/videos0/9bZkp7q19f0.mp4", 100, "dance (high motion)"),
        ("/projects/video-VAE/inference/test_videos/videos0/videos0/dQw4w9WgXcQ.mp4", 200, "music video (high motion)"),
    ]
    reconstruction_demos(vae, recon_paths)

    compress_paths = [
        ("/mnt/t9/videos/videos1/pixabay_seascapes_videos_145647_012.mp4", 0, "seascape"),
        ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-flying-over-a-palm-covered-beach-44364_000.mp4", 0, "palm beach aerial"),
        ("/projects/video-VAE/inference/test_videos/videos0/videos0/9bZkp7q19f0.mp4", 300, "dance (high motion)"),
        ("/projects/video-VAE/inference/test_videos/videos0/videos0/dQw4w9WgXcQ.mp4", 600, "music video (high motion)"),
    ]
    compression_demos(vae, compress_paths, max_frames=32)

    # --- DiT ---
    generate_grid(dit, vae, num_latent=8)

    # Write summary
    with open(os.path.join(DOC_DIR, "eval_results.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

    print("\n=== All done! ===")
    for fn in sorted(os.listdir(DOC_DIR)):
        sz = os.path.getsize(os.path.join(DOC_DIR, fn))
        print(f"  {fn} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
