"""
Bernoulli compression rate demo for slow- and fast-moving clips.

For each selected clip we encode the video, sample a compression mask with
Bernoulli sampling (matching training behavior), and reconstruct. The output
GIF filename includes the total frame count, e.g.:

    docs/bernoulli_slow_coast_32f.gif
    docs/bernoulli_fast_dance_32f.gif
"""

import os
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from model_loader import load_vae

DEVICE = "cuda"
DOC_DIR = "docs"


def to_uint8(x):
    if isinstance(x, torch.Tensor):
        x = x.float().cpu().numpy()
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def get_font(size=12):
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


def label_video(frames_01, text, font):
    out = []
    for f in frames_01:
        img = Image.fromarray(to_uint8(f))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.width, 18)], fill=(0, 0, 0))
        draw.text((3, 1), text, fill=(255, 255, 255), font=font)
        out.append(np.array(img).astype(np.float32) / 255.0)
    return np.stack(out)


@torch.no_grad()
def bernoulli_compress(vae, video_01, num_samples=8, seed=0):
    """Encode, Bernoulli-sample mask, decode. Returns recon + stats.

    num_samples: number of Bernoulli masks to average the keep rate over
                 (the reconstruction uses the first sample).
    """
    T = video_01.shape[0]
    video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
    mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

    mean, _, sel_probs = vae.encoder(video_t, mask4d, train=False)
    fill = vae.fill_token.to(torch.bfloat16)
    sel_probs_2d = sel_probs.squeeze(-1)  # (1, T)

    # Average keep rate over many Bernoulli draws
    torch.manual_seed(seed)
    keep_counts = []
    masks = []
    for _ in range(num_samples):
        m = torch.bernoulli(sel_probs_2d)  # (1, T)
        masks.append(m)
        keep_counts.append(float(m.sum().item()))
    avg_keep = float(np.mean(keep_counts))
    std_keep = float(np.std(keep_counts))

    # Reconstruction using the first Bernoulli mask
    sel_mask = masks[0].to(torch.bfloat16)
    sel_mask_4d = sel_mask.unsqueeze(-1).unsqueeze(-1)
    compressed = mean * sel_mask_4d + fill * (1 - sel_mask_4d)
    recon = vae.decoder(compressed, mask4d, train=False)
    recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)
    mse = float(np.mean((video_01 - recon_np) ** 2))

    return {
        "recon": recon_np,
        "selection_probs": sel_probs_2d.squeeze(0).float().cpu().numpy(),
        "first_mask": masks[0].squeeze(0).float().cpu().numpy(),
        "avg_keep_frames": avg_keep,
        "std_keep_frames": std_keep,
        "avg_keep_rate": avg_keep / T,
        "temporal_compression": T / max(avg_keep, 1e-6),
        "mse": mse,
        "T": T,
    }


# Curated slow + fast clips. Each entry: (path, start_frame, label, motion_class)
CLIPS = [
    # Slow-moving: aerial pans, static landscapes
    ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-aerial-panoramic-view-of-a-coastline-42492_000.mp4",
     0, "aerial_coastline_pan", "slow"),
    ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-a-luxury-tourist-island-with-a-pier-and-bungalows-2901_000.mp4",
     0, "tropical_island_aerial", "slow"),
    ("/mnt/t9/videos/videos1/pixabay_seascapes_videos_145647_012.mp4",
     0, "seascape_static", "slow"),

    # Fast-moving: jumping, music videos, action
    ("/mnt/t9/videos/videos1/mixkit_beach_mixkit-a-man-doing-jumping-tricks-at-the-beach-1222_001.mp4",
     0, "beach_jumping_tricks", "fast"),
    ("/projects/video-VAE/inference/test_videos/videos0/videos0/9bZkp7q19f0.mp4",
     100, "dance_performance", "fast"),
    ("/projects/video-VAE/inference/test_videos/videos0/videos0/dQw4w9WgXcQ.mp4",
     200, "music_video", "fast"),
]


@torch.no_grad()
def main():
    os.makedirs(DOC_DIR, exist_ok=True)
    vae = load_vae()
    font = get_font(11)

    print(f"\n{'='*80}")
    print("Bernoulli compression rate demos")
    print(f"{'='*80}\n")

    results = []
    for path, start, label, motion in CLIPS:
        if not os.path.isfile(path):
            print(f"  [skip] missing: {path}")
            continue
        video_01 = load_video(path, max_frames=32, start_frame=start)
        if video_01 is None:
            print(f"  [skip] empty: {path}")
            continue

        stats = bernoulli_compress(vae, video_01, num_samples=16, seed=42)
        T = stats["T"]

        # Filename includes motion class, label, and frame count
        out_name = f"bernoulli_{motion}_{label}_{T}f.gif"
        out_path = os.path.join(DOC_DIR, out_name)

        # Build labeled side-by-side: Original | Bernoulli reconstruction
        keep_f = stats["avg_keep_frames"]
        rate = stats["avg_keep_rate"]
        comp = stats["temporal_compression"]
        mse = stats["mse"]
        orig_label = f"Original ({T}f)"
        recon_label = f"Bernoulli keep={keep_f:.1f}/{T} ({comp:.1f}x)"
        composite = hstack_videos([
            label_video(video_01, orig_label, font),
            label_video(stats["recon"], recon_label, font),
        ])
        save_gif(composite, out_path, fps=12)
        shrink_gif(out_path)

        print(f"  [{motion:4s}] {label}")
        print(f"    Frames: {T}  keep: {keep_f:.2f} +/- {stats['std_keep_frames']:.2f}  "
              f"rate: {rate*100:.1f}%  compression: {comp:.2f}x  MSE: {mse:.4f}")
        print(f"    -> {out_path}")

        results.append({
            "motion": motion,
            "label": label,
            "T": T,
            "keep_frames": keep_f,
            "keep_std": stats["std_keep_frames"],
            "keep_rate": rate,
            "compression": comp,
            "mse": mse,
            "filename": out_name,
        })

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Motion':<6} {'Clip':<30} {'Frames':<8} {'Keep':<15} {'Rate':<8} {'Comp':<8} {'MSE':<8}")
    print(f"{'-'*80}")
    for r in results:
        keep_str = f"{r['keep_frames']:.1f} +/- {r['keep_std']:.1f}"
        print(f"{r['motion']:<6} {r['label']:<30} {r['T']:<8} {keep_str:<15} "
              f"{r['keep_rate']*100:>5.1f}%  {r['compression']:>5.2f}x  {r['mse']:.4f}")
    print(f"{'='*80}")

    slow = [r for r in results if r["motion"] == "slow"]
    fast = [r for r in results if r["motion"] == "fast"]
    if slow:
        print(f"\nSlow-moving mean compression: {np.mean([r['compression'] for r in slow]):.2f}x")
    if fast:
        print(f"Fast-moving mean compression: {np.mean([r['compression'] for r in fast]):.2f}x")

    # Write results to text file for the README
    with open(os.path.join(DOC_DIR, "bernoulli_rates.txt"), "w") as f:
        f.write("Motion  Clip                           Frames  Keep            Rate    Comp    MSE\n")
        f.write("-" * 85 + "\n")
        for r in results:
            keep_str = f"{r['keep_frames']:.1f} +/- {r['keep_std']:.1f}"
            f.write(f"{r['motion']:<6}  {r['label']:<30} {r['T']:<6}  {keep_str:<14}  "
                    f"{r['keep_rate']*100:>5.1f}%  {r['compression']:>5.2f}x  {r['mse']:.4f}\n")

    return results


if __name__ == "__main__":
    main()
