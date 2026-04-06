"""
Evaluation script for README documentation.
- Generate videos from DiT with frame-gap prediction
- Compress real videos at different frame budgets
- Compute MSE over real videos
- Produce comparison videos (GIF) for the README
"""

import os
import glob
import random
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from autoencoder import VideoVAE
from diffusion_model import VideoDiT, sample, gaps_to_positions


DEVICE = "cuda"
DOC_DIR = "docs"


def load_vae(path="vae_pytorch.pt"):
    model = VideoVAE()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False)
    model.to(device=DEVICE, dtype=torch.bfloat16).eval()
    return model


def load_dit(path="dit_pytorch.pt"):
    model = VideoDiT(depth=30)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False)
    model.to(device=DEVICE, dtype=torch.bfloat16).eval()
    return model


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


def load_video(path, max_frames=32, resize=(256, 256)):
    reader = imageio.get_reader(path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        if frame.shape[0] != resize[0] or frame.shape[1] != resize[1]:
            img = Image.fromarray(frame).resize((resize[1], resize[0]), Image.LANCZOS)
            frame = np.array(img)
        frames.append(frame)
    reader.close()
    return np.stack(frames).astype(np.float32) / 255.0


def save_gif(frames_01, path, fps=12):
    """Save (T,H,W,3) float [0,1] as looping GIF."""
    frames_u8 = [to_uint8(f) for f in frames_01]
    imageio.mimwrite(path, frames_u8, format="GIF", duration=1000/fps, loop=0)


def hstack_videos(videos_01, pad=4):
    """Horizontally stack a list of (T,H,W,3) videos (same T,H), return (T,H,W_total,3)."""
    T = min(v.shape[0] for v in videos_01)
    out_frames = []
    for t in range(T):
        row = [to_uint8(v[t]) for v in videos_01]
        sep = np.full((row[0].shape[0], pad, 3), 255, dtype=np.uint8)
        parts = []
        for i, r in enumerate(row):
            if i > 0:
                parts.append(sep)
            parts.append(r)
        out_frames.append(np.concatenate(parts, axis=1))
    return np.stack(out_frames).astype(np.float32) / 255.0


def label_frame(frame_u8, text, font):
    """Add a text label on top of a uint8 frame."""
    img = Image.fromarray(frame_u8)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 18)], fill=(0, 0, 0))
    draw.text((4, 1), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def label_video(frames_01, text, font):
    """Add persistent label to all frames of a video."""
    out = []
    for f in frames_01:
        out.append(label_frame(to_uint8(f), text, font).astype(np.float32) / 255.0)
    return np.stack(out)


# ──────────────────────────────────────────────────────────────
# 1. Generate videos with frame-gap prediction
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_videos(dit, vae, num_latent=8):
    print(f"\n=== Generating videos ({num_latent} latent frames) ===")
    font = get_font(12)
    all_videos = []

    for seed in [0, 42, 99, 256]:
        torch.manual_seed(seed)
        noise = torch.randn(1, num_latent, 256, 96, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(1, num_latent, dtype=torch.bool, device=DEVICE)
        latent, gap_pred = sample(dit, noise, mask, num_steps=100)
        positions, total = gaps_to_positions(gap_pred, mask)
        t_out = int(total[0].item())
        gaps_int = gap_pred.float().round().long()
        gaps_int[:, 0] = gaps_int[:, 0].clamp(min=0)
        if gaps_int.shape[1] > 1:
            gaps_int[:, 1:] = gaps_int[:, 1:].clamp(min=1)
        gaps_int = gaps_int * mask.long()

        video = vae.decompress(latent, None, gaps_int, mask,
                               train=False, output_length=t_out)
        video_np = np.clip(video[0].float().cpu().numpy(), 0, 1)
        gaps_str = ",".join(str(g) for g in gaps_int[0].tolist())
        print(f"  seed={seed}: {num_latent} latent -> {t_out} frames, gaps=[{gaps_str}]")

        save_gif(video_np, os.path.join(DOC_DIR, f"gen_seed{seed}.gif"), fps=12)
        all_videos.append((video_np, seed, t_out, gaps_int[0].tolist()))

    # Side-by-side composite of 4 seeds (trim to shortest)
    min_t = min(v[0].shape[0] for v in all_videos)
    trimmed = [v[0][:min_t] for v in all_videos]
    labeled = []
    for v, (_, seed, t_out, gaps) in zip(trimmed, all_videos):
        labeled.append(label_video(v, f"seed={seed} ({t_out}f)", font))
    composite = hstack_videos(labeled, pad=2)
    save_gif(composite, os.path.join(DOC_DIR, "generated_composite.gif"), fps=12)
    print(f"  Saved generated_composite.gif ({min_t} frames)")
    return all_videos


# ──────────────────────────────────────────────────────────────
# 2. Compress diverse real videos at different frame budgets
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compress_diverse_videos(vae, video_dir, max_frames=32):
    print(f"\n=== Compression comparison on diverse videos ===")
    font = get_font(11)

    # Pick diverse videos (different categories)
    all_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    # Sample spread out through the directory for diversity
    random.seed(42)
    candidates = random.sample(all_paths, min(200, len(all_paths)))
    # Pick the 3 most visually distinct (by filename prefix variation)
    seen_prefixes = set()
    picks = []
    for p in candidates:
        base = os.path.basename(p)
        prefix = "_".join(base.split("_")[:4])
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            picks.append(p)
        if len(picks) >= 3:
            break

    for vi, video_path in enumerate(picks):
        basename = os.path.basename(video_path)[:60]
        print(f"\n  Video {vi+1}: {basename}")
        video_01 = load_video(video_path, max_frames=max_frames)
        T = video_01.shape[0]

        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

        mean, _, selection_probs = vae.encoder(video_t, mask4d, train=False)
        latent = mean
        fill = vae.fill_token.to(torch.bfloat16)
        sel = selection_probs.squeeze(-1).squeeze(0).float().cpu().numpy()

        # All frames (baseline)
        recon_all = vae.decoder(latent, mask4d, train=False)
        recon_all_np = np.clip(recon_all[0].float().cpu().numpy(), 0, 1)
        mse_all = float(np.mean((video_01 - recon_all_np) ** 2))

        # Various budgets
        recons = {"all": (recon_all_np, mse_all, T)}
        for budget in [8, 4, 2, 1]:
            topk_idx = np.sort(np.argsort(sel)[-budget:])
            sel_mask = torch.zeros(1, T, 1, 1, device=DEVICE, dtype=torch.bfloat16)
            sel_mask[0, topk_idx, 0, 0] = 1.0
            compressed = latent * sel_mask + fill * (1 - sel_mask)
            recon = vae.decoder(compressed, mask4d, train=False)
            recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)
            mse = float(np.mean((video_01 - recon_np) ** 2))
            recons[budget] = (recon_np, mse, budget)
            print(f"    Top-{budget}: MSE={mse:.6f}")

        print(f"    All {T}f: MSE={mse_all:.6f}")

        # Build side-by-side GIF: original | all-frames | top-8 | top-4 | top-1
        parts = [
            (video_01, f"Original"),
            (recons["all"][0], f"All {T}f MSE={mse_all:.4f}"),
            (recons[8][0], f"Top-8 MSE={recons[8][1]:.4f}"),
            (recons[4][0], f"Top-4 MSE={recons[4][1]:.4f}"),
            (recons[1][0], f"Top-1 MSE={recons[1][1]:.4f}"),
        ]
        labeled = [label_video(v, lbl, font) for v, lbl in parts]
        composite = hstack_videos(labeled, pad=2)
        save_gif(composite, os.path.join(DOC_DIR, f"compress_video{vi}.gif"), fps=12)
        print(f"    Saved compress_video{vi}.gif")


# ──────────────────────────────────────────────────────────────
# 3. Reconstruction quality on real videos
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mse_and_recon(vae, video_dir, num_videos=50, max_frames=32):
    print(f"\n=== Evaluating autoencoder MSE on {num_videos} real videos ===")
    font = get_font(12)
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))[:num_videos]

    all_mse = []
    # Collect diverse samples for reconstruction demo
    demo_pairs = []  # (video_01, recon_np, mse, name)

    for idx, path in enumerate(paths):
        video_01 = load_video(path, max_frames=max_frames)
        T = video_01.shape[0]
        if T < 2:
            continue

        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)
        mean, _, _ = vae.encoder(video_t, mask4d, train=False)
        recon = vae.decoder(mean, mask4d, train=False)
        recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)

        mse = float(np.mean((video_01 - recon_np) ** 2))
        all_mse.append(mse)

        # Keep first 3 diverse videos for demo
        if len(demo_pairs) < 3:
            name = os.path.basename(path)[:40]
            demo_pairs.append((video_01, recon_np, mse, name))

        if idx % 10 == 0:
            print(f"  [{idx+1}/{len(paths)}] MSE={mse:.6f}")

    mean_mse = float(np.mean(all_mse))
    std_mse = float(np.std(all_mse))
    print(f"\n  Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f} over {len(all_mse)} videos")

    # Save reconstruction demos as side-by-side GIFs
    for i, (orig, recon, mse, name) in enumerate(demo_pairs):
        labeled_orig = label_video(orig, "Original", font)
        labeled_recon = label_video(recon, f"Reconstruction (MSE={mse:.4f})", font)
        composite = hstack_videos([labeled_orig, labeled_recon], pad=2)
        save_gif(composite, os.path.join(DOC_DIR, f"recon_video{i}.gif"), fps=12)
        print(f"  Saved recon_video{i}.gif ({name})")

    with open(os.path.join(DOC_DIR, "eval_results.txt"), "w") as f:
        f.write(f"Mean MSE: {mean_mse:.6f}\n")
        f.write(f"Std MSE: {std_mse:.6f}\n")
        f.write(f"Num videos: {len(all_mse)}\n")

    return mean_mse


def main():
    os.makedirs(DOC_DIR, exist_ok=True)

    vae = load_vae()
    dit = load_dit()

    generate_videos(dit, vae, num_latent=8)

    compress_diverse_videos(vae, "/mnt/t9/videos/videos1", max_frames=32)

    evaluate_mse_and_recon(vae, "/mnt/t9/videos/videos1",
                           num_videos=50, max_frames=32)

    print("\n=== All done! ===")
    for fn in sorted(os.listdir(DOC_DIR)):
        sz = os.path.getsize(os.path.join(DOC_DIR, fn))
        print(f"  {fn} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
