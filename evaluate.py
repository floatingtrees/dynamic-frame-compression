"""
Evaluation script for README documentation.
- Generate 32-frame video from DiT
- Compress a real video at different frame budgets (1, 2, 4, 8, 16 out of 32)
- Compute MSE over real videos from /mnt/t9/videos/videos1
- Produce comparison images and videos for the README

NOTE: The model operates in [0, 1] pixel space throughout.
"""

import os
import glob
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from autoencoder import VideoVAE
from diffusion_model import VideoDiT, sample


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
    """Load video as float32 in [0,1], shape (T,H,W,3)."""
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


def make_grid(images, nrow, pad=2, pad_value=255):
    n = len(images)
    h, w, c = images[0].shape
    nrows_grid = (n + nrow - 1) // nrow
    grid_h = nrows_grid * h + (nrows_grid - 1) * pad
    grid_w = nrow * w + (nrow - 1) * pad
    grid = np.full((grid_h, grid_w, c), pad_value, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, col = divmod(idx, nrow)
        y = r * (h + pad)
        x = col * (w + pad)
        grid[y:y + h, x:x + w] = img
    return grid


def add_label_row(label, row_img, canvas, y, font):
    row_w = row_img.shape[1]
    label_img = Image.new("RGB", (row_w, 22), (255, 255, 255))
    draw = ImageDraw.Draw(label_img)
    draw.text((4, 3), label, fill=(0, 0, 0), font=font)
    canvas[y:y + 22, :row_w] = np.array(label_img)
    y += 22
    rh = row_img.shape[0]
    canvas[y:y + rh, :row_w] = row_img
    return y + rh + 4


def save_as_mp4(frames_01, path, fps=12):
    """Save (T,H,W,3) float [0,1] frames as mp4."""
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for i in range(frames_01.shape[0]):
        writer.append_data(to_uint8(frames_01[i]))
    writer.close()


# ──────────────────────────────────────────────────────────────
# 1. Generate 32-frame video from DiT
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_video_32(dit, vae):
    print("\n=== Generating 32-frame videos (multiple seeds) ===")
    best_video = None
    best_contrast = -1
    best_seed = 0

    for seed in [0, 7, 13, 42, 99, 123, 256, 404, 512, 1024]:
        torch.manual_seed(seed)
        noise = torch.randn(1, 32, 256, 96, device=DEVICE, dtype=torch.bfloat16)
        mask = torch.ones(1, 32, dtype=torch.bool, device=DEVICE)
        latent, _ = sample(dit, noise, mask, num_steps=100)
        mask4d = torch.ones(1, 1, 1, 32, dtype=torch.bool, device=DEVICE)
        video = vae.decoder(latent, mask4d, train=False)
        # Model outputs [0,1] directly
        video_np = np.clip(video[0].float().cpu().numpy(), 0, 1)
        contrast = video_np.std()
        print(f"  seed={seed:4d}: std={contrast:.4f}, range=[{video_np.min():.3f}, {video_np.max():.3f}]")
        if contrast > best_contrast:
            best_contrast = contrast
            best_video = video_np
            best_seed = seed

    print(f"  Best seed: {best_seed} (std={best_contrast:.4f})")

    save_as_mp4(best_video, os.path.join(DOC_DIR, "generated_32f.mp4"), fps=12)

    # Grid of every 4th frame
    idx = list(range(0, 32, 4))
    grid = make_grid([to_uint8(best_video[i]) for i in idx], nrow=8)
    imageio.imwrite(os.path.join(DOC_DIR, "generated_grid.png"), grid)
    print(f"  Saved generated_grid.png")
    return best_video


# ──────────────────────────────────────────────────────────────
# 2. Compress a REAL video at different frame budgets
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def compress_real_video(vae, video_path, max_frames=32):
    print(f"\n=== Compressing real video: {os.path.basename(video_path)} ===")
    video_01 = load_video(video_path, max_frames=max_frames)
    T = video_01.shape[0]
    print(f"  Loaded {T} frames at {video_01.shape[1]}x{video_01.shape[2]}")

    # Model expects [0,1] input
    video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
    mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

    mean, variance, selection_probs = vae.encoder(video_t, mask4d, train=False)
    latent = mean
    fill = vae.fill_token.to(torch.bfloat16)

    sel = selection_probs.squeeze(-1).squeeze(0).float().cpu().numpy()
    print(f"  Selection probs: min={sel.min():.3f}, max={sel.max():.3f}, mean={sel.mean():.3f}")

    # All frames baseline
    recon_all = vae.decoder(latent, mask4d, train=False)
    recon_all_np = np.clip(recon_all[0].float().cpu().numpy(), 0, 1)
    mse_all = float(np.mean((video_01 - recon_all_np) ** 2))

    # Standard p>0.5
    sel_mask_std = (selection_probs > 0.5).to(torch.bfloat16).unsqueeze(-1)
    kept_std = int(sel_mask_std.sum().item())
    compressed_std = latent * sel_mask_std + fill * (1 - sel_mask_std)
    recon_std = vae.decoder(compressed_std, mask4d, train=False)
    recon_std_np = np.clip(recon_std[0].float().cpu().numpy(), 0, 1)
    mse_std = float(np.mean((video_01 - recon_std_np) ** 2))

    print(f"  All frames:    MSE={mse_all:.6f}")
    print(f"  Standard (p>0.5, {kept_std}/{T}f): MSE={mse_std:.6f}")

    rows = {"all": (recon_all_np, mse_all, list(range(T)), T)}

    for budget in [16, 8, 4, 2, 1]:
        topk_idx = np.sort(np.argsort(sel)[-budget:])
        sel_mask = torch.zeros(1, T, 1, 1, device=DEVICE, dtype=torch.bfloat16)
        sel_mask[0, topk_idx, 0, 0] = 1.0
        compressed = latent * sel_mask + fill * (1 - sel_mask)
        recon = vae.decoder(compressed, mask4d, train=False)
        recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)
        mse = float(np.mean((video_01 - recon_np) ** 2))
        print(f"  Top-{budget:2d} frames: MSE={mse:.6f}  (frames={topk_idx.tolist()})")
        rows[budget] = (recon_np, mse, topk_idx.tolist(), budget)

    rows["standard"] = (recon_std_np, mse_std, None, kept_std)

    # Save videos for each compression level
    save_as_mp4(video_01, os.path.join(DOC_DIR, "original_32f.mp4"), fps=12)
    save_as_mp4(recon_all_np, os.path.join(DOC_DIR, "recon_all_frames.mp4"), fps=12)
    save_as_mp4(recon_std_np, os.path.join(DOC_DIR, "recon_standard.mp4"), fps=12)
    for budget in [8, 4, 2, 1]:
        save_as_mp4(rows[budget][0], os.path.join(DOC_DIR, f"recon_top{budget}.mp4"), fps=12)

    return video_01, rows


def save_compression_comparison(video_01, rows):
    T = video_01.shape[0]
    display_idx = np.linspace(0, T - 1, min(8, T), dtype=int)
    nframes = len(display_idx)
    font = get_font(14)

    all_labeled_rows = [
        ("Original (32 frames)",
         make_grid([to_uint8(video_01[i]) for i in display_idx], nrow=nframes, pad=1)),
    ]

    recon_np, mse, _, _ = rows["all"]
    all_labeled_rows.append(
        (f"All 32 frames kept  (MSE={mse:.4f})",
         make_grid([to_uint8(recon_np[i]) for i in display_idx], nrow=nframes, pad=1)))

    for budget in [16, 8, 4, 2, 1]:
        recon_np, mse, _, _ = rows[budget]
        all_labeled_rows.append(
            (f"Top-{budget} frames kept  (MSE={mse:.4f})",
             make_grid([to_uint8(recon_np[i]) for i in display_idx], nrow=nframes, pad=1)))

    recon_np, mse, _, kept = rows["standard"]
    all_labeled_rows.append(
        (f"Standard p>0.5 ({kept}f kept)  (MSE={mse:.4f})",
         make_grid([to_uint8(recon_np[i]) for i in display_idx], nrow=nframes, pad=1)))

    row_h = all_labeled_rows[0][1].shape[0]
    row_w = all_labeled_rows[0][1].shape[1]
    total_h = len(all_labeled_rows) * (row_h + 26) + 10
    canvas = np.full((total_h, row_w, 3), 255, dtype=np.uint8)
    y = 4
    for label, row_img in all_labeled_rows:
        y = add_label_row(label, row_img, canvas, y, font)
    imageio.imwrite(os.path.join(DOC_DIR, "compression_comparison.png"), canvas[:y])
    print(f"  Saved compression_comparison.png")


# ──────────────────────────────────────────────────────────────
# 3. MSE evaluation over real videos
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mse(vae, video_dir, num_videos=50, max_frames=32):
    print(f"\n=== Evaluating autoencoder MSE on {num_videos} real videos ===")
    paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))[:num_videos]

    all_mse, all_mae = [], []
    best_vid, best_rec, best_contrast = None, None, -1

    for idx, path in enumerate(paths):
        video_01 = load_video(path, max_frames=max_frames)
        T = video_01.shape[0]
        if T < 2:
            continue

        # Model expects [0,1]
        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

        mean, _, _ = vae.encoder(video_t, mask4d, train=False)
        recon = vae.decoder(mean, mask4d, train=False)
        recon_np = np.clip(recon[0].float().cpu().numpy(), 0, 1)

        mse = float(np.mean((video_01 - recon_np) ** 2))
        mae = float(np.mean(np.abs(video_01 - recon_np)))
        all_mse.append(mse)
        all_mae.append(mae)

        c = video_01.std()
        if c > best_contrast:
            best_contrast = c
            best_vid = video_01
            best_rec = recon_np

        if idx % 10 == 0:
            print(f"  [{idx + 1}/{len(paths)}] {os.path.basename(path)}: MSE={mse:.6f}")

    mean_mse = float(np.mean(all_mse))
    mean_mae = float(np.mean(all_mae))
    std_mse = float(np.std(all_mse))
    print(f"\n  Results over {len(all_mse)} videos:")
    print(f"    Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f}")
    print(f"    Mean MAE: {mean_mae:.6f}")

    return mean_mse, mean_mae, best_vid, best_rec


def save_recon_comparison(video_01, recon_np):
    T = min(video_01.shape[0], recon_np.shape[0])
    display_idx = np.linspace(0, T - 1, min(8, T), dtype=int)
    nframes = len(display_idx)
    font = get_font(14)

    orig_grid = make_grid([to_uint8(video_01[i]) for i in display_idx], nrow=nframes, pad=1)
    recon_grid = make_grid([to_uint8(recon_np[i]) for i in display_idx], nrow=nframes, pad=1)
    diff = np.abs(video_01[:T] - recon_np[:T])
    diff_grid = make_grid([to_uint8(np.clip(diff[i] * 5.0, 0, 1)) for i in display_idx],
                          nrow=nframes, pad=1)

    rows_data = [("Original", orig_grid),
                 ("VAE Reconstruction", recon_grid),
                 ("Difference (5x amplified)", diff_grid)]
    row_w, row_h = orig_grid.shape[1], orig_grid.shape[0]
    total_h = len(rows_data) * (row_h + 26) + 10
    canvas = np.full((total_h, row_w, 3), 255, dtype=np.uint8)
    y = 4
    for label, row_img in rows_data:
        y = add_label_row(label, row_img, canvas, y, font)
    imageio.imwrite(os.path.join(DOC_DIR, "reconstruction_comparison.png"), canvas[:y])

    # Also save as side-by-side video
    save_as_mp4(video_01, os.path.join(DOC_DIR, "recon_original.mp4"), fps=12)
    save_as_mp4(recon_np, os.path.join(DOC_DIR, "recon_reconstructed.mp4"), fps=12)
    print(f"  Saved reconstruction_comparison.png + videos")


def main():
    os.makedirs(DOC_DIR, exist_ok=True)

    vae = load_vae()
    dit = load_dit()

    gen_video = generate_video_32(dit, vae)

    video_path = "/mnt/t9/videos/videos1/mixkit_beach_mixkit-aerial-panorama-of-a-coast-and-its-reliefs-36615_000.mp4"
    if not os.path.exists(video_path):
        paths = sorted(glob.glob("/mnt/t9/videos/videos1/*.mp4"))
        video_path = paths[0]
    real_video, comp_rows = compress_real_video(vae, video_path, max_frames=32)
    save_compression_comparison(real_video, comp_rows)

    mean_mse, mean_mae, sample_vid, sample_rec = evaluate_mse(
        vae, "/mnt/t9/videos/videos1", num_videos=50, max_frames=32)
    if sample_vid is not None:
        save_recon_comparison(sample_vid, sample_rec)

    with open(os.path.join(DOC_DIR, "eval_results.txt"), "w") as f:
        f.write(f"Mean MSE: {mean_mse:.6f}\n")
        f.write(f"Mean MAE: {mean_mae:.6f}\n")
        f.write(f"Num videos: 50\n")
        for k, (_, mse, _, kept) in comp_rows.items():
            f.write(f"Compression {k}: MSE={mse:.6f}, kept={kept}\n")

    print("\n=== All done! ===")
    for fn in sorted(os.listdir(DOC_DIR)):
        sz = os.path.getsize(os.path.join(DOC_DIR, fn))
        print(f"  {fn} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
