"""
Calculate the probability that two independently sampled compression masks
are identical, averaged over the video dataset.

For a single frame with selection probability p, the probability that two
independent Bernoulli samples agree is: p^2 + (1-p)^2.

For a video with T frames (independent), the probability that the entire
mask matches is: prod_t [p_t^2 + (1-p_t)^2].
"""

import glob
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image

from model_loader import load_vae

DEVICE = "cuda"


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
    if len(frames) == 0:
        return None
    return np.stack(frames).astype(np.float32) / 255.0


@torch.no_grad()
def main():
    vae = load_vae()

    video_dir = "/mnt/t9/videos/videos1"
    paths = sorted(glob.glob(f"{video_dir}/*.mp4"))[:600]

    all_mask_match_probs = []
    all_per_frame_match_probs = []
    all_sel_probs = []

    for idx, path in enumerate(paths):
        video_01 = load_video(path, max_frames=32)
        if video_01 is None or video_01.shape[0] < 2:
            continue
        T = video_01.shape[0]
        video_t = torch.tensor(video_01, device=DEVICE, dtype=torch.bfloat16).unsqueeze(0)
        mask4d = torch.ones(1, 1, 1, T, dtype=torch.bool, device=DEVICE)

        _, _, sel_probs = vae.encoder(video_t, mask4d, train=False)
        p = sel_probs.squeeze(0).squeeze(-1).float().cpu().numpy()  # (T,)

        per_frame_match = p**2 + (1 - p)**2
        log_match_prob = np.sum(np.log(per_frame_match + 1e-30))
        mask_match_prob = np.exp(log_match_prob)

        all_mask_match_probs.append(mask_match_prob)
        all_per_frame_match_probs.extend(per_frame_match.tolist())
        all_sel_probs.extend(p.tolist())

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(paths)}] "
                  f"mean P(mask match)={np.mean(all_mask_match_probs):.6e}, "
                  f"mean per-frame match={np.mean(all_per_frame_match_probs):.4f}")

    probs = np.array(all_mask_match_probs)
    per_frame = np.array(all_per_frame_match_probs)
    sel = np.array(all_sel_probs)

    print(f"\n{'='*60}")
    print(f"Results over {len(probs)} videos (32 frames each)")
    print(f"{'='*60}")

    print(f"\nSelection probability statistics:")
    print(f"  Mean:   {np.mean(sel):.4f}")
    print(f"  Median: {np.median(sel):.4f}")
    print(f"  Std:    {np.std(sel):.4f}")
    print(f"  Min:    {np.min(sel):.4f}")
    print(f"  Max:    {np.max(sel):.4f}")

    print(f"\nSelection probability distribution:")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        count = np.sum((sel >= lo) & (sel < hi))
        pct = 100 * count / len(sel)
        print(f"  [{lo:.1f}, {hi:.1f}): {count:5d} ({pct:5.1f}%)")

    print(f"\nPer-frame P(two samples agree):")
    print(f"  Mean:   {np.mean(per_frame):.4f}")
    print(f"  Median: {np.median(per_frame):.4f}")
    print(f"  Min:    {np.min(per_frame):.4f}")

    print(f"\nP(entire 32-frame mask identical) per video:")
    print(f"  Mean:   {np.mean(probs):.6e}")
    print(f"  Median: {np.median(probs):.6e}")
    print(f"  Min:    {np.min(probs):.6e}")
    print(f"  Max:    {np.max(probs):.6e}")
    print(f"  Std:    {np.std(probs):.6e}")

    per_frame_entropy = -(sel * np.log2(sel + 1e-30) + (1 - sel) * np.log2(1 - sel + 1e-30))
    print(f"\nEntropy of compression mask:")
    print(f"  Per-frame entropy: {np.mean(per_frame_entropy):.4f} bits")
    print(f"  Total mask entropy (32 frames): {np.mean(per_frame_entropy) * 32:.2f} bits")
    print(f"  Effective distinct masks: ~2^{np.mean(per_frame_entropy) * 32:.1f}")


if __name__ == "__main__":
    main()
