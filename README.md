# Dynamic Frame Compression

PyTorch port of a JAX/Flax video generation and compression system combining a **Video VAE** (Variational Autoencoder) with a **Video DiT** (Diffusion Transformer).

- **Video compression** via learned frame selection + spatial compression (up to 256x)
- **Video generation** from noise using flow-matching diffusion with learned frame spacing

All inference runs in **bfloat16**. Both models were trained for 10 days on 32 Google TPU v6e chips.

---

## VAE Results

### Reconstruction Quality

The VAE encodes 256x256 video into a compact latent (8x spatial compression: 768→96 channels per patch) and reconstructs it. Averaged over **600 real videos**:

| Metric | Value |
|--------|-------|
| **Mean MSE** | **0.0026** |
| Std MSE | 0.0029 |

![Aerial coast](docs/recon_video0.gif)

![Beach jumping](docs/recon_video1.gif)

![Dance — high motion](docs/recon_video2.gif)

![Music video — high motion](docs/recon_video3.gif)

### Frame-Budget Compression

The encoder assigns each frame an importance score. By keeping only the top-K frames and filling the rest with a learned token, the model compresses video at arbitrary temporal ratios on top of the 8x spatial compression.

Shown: Original | All 32 frames | Top-8 | Top-4 | Top-1

![Seascape](docs/compress_video0.gif)

![Palm beach aerial](docs/compress_video1.gif)

![Dance — high motion](docs/compress_video2.gif)

![Music video — high motion](docs/compress_video3.gif)

*High-motion videos (dance, music video) degrade more with fewer frames, as expected — temporal information is harder to reconstruct from a single keyframe.*

---

## DiT Results

### Video Generation with Frame Gap Prediction

The DiT generates compressed latent frames **and** predicts the temporal spacing between them. Each latent frame is placed at its predicted position; the VAE fills gaps with a learned token. Output length = `sum(predicted_gaps)`.

Example: 8 latent frames with gaps `[2,6,6,3,5,2,2,4]` → 31 output frames.

![4x4 generation grid](docs/generated_grid.gif)

*16 seeds, 8 latent frames each. Labels show seed and output frame count (28–33 frames). The model generates diverse scenes: animals, water, landscapes, people, close-ups.*

---

## Architecture

### Video VAE
- **Encoder**: 16x16 PatchEmbed → 9 FactoredAttention layers → spatial compression (768→96) + frame selection head
- **Decoder**: Spatial decompression (96→768) → 12 FactoredAttention layers → PatchUnEmbed + 3D UNet refinement
- **FactoredAttention**: Temporal attention+MLP, then spatial attention+MLP, with RoPE and QK-norm
- **Frame selection**: Learned per-frame importance; unselected frames replaced by a learned fill token

### Video DiT
- 30 FactoredAttention layers, residual_dim=1024
- Flow matching with Euler integration (continuous timesteps [0,1])
- **Dual-head output**: velocity field (denoising) + frame gaps (temporal spacing)
- Gaps determine output length — no trailing blank frames

### 3D UNet (Decoder Refinement)
- 3-level encoder-decoder with skip connections and 3D convolutions

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install einops imageio imageio-ffmpeg pillow

# Optional: JAX comparison tests
pip install "jax[cuda12]" "flax==0.10.4" optax orbax-checkpoint beartype jaxtyping
```

### Weight Conversion

```bash
python convert_weights.py --model vae \
    --jax_checkpoint /mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_290000 \
    --output vae_pytorch.pt

python convert_weights.py --model dit \
    --jax_checkpoint /mnt/t9/DiT_longterm_saves/midpoint_save/checkpoint_step_250000_master \
    --output dit_pytorch.pt
```

---

## Usage

### Generate Video

```bash
python generate.py --num_latent_frames 8 --num_steps 100 --seed 256 --output generated.mp4
```

### Compress / Decompress

```bash
python compress.py --input video.mp4 --output compressed.pt --max_frames 32
python decompress.py --input compressed.pt --output reconstructed.mp4
```

### Evaluate

```bash
python evaluate.py  # regenerates all docs/ assets
```

---

## Correctness

All outputs match JAX within **1e-3** (TF32 disabled):

```
NVIDIA_TF32_OVERRIDE=0 python test_jax_vs_pytorch.py
```

| Test | Max Diff |
|------|----------|
| VAE Encoder | 2.1e-05 |
| VAE Decoder | 4.2e-07 |
| DiT Forward (30 layers) | 1.6e-05 |
| DiT Sampling (100 steps) | 8.9e-04 |
| Full Pipeline | 1.3e-04 |

Key details: LayerNorm/GroupNorm eps=1e-6, ConvTranspose3d kernel flip, model uses [0,1] pixel range.

---

## File Structure

```
layers.py              # Attention, MLP, FactoredAttention, RoPE, PatchEmbed
unet.py                # 3D UNet
autoencoder.py         # VideoVAE: Encoder, Decoder, compress/decompress
diffusion_model.py     # VideoDiT, Euler sampling, gaps_to_positions
convert_weights.py     # JAX Orbax → PyTorch conversion
generate.py            # Generation with frame gap prediction
compress.py            # Video compression
decompress.py          # Video decompression
evaluate.py            # Evaluation & doc generation
test_jax_vs_pytorch.py # Correctness tests
```

## License

Apache License 2.0
