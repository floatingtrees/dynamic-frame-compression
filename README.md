# Dynamic Frame Compression

PyTorch port of a JAX/Flax video generation and compression system combining a **Video VAE** (Variational Autoencoder) with a **Video DiT** (Diffusion Transformer). The system supports:

- **Video generation** from noise using flow-matching diffusion
- **Video compression** via learned frame selection + spatial compression
- **Video decompression** back to pixel space

All inference runs in **bfloat16** for efficient GPU usage.

---

## Results

### Autoencoder Reconstruction Quality

The VAE encodes 256x256 video frames into a compressed latent space (8x spatial compression: 768 -> 96 channels) and reconstructs them. Evaluated on 50 real videos from the training set:

| Metric | Value |
|--------|-------|
| Mean MSE | 0.0886 |
| Mean MAE | 0.2751 |

Below: original frames (top), VAE reconstruction (middle), and 5x-amplified difference (bottom).

![Reconstruction comparison](docs/reconstruction_comparison.png)

### Frame-Budget Compression

The model learns which frames are most important and can compress a video by keeping only a subset of frames. Unselected frames are replaced with a learned fill token, then the decoder reconstructs the full video.

Here we compress a 32-frame aerial coastal video, keeping the top-K most important frames (by the encoder's learned selection scores), and compare reconstruction quality:

| Frames Kept | MSE | Compression |
|-------------|---------|-------------|
| 32 (all) | 0.1224 | 1x (baseline) |
| 16 | 0.1246 | 2x |
| 8 | 0.1248 | 4x |
| 4 | 0.1279 | 8x |
| 2 | 0.1285 | 16x |
| 1 | 0.1190 | 32x |
| Standard (p>0.5) | 0.1276 | ~11x (3 frames) |

![Compression comparison](docs/compression_comparison.png)

*Each row shows 8 evenly-spaced frames from the 32-frame video. Top row is the original; subsequent rows show reconstruction quality as fewer frames are kept.*

### Generated Video

32-frame video generated from noise using the DiT (100 Euler steps) + VAE decoder. Every 4th frame shown:

![Generated video frames](docs/generated_grid.png)

---

## Architecture

### Video VAE (Autoencoder)
- **Encoder**: PatchEmbedding (16x16 patches) -> 9 FactoredAttention layers -> spatial compression (768->96) + variance estimation + frame selection
- **Decoder**: Spatial decompression (96->768) -> 12 FactoredAttention layers -> PatchUnEmbedding + 3D UNet refinement
- **FactoredAttention**: Temporal attention+MLP then spatial attention+MLP, with RoPE and QK-norm
- **Frame selection**: Learned per-frame importance scores for dynamic temporal compression
- Resolution: 256x256, patch size 16x16 (256 spatial tokens), up to 64 temporal frames

### Video DiT (Diffusion Transformer)
- 30 FactoredAttention layers with residual_dim=1024
- Flow matching with Euler integration (continuous timesteps in [0,1])
- Timestep conditioning via zero-initialized linear projection
- Predicts both velocity field and frame spacing scores
- Operates on compressed VAE latents (96-dim)

### 3D UNet (Decoder Refinement)
- 3-level encoder-decoder with skip connections
- 3D convolutions (temporal_kernel=3) for spatiotemporal coherence
- GroupNorm + SiLU, zero-initialized final conv

---

## File Structure

```
dynamic-frame-compression/
├── layers.py                 # Attention, MLP, FactoredAttention, RoPE, PatchEmbed
├── unet.py                   # 3D UNet: ConvBlock3D, DownBlock3D, UpBlock3D
├── autoencoder.py            # VideoVAE: Encoder, Decoder, compress/decompress
├── diffusion_model.py        # VideoDiT + Euler sampling
├── convert_weights.py        # JAX Orbax -> PyTorch weight conversion
├── generate.py               # Video generation (DiT + VAE decode)
├── compress.py               # Video compression (VAE encode)
├── decompress.py             # Video decompression (VAE decode)
├── evaluate.py               # Evaluation & documentation image generation
├── test_jax_vs_pytorch.py    # JAX vs PyTorch correctness tests
└── docs/                     # Generated documentation images
```

---

## Setup

### Prerequisites
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- Python 3.10+

### Environment

```bash
python3 -m venv venv && source venv/bin/activate

# PyTorch (adjust cu126 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Dependencies
pip install einops imageio imageio-ffmpeg pillow

# For JAX comparison tests (optional):
pip install "jax[cuda12]" "flax==0.10.4" optax orbax-checkpoint beartype jaxtyping
```

### Weight Conversion

Convert JAX/Orbax checkpoints to PyTorch:

```bash
# VAE (encoder + decoder + UNet)
python convert_weights.py \
    --model vae \
    --jax_checkpoint /mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_290000 \
    --output vae_pytorch.pt

# DiT (diffusion transformer)
python convert_weights.py \
    --model dit \
    --jax_checkpoint /mnt/t9/DiT_longterm_saves/midpoint_save/checkpoint_step_250000_master \
    --output dit_pytorch.pt
```

The conversion handles:
- Linear weight transposition (`kernel` (in,out) -> `weight` (out,in))
- Conv3d kernel reordering (JAX `THWIO` -> PyTorch `OITHW`)
- **ConvTranspose3d kernel flip** (Flax internally flips spatial dims; PyTorch does not)
- LayerNorm/GroupNorm `scale` -> `weight`, epsilon 1e-6
- ROPE cos/sin buffers are recomputed deterministically

---

## Usage

### Generate Video

```bash
# 32 frames, 100 Euler steps, bfloat16
python generate.py --num_frames 32 --num_steps 100 --seed 256 --output generated.mp4

# Save individual frames as PNGs
python generate.py --num_frames 16 --num_steps 100 --output video.mp4 --save_frames frames/
```

### Compress a Video

```bash
python compress.py --input video.mp4 --output compressed.pt --max_frames 32
```

Output `.pt` contains: `compressed` (latent), `selection_indices`, `compression_mask`, metadata.

### Decompress

```bash
python decompress.py --input compressed.pt --output reconstructed.mp4
```

### Evaluate & Generate Documentation

```bash
python evaluate.py
# Outputs to docs/: generated_grid.png, compression_comparison.png, reconstruction_comparison.png
```

---

## Correctness Verification

All model outputs match JAX within 1e-3 (tested with TF32 disabled):

```bash
NVIDIA_TF32_OVERRIDE=0 python test_jax_vs_pytorch.py
```

```
=== Testing VAE Encoder ===
  [PASS] encoder mean:       max_diff=2.12e-05
  [PASS] encoder variance:   max_diff=9.91e-07
  [PASS] encoder selection:  max_diff=5.66e-06

=== Testing VAE Decoder ===
  [PASS] decoder output:     max_diff=4.17e-07

=== Testing DiT Forward ===
  [PASS] dit latent:         max_diff=1.62e-05
  [PASS] dit spacing:        max_diff=9.54e-06

=== Testing DiT Sampling (100 steps) ===
  [PASS] sampling output:    max_diff=8.85e-04

=== Testing Full Pipeline (DiT -> VAE Decode) ===
  [PASS] video output:       max_diff=1.32e-04

All tests PASSED!
```

Key conversion details that were required for correctness:
- **LayerNorm epsilon**: Flax defaults to `1e-6`, PyTorch defaults to `1e-5`
- **ConvTranspose3d kernel flip**: Flax's `lax.conv_transpose` flips the kernel internally; the conversion compensates
- **TF32 matmul**: JAX enables TF32 by default on GPU, reducing float32 matmul precision

---

## Model Hyperparameters

| | VAE | DiT |
|---|---|---|
| **Depth** | 9 enc / 12 dec | 30 |
| **Hidden dim** | 768 | 1024 |
| **MLP dim** | 1536 | 2048 |
| **Heads** | 8 | 8 |
| **QKV features** | 512 | 1024 |
| **Spatial patches** | 256 | 256 |
| **Max temporal** | 64 | 64 |
| **Compressed dim** | 96 | 96 (input) |

## Source Checkpoints

| Model | Path | Format |
|-------|------|--------|
| VAE | `/mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_290000` | Orbax |
| DiT | `/mnt/t9/DiT_longterm_saves/midpoint_save/checkpoint_step_250000_master` | Orbax |
| JAX source | `/projects/video-VAE/diffusion/` | Python |

## License

Apache License 2.0
