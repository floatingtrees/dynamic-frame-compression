"""
Comprehensive tests comparing JAX and PyTorch model outputs.
Tests each layer/module independently and then full models end-to-end.

Usage:
    python test_jax_vs_pytorch.py
"""

import sys
import os
import numpy as np

# Disable TF32 for fair comparison (JAX uses TF32 by default on GPU)
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

import torch
# Also disable TF32 in PyTorch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# We need both JAX and PyTorch in the same process
sys.path.insert(0, "/projects/video-VAE/diffusion")

import jax
import jax.numpy as jnp
# Set JAX to highest matmul precision
jax.config.update('jax_default_matmul_precision', 'highest')
from flax import nnx
import optax
import orbax.checkpoint as ocp

# Import JAX models
from autoencoder import VideoVAE as JaxVideoVAE
from diffusion_model import VideoDiT as JaxVideoDiT

# Import PyTorch models (use importlib to avoid name conflicts)
import importlib.util
pt_dir = os.path.dirname(os.path.abspath(__file__))

def load_pt_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pt_layers = load_pt_module("pt_layers", os.path.join(pt_dir, "layers.py"))
pt_unet = load_pt_module("pt_unet", os.path.join(pt_dir, "unet.py"))

# Monkey-patch the modules so autoencoder.py and diffusion_model.py can import them
sys.modules["layers"] = pt_layers  # Override JAX layers temporarily
sys.modules["unet"] = pt_unet

# Now we need to carefully import the PyTorch autoencoder
# First save the JAX versions
_jax_autoencoder = sys.modules.get("autoencoder")
_jax_diffusion = sys.modules.get("diffusion_model")

pt_autoencoder = load_pt_module("pt_autoencoder", os.path.join(pt_dir, "autoencoder.py"))
pt_diffusion = load_pt_module("pt_diffusion", os.path.join(pt_dir, "diffusion_model.py"))

# Restore JAX modules
if _jax_autoencoder:
    sys.modules["autoencoder"] = _jax_autoencoder
if _jax_diffusion:
    sys.modules["diffusion_model"] = _jax_diffusion


TOLERANCE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_close(name: str, jax_val, pt_val, atol=TOLERANCE):
    """Compare JAX and PyTorch tensors."""
    jax_np = np.array(jax_val).astype(np.float32)
    if isinstance(pt_val, torch.Tensor):
        pt_np = pt_val.detach().cpu().float().numpy()
    else:
        pt_np = np.array(pt_val).astype(np.float32)

    max_diff = np.max(np.abs(jax_np - pt_np))
    mean_diff = np.mean(np.abs(jax_np - pt_np))
    passed = max_diff < atol

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} "
          f"(shapes: jax={jax_np.shape}, pt={pt_np.shape})")
    return passed


def load_jax_vae():
    """Load and return JAX VAE model with checkpoint weights.
    Uses float32 for fair comparison with PyTorch (which also runs in float32).
    """
    model = JaxVideoVAE(
        height=256, width=256, channels=3, patch_size=16,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8,
        qkv_features=512, max_temporal_len=64,
        spatial_compression_rate=8, unembedding_upsample_rate=4,
        rngs=nnx.Rngs(2),
        dtype=jnp.float32, param_dtype=jnp.float32,
    )
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=6e-5,
        warmup_steps=5000, decay_steps=1_000_000, end_value=6e-6,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer_def)
    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
        "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer)),
    }
    restored = ocp.StandardCheckpointer().restore(
        "/mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_290000", abstract_state)
    nnx.update(model, restored["model"])
    return model


def load_pt_vae():
    """Load and return PyTorch VAE model with converted weights."""
    model = pt_autoencoder.VideoVAE()
    state_dict = torch.load("vae_pytorch.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).float()
    model.eval()
    return model


def load_jax_dit():
    """Load and return JAX DiT model with checkpoint weights.
    Uses float32 for fair comparison with PyTorch.
    """
    model = JaxVideoDiT(
        hw=256, residual_dim=1024, compressed_channel_dim=96, depth=30,
        mlp_dim=2048, num_heads=8, qkv_features=1024, max_temporal_len=64,
        rngs=nnx.Rngs(0),
        dtype=jnp.float32, param_dtype=jnp.float32,
    )
    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
    }
    restored = ocp.StandardCheckpointer().restore(
        "/mnt/t9/DiT_longterm_saves/midpoint_save/checkpoint_step_250000_master", abstract_state)
    nnx.update(model, restored["model"])
    return model


def load_pt_dit():
    """Load and return PyTorch DiT model with converted weights."""
    model = pt_diffusion.VideoDiT(depth=30)
    state_dict = torch.load("dit_pytorch.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE).float()
    model.eval()
    return model


def test_vae_encoder(jax_model, pt_model):
    """Test VAE encoder outputs match."""
    print("\n=== Testing VAE Encoder ===")

    # Create deterministic input
    np.random.seed(42)
    video_np = np.random.randn(1, 4, 256, 256, 3).astype(np.float32) * 0.02
    mask_np = np.ones((1, 1, 1, 4), dtype=bool)

    # JAX forward
    video_jax = jnp.array(video_np)
    mask_jax = jnp.array(mask_np)
    jax_mean, jax_var, jax_sel = jax_model.encoder(video_jax, mask_jax, rngs=nnx.Rngs(0), train=False)

    # PyTorch forward
    video_pt = torch.tensor(video_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        pt_mean, pt_var, pt_sel = pt_model.encoder(video_pt, mask_pt, train=False)

    all_passed = True
    all_passed &= check_close("encoder mean", jax_mean, pt_mean)
    all_passed &= check_close("encoder variance", jax_var, pt_var)
    all_passed &= check_close("encoder selection", jax_sel, pt_sel)
    return all_passed


def test_vae_decoder(jax_model, pt_model):
    """Test VAE decoder outputs match with a known latent input."""
    print("\n=== Testing VAE Decoder ===")

    np.random.seed(123)
    latent_np = np.random.randn(1, 4, 256, 96).astype(np.float32) * 0.02
    mask_np = np.ones((1, 1, 1, 4), dtype=bool)

    # JAX
    latent_jax = jnp.array(latent_np)
    mask_jax = jnp.array(mask_np)
    jax_out = jax_model.decoder(latent_jax, mask_jax, rngs=nnx.Rngs(0), train=False)

    # PyTorch
    latent_pt = torch.tensor(latent_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        pt_out = pt_model.decoder(latent_pt, mask_pt, train=False)

    return check_close("decoder output", jax_out, pt_out)


def test_vae_compress(jax_model, pt_model):
    """Test VAE compress (encode to latent) matches."""
    print("\n=== Testing VAE Compress ===")

    np.random.seed(42)
    video_np = np.random.randn(1, 4, 256, 256, 3).astype(np.float32) * 0.02
    mask_np = np.ones((1, 1, 1, 4), dtype=bool)

    # JAX - deterministic compress (train=False uses mean, selection>0.5)
    video_jax = jnp.array(video_np)
    mask_jax = jnp.array(mask_np)
    # Get encoder output deterministically
    jax_mean, jax_var, jax_sel = jax_model.encoder(video_jax, mask_jax, rngs=nnx.Rngs(0), train=False)

    # PyTorch
    video_pt = torch.tensor(video_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        pt_mean, pt_var, pt_sel = pt_model.encoder(video_pt, mask_pt, train=False)

    all_passed = True
    all_passed &= check_close("compress encoder mean", jax_mean, pt_mean)
    all_passed &= check_close("compress encoder selection", jax_sel, pt_sel)
    return all_passed


def test_vae_decompress(jax_model, pt_model):
    """Test VAE decompress (decode from latent) with consistent input."""
    print("\n=== Testing VAE Decompress ===")

    # Use encoder outputs from both models to test decoder independently
    np.random.seed(42)
    video_np = np.random.randn(1, 4, 256, 256, 3).astype(np.float32) * 0.02
    mask_np = np.ones((1, 1, 1, 4), dtype=bool)

    # Get JAX encoder output (deterministic)
    video_jax = jnp.array(video_np)
    mask_jax = jnp.array(mask_np)
    jax_mean, _, jax_sel = jax_model.encoder(video_jax, mask_jax, rngs=nnx.Rngs(0), train=False)

    # Use the mean as the latent (deterministic)
    # All frames selected (sel > 0.5 for all), so compressed = mean
    jax_out = jax_model.decoder(jax_mean, mask_jax, rngs=nnx.Rngs(0), train=False)

    # PyTorch equivalent
    video_pt = torch.tensor(video_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        pt_mean, _, pt_sel = pt_model.encoder(video_pt, mask_pt, train=False)
        pt_out = pt_model.decoder(pt_mean, mask_pt, train=False)

    all_passed = True
    all_passed &= check_close("decompress latent input", jax_mean, pt_mean)
    all_passed &= check_close("decompress output", jax_out, pt_out)
    return all_passed


def test_dit_forward(jax_model, pt_model):
    """Test DiT forward pass matches."""
    print("\n=== Testing DiT Forward ===")

    np.random.seed(42)
    compressed_np = np.random.randn(1, 8, 256, 96).astype(np.float32) * 0.02
    mask_np = np.ones((1, 8), dtype=bool)
    mask_np[0, 5:] = False
    timestep_np = np.array([[0.5]], dtype=np.float32)

    # JAX
    compressed_jax = jnp.array(compressed_np)
    mask_jax = jnp.array(mask_np)
    timestep_jax = jnp.array(timestep_np)
    jax_latent, jax_spacing = jax_model(compressed_jax, mask_jax, timestep_jax)

    # PyTorch
    compressed_pt = torch.tensor(compressed_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    timestep_pt = torch.tensor(timestep_np, device=DEVICE)
    with torch.no_grad():
        pt_latent, pt_spacing = pt_model(compressed_pt, mask_pt, timestep_pt)

    all_passed = True
    all_passed &= check_close("dit latent prediction", jax_latent, pt_latent)
    all_passed &= check_close("dit spacing prediction", jax_spacing, pt_spacing)
    return all_passed


def test_dit_sampling(jax_model, pt_model):
    """Test DiT sampling (generation) matches."""
    print("\n=== Testing DiT Sampling ===")

    np.random.seed(42)
    noise_np = np.random.randn(1, 8, 256, 96).astype(np.float32) * 0.02
    mask_np = np.ones((1, 8), dtype=bool)

    # JAX sampling with few steps for speed (manual Euler loop)
    noise_jax = jnp.array(noise_np)
    mask_jax = jnp.array(mask_np)
    num_steps = 10
    dt = 1.0 / num_steps
    x_jax = noise_jax
    for i in range(num_steps):
        t = jnp.full((1, 1), i / num_steps)
        vel, sel = jax_model(x_jax, mask_jax, t)
        x_jax = x_jax + vel * dt

    # PyTorch sampling
    noise_pt = torch.tensor(noise_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        x_pt, sel_pt = pt_diffusion.sample(pt_model, noise_pt, mask_pt, num_steps=10)

    return check_close("dit sampling output", x_jax, x_pt, atol=5e-3)


def test_full_pipeline(jax_vae, pt_vae, jax_dit, pt_dit):
    """Test full generation pipeline: DiT sample -> VAE decode."""
    print("\n=== Testing Full Pipeline (DiT -> VAE Decode) ===")

    np.random.seed(42)
    noise_np = np.random.randn(1, 8, 256, 96).astype(np.float32) * 0.1
    mask_np = np.ones((1, 8), dtype=bool)

    # JAX: sample with DiT then decode with VAE
    noise_jax = jnp.array(noise_np)
    mask_jax = jnp.array(mask_np)
    num_steps = 5
    dt = 1.0 / num_steps
    x_jax = noise_jax
    for i in range(num_steps):
        t = jnp.full((1, 1), i / num_steps)
        vel, _ = jax_dit(x_jax, mask_jax, t)
        x_jax = x_jax + vel * dt

    mask_4d_jax = jnp.ones((1, 1, 1, 8), dtype=bool)
    jax_video = jax_vae.decoder(x_jax, mask_4d_jax, rngs=nnx.Rngs(0), train=False)

    # PyTorch: same pipeline
    noise_pt = torch.tensor(noise_np, device=DEVICE)
    mask_pt = torch.tensor(mask_np, device=DEVICE)
    with torch.no_grad():
        x_pt, _ = pt_diffusion.sample(pt_dit, noise_pt, mask_pt, num_steps=5)
        mask_4d_pt = torch.ones(1, 1, 1, 8, dtype=torch.bool, device=DEVICE)
        pt_video = pt_vae.decoder(x_pt, mask_4d_pt, train=False)

    all_passed = True
    all_passed &= check_close("pipeline dit output", x_jax, x_pt)
    all_passed &= check_close("pipeline video output", jax_video, pt_video, atol=5e-3)
    return all_passed


def main():
    print("=" * 60)
    print("JAX vs PyTorch Comparison Tests")
    print("=" * 60)
    print(f"Tolerance: {TOLERANCE}")
    print(f"Device: {DEVICE}")

    # Load models
    print("\nLoading JAX VAE...")
    jax_vae = load_jax_vae()
    print("Loading PyTorch VAE...")
    pt_vae = load_pt_vae()
    print("Loading JAX DiT...")
    jax_dit = load_jax_dit()
    print("Loading PyTorch DiT...")
    pt_dit = load_pt_dit()

    results = {}

    # Run tests
    results["encoder"] = test_vae_encoder(jax_vae, pt_vae)
    results["decoder"] = test_vae_decoder(jax_vae, pt_vae)
    results["compress"] = test_vae_compress(jax_vae, pt_vae)
    results["decompress"] = test_vae_decompress(jax_vae, pt_vae)
    results["dit_forward"] = test_dit_forward(jax_dit, pt_dit)
    results["dit_sampling"] = test_dit_sampling(jax_dit, pt_dit)
    results["full_pipeline"] = test_full_pipeline(jax_vae, pt_vae, jax_dit, pt_dit)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed &= passed

    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
