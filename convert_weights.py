"""
Convert JAX/Orbax checkpoints to PyTorch state dicts.

Handles both the VideoVAE and VideoDiT models.
JAX Linear kernels are (in, out) -> PyTorch weight is (out, in), so we transpose.
JAX Conv3d kernels are (T, H, W, C_in, C_out) -> PyTorch weight is (C_out, C_in, T, H, W).
JAX LayerNorm 'scale' -> PyTorch 'weight'.

Usage:
    python convert_weights.py --model vae --jax_checkpoint /path/to/vae_ckpt --output vae_pytorch.pt
    python convert_weights.py --model dit --jax_checkpoint /path/to/dit_ckpt --output dit_pytorch.pt
"""

import argparse
import sys
import os
import numpy as np
import torch

# Add the JAX source to path for model loading
sys.path.insert(0, "/projects/video-VAE/diffusion")


def load_jax_vae_state(checkpoint_path: str) -> dict:
    """Load JAX VAE checkpoint and return flattened param dict."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax
    import orbax.checkpoint as ocp
    from autoencoder import VideoVAE

    model = VideoVAE(
        height=256, width=256, channels=3, patch_size=16,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8,
        qkv_features=512, max_temporal_len=64,
        spatial_compression_rate=8, unembedding_upsample_rate=4,
        rngs=nnx.Rngs(2),
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
    restored = ocp.StandardCheckpointer().restore(checkpoint_path, abstract_state)
    nnx.update(model, restored["model"])

    # Extract all params
    state = nnx.state(model)
    flat = jax.tree_util.tree_leaves_with_path(state)
    params = {}
    for path, leaf in flat:
        key_parts = []
        for k in path:
            s = str(k)
            # Clean up the path: remove brackets and quotes
            s = s.replace("['", "").replace("']", "").replace("[", "").replace("]", "")
            if s == ".value" or s == "value" or s == "raw_value":
                continue
            key_parts.append(s)
        key = ".".join(key_parts)
        if key:
            params[key] = np.array(leaf)
    return params


def load_jax_dit_state(checkpoint_path: str) -> dict:
    """Load JAX DiT checkpoint and return flattened param dict."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import orbax.checkpoint as ocp
    from diffusion_model import VideoDiT

    model = VideoDiT(
        hw=256, residual_dim=1024, compressed_channel_dim=96, depth=30,
        mlp_dim=2048, num_heads=8, qkv_features=1024, max_temporal_len=64,
        rngs=nnx.Rngs(0),
    )

    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
    }
    restored = ocp.StandardCheckpointer().restore(checkpoint_path, abstract_state)
    nnx.update(model, restored["model"])

    state = nnx.state(model)
    flat = jax.tree_util.tree_leaves_with_path(state)
    params = {}
    for path, leaf in flat:
        key_parts = []
        for k in path:
            s = str(k)
            s = s.replace("['", "").replace("']", "").replace("[", "").replace("]", "")
            if s == ".value" or s == "value" or s == "raw_value":
                continue
            key_parts.append(s)
        key = ".".join(key_parts)
        if key:
            params[key] = np.array(leaf)
    return params


def convert_attention_params(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a single Attention module's parameters."""
    pt_state = {}

    # input_norm: scale -> weight, bias -> bias
    pt_state[f"{pt_prefix}.input_norm.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.input_norm.scale"])
    pt_state[f"{pt_prefix}.input_norm.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.input_norm.bias"])

    # qkv_projection: kernel (in, out) -> weight (out, in)
    pt_state[f"{pt_prefix}.qkv_projection.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.qkv_projection.kernel"].T.copy())
    pt_state[f"{pt_prefix}.qkv_projection.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.qkv_projection.bias"])

    # out_projection
    pt_state[f"{pt_prefix}.out_projection.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.out_projection.kernel"].T.copy())
    pt_state[f"{pt_prefix}.out_projection.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.out_projection.bias"])

    # q_norm, k_norm: scale -> weight (no bias)
    pt_state[f"{pt_prefix}.q_norm.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.q_norm.scale"])
    pt_state[f"{pt_prefix}.k_norm.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.k_norm.scale"])

    # ROPE: cos_cached, sin_cached (buffers, not params - but we store them)
    # These are recomputed in PyTorch, so we skip them
    return pt_state


def convert_mlp_params(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a single MLP module's parameters."""
    pt_state = {}

    # norm: scale -> weight, bias -> bias
    pt_state[f"{pt_prefix}.norm.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.norm.scale"])
    pt_state[f"{pt_prefix}.norm.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.norm.bias"])

    # linear1, linear2: kernel transpose
    pt_state[f"{pt_prefix}.linear1.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.linear1.kernel"].T.copy())
    pt_state[f"{pt_prefix}.linear1.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.linear1.bias"])
    pt_state[f"{pt_prefix}.linear2.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.linear2.kernel"].T.copy())
    pt_state[f"{pt_prefix}.linear2.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.linear2.bias"])

    return pt_state


def convert_factored_attention(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a FactoredAttention block."""
    pt_state = {}
    pt_state.update(convert_attention_params(
        jax_params, f"{jax_prefix}.SpatialAttention", f"{pt_prefix}.SpatialAttention"))
    pt_state.update(convert_mlp_params(
        jax_params, f"{jax_prefix}.SpatialMLP", f"{pt_prefix}.SpatialMLP"))
    pt_state.update(convert_attention_params(
        jax_params, f"{jax_prefix}.TemporalAttention", f"{pt_prefix}.TemporalAttention"))
    pt_state.update(convert_mlp_params(
        jax_params, f"{jax_prefix}.TemporalMLP", f"{pt_prefix}.TemporalMLP"))
    return pt_state


def convert_linear(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a linear layer."""
    pt_state = {}
    pt_state[f"{pt_prefix}.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.kernel"].T.copy())
    pt_state[f"{pt_prefix}.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.bias"])
    return pt_state


def convert_conv3d(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a Conv3d layer. JAX: (T,H,W,Cin,Cout) -> PyTorch: (Cout,Cin,T,H,W)"""
    pt_state = {}
    jax_kernel = jax_params[f"{jax_prefix}.kernel"]
    # JAX conv kernel: (T, H, W, C_in, C_out) -> PyTorch: (C_out, C_in, T, H, W)
    pt_kernel = np.transpose(jax_kernel, (4, 3, 0, 1, 2))
    pt_state[f"{pt_prefix}.weight"] = torch.tensor(pt_kernel.copy())
    pt_state[f"{pt_prefix}.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.bias"])
    return pt_state


def convert_conv_transpose3d(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a ConvTranspose3d layer.
    JAX/Flax ConvTranspose kernel shape: (T, H, W, in_features, out_features)
    PyTorch ConvTranspose3d weight shape: (in_channels, out_channels, T, H, W)

    IMPORTANT: Flax's lax.conv_transpose internally flips the kernel along spatial dims,
    while PyTorch's ConvTranspose3d does not. To compensate, we flip the kernel.
    """
    pt_state = {}
    jax_kernel = jax_params[f"{jax_prefix}.kernel"]
    # Transpose to PyTorch layout: (in, out, T, H, W)
    pt_kernel = np.transpose(jax_kernel, (3, 4, 0, 1, 2))
    # Flip spatial dimensions to match Flax's internal kernel flip
    pt_kernel = pt_kernel[:, :, ::-1, ::-1, ::-1]
    pt_state[f"{pt_prefix}.weight"] = torch.tensor(pt_kernel.copy())
    pt_state[f"{pt_prefix}.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.bias"])
    return pt_state


def convert_groupnorm(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a GroupNorm layer."""
    pt_state = {}
    pt_state[f"{pt_prefix}.weight"] = torch.tensor(
        jax_params[f"{jax_prefix}.scale"])
    pt_state[f"{pt_prefix}.bias"] = torch.tensor(
        jax_params[f"{jax_prefix}.bias"])
    return pt_state


def convert_convblock3d(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a ConvBlock3D (conv + groupnorm)."""
    pt_state = {}
    pt_state.update(convert_conv3d(jax_params, f"{jax_prefix}.conv", f"{pt_prefix}.conv"))
    pt_state.update(convert_groupnorm(jax_params, f"{jax_prefix}.norm", f"{pt_prefix}.norm"))
    return pt_state


def convert_downblock3d(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert a DownBlock3D."""
    pt_state = {}
    pt_state.update(convert_convblock3d(jax_params, f"{jax_prefix}.conv1", f"{pt_prefix}.conv1"))
    pt_state.update(convert_convblock3d(jax_params, f"{jax_prefix}.conv2", f"{pt_prefix}.conv2"))
    return pt_state


def convert_upblock3d(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert an UpBlock3D."""
    pt_state = {}
    pt_state.update(convert_conv_transpose3d(
        jax_params, f"{jax_prefix}.upsample", f"{pt_prefix}.upsample"))
    pt_state.update(convert_convblock3d(jax_params, f"{jax_prefix}.conv1", f"{pt_prefix}.conv1"))
    pt_state.update(convert_convblock3d(jax_params, f"{jax_prefix}.conv2", f"{pt_prefix}.conv2"))
    return pt_state


def convert_unet(jax_params: dict, jax_prefix: str, pt_prefix: str) -> dict:
    """Convert the full UNet."""
    pt_state = {}

    # patch_mixer
    pt_state.update(convert_conv3d(
        jax_params, f"{jax_prefix}.patch_mixer", f"{pt_prefix}.patch_mixer"))

    # encoders
    for i in range(3):
        pt_state.update(convert_downblock3d(
            jax_params, f"{jax_prefix}.encoders.{i}",
            f"{pt_prefix}.encoders.{i}"))

    # bottleneck
    pt_state.update(convert_convblock3d(
        jax_params, f"{jax_prefix}.bottleneck1", f"{pt_prefix}.bottleneck1"))
    pt_state.update(convert_convblock3d(
        jax_params, f"{jax_prefix}.bottleneck2", f"{pt_prefix}.bottleneck2"))

    # decoders
    for i in range(3):
        pt_state.update(convert_upblock3d(
            jax_params, f"{jax_prefix}.decoders.{i}",
            f"{pt_prefix}.decoders.{i}"))

    # final_conv
    pt_state.update(convert_conv3d(
        jax_params, f"{jax_prefix}.final_conv", f"{pt_prefix}.final_conv"))

    return pt_state


def convert_vae(jax_params: dict) -> dict:
    """Convert the full VideoVAE."""
    pt_state = {}

    # fill_token
    pt_state["fill_token"] = torch.tensor(jax_params["fill_token"])

    # --- Encoder ---
    # patch_embedding
    pt_state.update(convert_linear(
        jax_params, "encoder.patch_embedding.linear",
        "encoder.patch_embedding.linear"))
    pt_state["encoder.patch_embedding.norm.weight"] = torch.tensor(
        jax_params["encoder.patch_embedding.norm.scale"])
    pt_state["encoder.patch_embedding.norm.bias"] = torch.tensor(
        jax_params["encoder.patch_embedding.norm.bias"])

    # spatial_compression, variance_estimator, selection layers
    pt_state.update(convert_linear(
        jax_params, "encoder.spatial_compression",
        "encoder.spatial_compression"))
    pt_state.update(convert_linear(
        jax_params, "encoder.variance_estimator",
        "encoder.variance_estimator"))
    pt_state.update(convert_linear(
        jax_params, "encoder.selection_layer1",
        "encoder.selection_layer1"))
    pt_state.update(convert_linear(
        jax_params, "encoder.selection_layer2",
        "encoder.selection_layer2"))

    # encoder layers (9 FactoredAttention blocks)
    for i in range(9):
        pt_state.update(convert_factored_attention(
            jax_params, f"encoder.layers.{i}", f"encoder.layers.{i}"))

    # --- Decoder ---
    # spatial_decompression
    pt_state.update(convert_linear(
        jax_params, "decoder.spatial_decompression",
        "decoder.spatial_decompression"))

    # patch_unembedding
    pt_state.update(convert_linear(
        jax_params, "decoder.patch_unembedding.linear",
        "decoder.patch_unembedding.linear"))
    pt_state.update(convert_linear(
        jax_params, "decoder.patch_unembedding.upsample",
        "decoder.patch_unembedding.upsample"))
    pt_state.update(convert_linear(
        jax_params, "decoder.patch_unembedding.downsample",
        "decoder.patch_unembedding.downsample"))

    # decoder layers (12 FactoredAttention blocks)
    for i in range(12):
        pt_state.update(convert_factored_attention(
            jax_params, f"decoder.layers.{i}", f"decoder.layers.{i}"))

    # UNet
    pt_state.update(convert_unet(
        jax_params, "decoder.unet", "decoder.unet"))

    return pt_state


def convert_dit(jax_params: dict) -> dict:
    """Convert the full VideoDiT."""
    pt_state = {}

    # Top-level linear layers
    pt_state.update(convert_linear(jax_params, "timestep_proj", "timestep_proj"))
    pt_state.update(convert_linear(jax_params, "up_proj", "up_proj"))
    pt_state.update(convert_linear(jax_params, "down_proj", "down_proj"))
    pt_state.update(convert_linear(jax_params, "spacing_pred1", "spacing_pred1"))
    pt_state.update(convert_linear(jax_params, "spacing_pred2", "spacing_pred2"))

    # 30 FactoredAttention layers
    for i in range(30):
        pt_state.update(convert_factored_attention(
            jax_params, f"layers.{i}", f"layers.{i}"))

    return pt_state


def main():
    parser = argparse.ArgumentParser(description="Convert JAX checkpoints to PyTorch")
    parser.add_argument("--model", choices=["vae", "dit"], required=True)
    parser.add_argument("--jax_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading JAX {args.model.upper()} checkpoint from {args.jax_checkpoint}...")

    if args.model == "vae":
        jax_params = load_jax_vae_state(args.jax_checkpoint)
        print(f"Loaded {len(jax_params)} JAX parameters")
        pt_state = convert_vae(jax_params)
    else:
        jax_params = load_jax_dit_state(args.jax_checkpoint)
        print(f"Loaded {len(jax_params)} JAX parameters")
        pt_state = convert_dit(jax_params)

    print(f"Converted to {len(pt_state)} PyTorch parameters")

    # Save
    torch.save(pt_state, args.output)
    print(f"Saved PyTorch state dict to {args.output}")

    # Verify by loading into PyTorch model
    print("Verifying load into PyTorch model...")
    # Use importlib to explicitly load from our PyTorch directory
    import importlib.util
    pt_dir = os.path.dirname(os.path.abspath(__file__))

    if args.model == "vae":
        spec = importlib.util.spec_from_file_location("pt_autoencoder", os.path.join(pt_dir, "autoencoder.py"))
        pt_mod = importlib.util.module_from_spec(spec)
        # Need to load layers and unet first
        for dep_name in ["layers", "unet"]:
            dep_spec = importlib.util.spec_from_file_location(dep_name, os.path.join(pt_dir, f"{dep_name}.py"))
            dep_mod = importlib.util.module_from_spec(dep_spec)
            sys.modules[dep_name] = dep_mod
            dep_spec.loader.exec_module(dep_mod)
        sys.modules["pt_autoencoder"] = pt_mod
        spec.loader.exec_module(pt_mod)
        pt_model = pt_mod.VideoVAE()
        missing, unexpected = pt_model.load_state_dict(pt_state, strict=False)
        print(f"Missing keys ({len(missing)}): {missing[:10]}...")
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")
    else:
        spec = importlib.util.spec_from_file_location("pt_diffusion", os.path.join(pt_dir, "diffusion_model.py"))
        pt_mod = importlib.util.module_from_spec(spec)
        for dep_name in ["layers"]:
            dep_spec = importlib.util.spec_from_file_location(dep_name, os.path.join(pt_dir, f"{dep_name}.py"))
            dep_mod = importlib.util.module_from_spec(dep_spec)
            sys.modules[dep_name] = dep_mod
            dep_spec.loader.exec_module(dep_mod)
        sys.modules["pt_diffusion"] = pt_mod
        spec.loader.exec_module(pt_mod)
        pt_model = pt_mod.VideoDiT(depth=30)
        missing, unexpected = pt_model.load_state_dict(pt_state, strict=False)
        print(f"Missing keys ({len(missing)}): {missing[:10]}...")
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

    print("Done!")


if __name__ == "__main__":
    main()
