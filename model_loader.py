"""
Utilities for loading VAE and DiT models, with automatic download from HuggingFace.
"""

import os
import torch
from huggingface_hub import hf_hub_download

from autoencoder import VideoVAE
from diffusion_model import VideoDiT

HF_REPO = "floatingtrees2/dynamic-frame-compression"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "dynamic-frame-compression")


def _resolve_path(filename: str, local_path: str | None) -> str:
    """Return a local path to the weight file, downloading from HF if needed."""
    if local_path and os.path.isfile(local_path):
        return local_path

    # Check default locations
    for candidate in [filename, os.path.join(DEFAULT_CACHE, filename)]:
        if os.path.isfile(candidate):
            return candidate

    # Download from HuggingFace
    print(f"Downloading {filename} from {HF_REPO}...")
    path = hf_hub_download(repo_id=HF_REPO, filename=filename,
                           cache_dir=DEFAULT_CACHE)
    return path


def load_vae(checkpoint: str | None = None, device: str = "cuda") -> VideoVAE:
    """Load VideoVAE, downloading weights from HuggingFace if not found locally."""
    path = _resolve_path("vae_pytorch.pt", checkpoint)
    model = VideoVAE()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def load_dit(checkpoint: str | None = None, device: str = "cuda") -> VideoDiT:
    """Load VideoDiT, downloading weights from HuggingFace if not found locally."""
    path = _resolve_path("dit_pytorch.pt", checkpoint)
    model = VideoDiT(depth=30)
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model
