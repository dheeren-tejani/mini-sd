import os
from dataclasses import dataclass, field
from typing import List, Tuple


def _path(env_var: str, default: str) -> str:
    """
    Read a file path from env var, always returning a plain str.
    Protects against accidental tuple wrapping like: path = ("file.pt",)
    """
    raw = os.getenv(env_var, default)
    if isinstance(raw, (tuple, list)):   # unwrap tuple if someone made that mistake
        raw = raw[0]
    return str(raw).strip().strip("'\"")  # also strip accidental quote chars


@dataclass
class ModelConfig:
    # VAE
    vae_image_size: int = 512
    vae_in_channels: int = 3
    vae_latent_dim: int = 4
    vae_latent_size: int = 64
    vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    vae_beta: float = 0.000001

    # CLIP
    clip_vocab_size: int = 49408
    clip_embed_dim: int = 512
    clip_num_layers: int = 12
    clip_num_heads: int = 8
    clip_mlp_ratio: int = 4
    clip_max_seq_length: int = 77
    clip_dropout: float = 0.1
    clip_image_size: int = 256
    clip_patch_size: int = 16
    clip_vision_layers: int = 12

    # UNet
    unet_image_size: int = 64
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_model_channels: int = 192
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: Tuple[int, ...] = (16, 8)
    unet_channel_mult: Tuple[int, ...] = (1, 2, 2, 4)
    unet_dropout: float = 0.1
    unet_num_heads: int = 3
    unet_context_dim: int = 512
    unet_use_checkpoint: bool = False

    # Flow Matching
    use_flow_matching: bool = True
    flow_sigma_min: float = 0.0
    flow_inference_steps: int = 20
    num_diffusion_steps: int = 1000


@dataclass
class ServerConfig:
    # ── Checkpoint paths — EDIT THESE DEFAULTS ────────────────────────
    vae_path: str       = field(default_factory=lambda: _path("VAE_PATH",       "./models/vae/vae_final.pt"))
    clip_path: str      = field(default_factory=lambda: _path("CLIP_PATH",      "./models/clip/clip_final.pt"))
    unet_path: str      = field(default_factory=lambda: _path("UNET_PATH",      "./models/unet/unet_final.pt"))
    tokenizer_path: str = field(default_factory=lambda: _path("TOKENIZER_PATH", "./models/tokenizer/tokenizer.pt"))

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
        ]
    )

    # Inference limits
    max_inference_steps: int = 50
    min_inference_steps: int = 1
    max_cfg_scale: float     = 20.0
    min_cfg_scale: float     = 1.0
    output_image_size: int   = 512


MODEL_CONFIG  = ModelConfig()
SERVER_CONFIG = ServerConfig()