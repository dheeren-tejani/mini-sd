import math
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import MODEL_CONFIG as cfg


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    """
    Word-level tokenizer with special tokens: <PAD>=0  <UNK>=1  <SOS>=2  <EOS>=3
    Vocabulary is loaded from a .pt file saved during training.
    """

    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_built = False

    def encode(self, text: str) -> torch.Tensor:
        words = re.findall(r'\b\w+\b', text.lower())
        ids = [self.word2idx["<SOS>"]]
        for word in words[: self.max_length - 2]:
            ids.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        ids.append(self.word2idx["<EOS>"])
        ids += [self.word2idx["<PAD>"]] * (self.max_length - len(ids))
        return torch.tensor(ids[: self.max_length], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        words = []
        for i in ids.tolist():
            w = self.idx2word.get(i, "<UNK>")
            if w in ("<EOS>", "<PAD>"):
                break
            if w not in ("<SOS>", "<PAD>"):
                words.append(w)
        return " ".join(words)


# ══════════════════════════════════════════════════════════════════════════════
# Flow Matching Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class FlowMatchingScheduler:
    """
    Linear flow matching: x_t = (1-t)*noise + t*data
    Euler integration: x_{t+dt} = x_t + v * dt
    """

    def __init__(self, num_train_steps: int = 1000, sigma_min: float = 0.0):
        self.num_train_steps = num_train_steps

    def euler_step(self, velocity: torch.Tensor, sample: torch.Tensor, num_inference_steps: int) -> torch.Tensor:
        """Single Euler step — matches inference.py's sample_prev_timestep."""
        dt = 1.0 / num_inference_steps
        return sample + velocity * dt


# ══════════════════════════════════════════════════════════════════════════════
# VAE
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """VAE residual block — matches inference.py exactly."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act  = nn.LeakyReLU(0.2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Decoder(nn.Module):
    """VAE decoder — layer names must match checkpoint exactly."""
    def __init__(self, latent_dim=4, hidden_dims=[64, 128, 256], out_channels=3):
        super().__init__()
        hidden_dims = list(reversed(hidden_dims))
        self.decoder_input = nn.Conv2d(latent_dim, hidden_dims[0], 3, padding=1)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return self.final_layer(x)


class VAE(nn.Module):
    """
    VAE for inference.
    Only the decoder is needed; the encoder stub is kept so that
    load_state_dict(strict=False) handles encoder keys gracefully.
    """
    def __init__(self):
        super().__init__()
        # Minimal encoder stub — only here so strict=False doesn't complain about
        # completely missing sub-modules; actual inference only uses decoder.
        self.encoder = nn.Sequential(
            nn.Conv2d(cfg.vae_in_channels, 64, 3, 2, 1),
        )
        self.decoder = Decoder(
            latent_dim=cfg.vae_latent_dim,
            hidden_dims=cfg.vae_hidden_dims,
            out_channels=cfg.vae_in_channels,
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════════════
# CLIP — Text Encoder only (image encoder not needed at inference)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Self-attention used inside CLIP transformer blocks."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = embed_dim * mlp_ratio
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """
    CLIP text encoder — layer names match checkpoint keys exactly.
    Note: uses self.norm (NOT ln_final) to match inference.py.
    Output: L2-normalised [SOS] token embedding, shape (B, embed_dim).
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.clip_vocab_size, cfg.clip_embed_dim)
        self.pos_embedding   = nn.Parameter(torch.randn(1, cfg.clip_max_seq_length, cfg.clip_embed_dim))
        self.transformer     = nn.ModuleList([
            TransformerBlock(cfg.clip_embed_dim, cfg.clip_num_heads, cfg.clip_mlp_ratio, cfg.clip_dropout)
            for _ in range(cfg.clip_num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.clip_embed_dim)   # ← must be 'norm', not 'ln_final'

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        seq_len = text_ids.shape[1]
        x = self.token_embedding(text_ids) + self.pos_embedding[:, :seq_len, :]
        for block in self.transformer:
            x = block(x)
        x = self.norm(x)
        return F.normalize(x[:, 0, :], dim=-1)   # take SOS token


class CLIP(nn.Module):
    """
    Full CLIP model wrapper.
    Only text_encoder is used at inference; image_encoder stub kept for
    load_state_dict compatibility.
    """
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        # Minimal image_encoder stub so unexpected/missing key warnings are minimised
        # (the checkpoint may contain image_encoder.norm.* keys)
        self.image_encoder = nn.ModuleDict({
            "norm": nn.LayerNorm(cfg.clip_embed_dim),
        })


# ══════════════════════════════════════════════════════════════════════════════
# UNet
# ══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding — matches inference.py exactly."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim  = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = query_dim // num_heads
        self.to_q      = nn.Linear(query_dim, query_dim)
        self.to_k      = nn.Linear(context_dim, query_dim)
        self.to_v      = nn.Linear(context_dim, query_dim)
        self.to_out    = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out  = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)


class SpatialTransformer(nn.Module):
    """UNet spatial attention block — matches inference.py exactly."""
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.norm    = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(channels),
                'attn1': MultiHeadAttention(channels, num_heads),
                'norm2': nn.LayerNorm(channels),
                'attn2': CrossAttention(channels, context_dim, num_heads),
                'norm3': nn.LayerNorm(channels),
                'mlp':   nn.Sequential(
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels),
                ),
            })
        ])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = x + block['attn1'](block['norm1'](x))
            x = x + block['attn2'](block['norm2'](x), context)
            x = x + block['mlp'](block['norm3'](x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x_in + self.proj_out(x)


class ResBlock(nn.Module):
    """
    UNet residual block.
    IMPORTANT: time projection is named 'time_emb' (not 'time_proj') to match
    the saved checkpoint keys.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(32, in_channels)
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))  # ← 'time_emb'
        self.norm2    = nn.GroupNorm(32, out_channels)
        self.dropout  = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip     = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act      = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    """
    Flow Matching UNet with cross-attention conditioning.
    Architecture exactly mirrors inference.py to guarantee state dict compatibility.

    Down path uses strided Conv2d (not ConvTranspose) for downsampling.
    Up path uses ConvTranspose2d for upsampling.
    """

    def __init__(self):
        super().__init__()
        time_emb_dim = cfg.unet_model_channels * 4

        # ── Time embedding ─────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            TimestepEmbedding(cfg.unet_model_channels),
            nn.Linear(cfg.unet_model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv_in = nn.Conv2d(cfg.unet_in_channels, cfg.unet_model_channels, 3, padding=1)

        # ── Down blocks ────────────────────────────────────────────────
        self.down_blocks = nn.ModuleList()
        channels     = [cfg.unet_model_channels]
        now_channels = cfg.unet_model_channels

        for level, mult in enumerate(cfg.unet_channel_mult):
            out_ch = cfg.unet_model_channels * mult
            for _ in range(cfg.unet_num_res_blocks):
                use_attn = (cfg.unet_image_size // (2 ** level)) in cfg.unet_attention_resolutions
                self.down_blocks.append(nn.ModuleList([
                    ResBlock(now_channels, out_ch, time_emb_dim, cfg.unet_dropout),
                    SpatialTransformer(out_ch, cfg.unet_context_dim, cfg.unet_num_heads) if use_attn else None,
                ]))
                now_channels = out_ch
                channels.append(now_channels)
            if level != len(cfg.unet_channel_mult) - 1:
                # Strided conv downsampling (matches inference.py)
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1),
                    None,
                ]))
                channels.append(now_channels)

        # ── Mid block ──────────────────────────────────────────────────
        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim, cfg.unet_dropout)
        self.mid_attn   = SpatialTransformer(now_channels, cfg.unet_context_dim, cfg.unet_num_heads)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim, cfg.unet_dropout)

        # ── Up blocks ──────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(cfg.unet_channel_mult)):
            out_ch = cfg.unet_model_channels * mult
            for i in range(cfg.unet_num_res_blocks + 1):
                res_level = len(cfg.unet_channel_mult) - 1 - level
                use_attn  = (cfg.unet_image_size // (2 ** res_level)) in cfg.unet_attention_resolutions
                self.up_blocks.append(nn.ModuleList([
                    ResBlock(now_channels + channels.pop(), out_ch, time_emb_dim, cfg.unet_dropout),
                    SpatialTransformer(out_ch, cfg.unet_context_dim, cfg.unet_num_heads) if use_attn else None,
                ]))
                now_channels = out_ch
            if level != len(cfg.unet_channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1),
                    None,
                ]))

        # ── Output ─────────────────────────────────────────────────────
        self.out = nn.Sequential(
            nn.GroupNorm(32, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, cfg.unet_out_channels, 3, padding=1),
        )

    def forward(
        self,
        x:         torch.Tensor,   # (B, C, H, W)  noisy latent
        timesteps: torch.Tensor,   # (B,)           integer timestep
        context:   torch.Tensor,   # (B, seq, dim)  text conditioning
    ) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        h     = self.conv_in(x)
        skips = [h]

        for block, attn in self.down_blocks:
            if isinstance(block, nn.Conv2d) and block.stride[0] == 2:
                h = block(h)
            else:
                h = block(h, t_emb)
                if attn is not None:
                    h = attn(h, context)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        for block, attn in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d):
                h = block(h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                if attn is not None:
                    h = attn(h, context)

        return self.out(h)