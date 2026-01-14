import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import numpy as np  # <--- Added this
from dataclasses import dataclass
from typing import List, Tuple, Optional
from einops import rearrange

# ===========================
# 1. CONFIGS
# ===========================
@dataclass
class VAEConfig:
    image_size: int = 128
    latent_size: int = 16
    in_channels: int = 3
    latent_channels: int = 4
    hidden_dims: List[int] = None
    def __post_init__(self):
        if self.hidden_dims is None: self.hidden_dims = [64, 128, 256, 512]

@dataclass
class CLIPConfig:
    vocab_size: int = 10000
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: int = 4
    max_seq_length: int = 77
    dropout: float = 0.1
    image_size: int = 128
    patch_size: int = 16
    vision_layers: int = 6

@dataclass
class UNetConfig:
    image_size: int = 16
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 192
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (8, 4, 2)
    channel_mult: Tuple[int] = (1, 2, 3, 4)
    dropout: float = 0.1
    num_heads: int = 8
    context_dim: int = 512
    use_checkpoint: bool = False

# ===========================
# 2. TOKENIZER
# ===========================
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 10000, max_length: int = 77):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode(self, text: str) -> torch.Tensor:
        words = re.findall(r'\b\w+\b', text.lower())
        ids = [self.word2idx["<SOS>"]]
        for word in words[:self.max_length - 2]:
            ids.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        ids.append(self.word2idx["<EOS>"])
        ids += [self.word2idx["<PAD>"]] * (self.max_length - len(ids))
        return torch.tensor(ids[:self.max_length], dtype=torch.long)

# ===========================
# 3. HELPER MODULES
# ===========================
class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None: attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.to_out(out)

# ===========================
# 4. VAE COMPONENTS
# ===========================
class ResidualBlockVAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1 if in_channels < 8 else 8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list, latent_channels: int):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        layers = []
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            layers.append(ResidualBlockVAE(prev_dim, h_dim))
            layers.append(nn.Conv2d(h_dim, h_dim, 3, stride=2, padding=1))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_channels, 1)
        self.fc_logvar = nn.Conv2d(hidden_dims[-1], latent_channels, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_channels: int, hidden_dims: list, out_channels: int):
        super().__init__()
        self.initial_conv = nn.Conv2d(latent_channels, hidden_dims[-1], 3, padding=1)
        hidden_dims = list(reversed(hidden_dims))
        layers = []
        prev_dim = hidden_dims[0]
        for h_dim in hidden_dims:
            layers.append(ResidualBlockVAE(prev_dim, h_dim))
            layers.append(nn.ConvTranspose2d(h_dim, h_dim, 4, stride=2, padding=1))
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.GroupNorm(8, hidden_dims[-1]), nn.SiLU(), nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1))

    def forward(self, z):
        z = self.initial_conv(z)
        h = self.decoder(z)
        return self.out(h)

class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.in_channels, config.hidden_dims, config.latent_channels)
        self.decoder = Decoder(config.latent_channels, config.hidden_dims, config.in_channels)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# ===========================
# 5. CLIP COMPONENTS
# ===========================
class TextEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_length, config.embed_dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, text_ids):
        # 1. Embeddings
        x = self.token_embedding(text_ids) + self.pos_embedding[:, :text_ids.shape[1]]
        
        # 2. Transformer Loop
        for block in self.transformer: 
            x = block(x)
            
        # 3. Take CLS token
        text_features = self.norm(x)[:, 0, :]
        
        return F.normalize(text_features, dim=-1)

class ImageEncoder(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.patch_size = config.patch_size
        patch_dim = 3 * config.patch_size * config.patch_size
        self.patch_embed = nn.Linear(patch_dim, config.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.embed_dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.vision_layers)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, images):
        B = images.shape[0]
        x = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        for block in self.transformer: x = block(x)
        return self.norm(x)[:, 0, :]

class CLIP(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.text_encoder = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)
        # ADDED THIS LINE TO MATCH TRAINING SCRIPT
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def text_encoder_forward(self, text_ids):
        return self.text_encoder(text_ids)

# ===========================
# 6. UNET COMPONENTS
# ===========================
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1, use_checkpoint: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, context):
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.mlp(self.norm3(x))

class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([CrossAttentionBlock(channels, context_dim, num_heads)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks: x = block(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x_in + self.proj_out(x)

class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        time_emb_dim = config.model_channels * 4
        self.time_embed = nn.Sequential(TimestepEmbedding(config.model_channels), nn.Linear(config.model_channels, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))
        self.conv_in = nn.Conv2d(config.in_channels, config.model_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        ch = [config.model_channels]
        now_ch = config.model_channels
        
        for level, mult in enumerate(config.channel_mult):
            out_ch = config.model_channels * mult
            for _ in range(config.num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResBlock(now_ch, out_ch, time_emb_dim, config.dropout),
                    SpatialTransformer(out_ch, config.context_dim, config.num_heads) if config.image_size // (2**level) in config.attention_resolutions else None
                ]))
                now_ch = out_ch
                ch.append(now_ch)
            if level != len(config.channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(now_ch, now_ch, 3, stride=2, padding=1), None]))
                ch.append(now_ch)

        self.mid_block1 = ResBlock(now_ch, now_ch, time_emb_dim, config.dropout)
        self.mid_attn = SpatialTransformer(now_ch, config.context_dim, config.num_heads)
        self.mid_block2 = ResBlock(now_ch, now_ch, time_emb_dim, config.dropout)

        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(reversed(config.channel_mult)):
            out_ch = config.model_channels * mult
            for _ in range(config.num_res_blocks + 1):
                self.up_blocks.append(nn.ModuleList([
                    ResBlock(now_ch + ch.pop(), out_ch, time_emb_dim, config.dropout),
                    SpatialTransformer(out_ch, config.context_dim, config.num_heads) if config.image_size // (2**(len(config.channel_mult)-1-level)) in config.attention_resolutions else None
                ]))
                now_ch = out_ch
            if level != len(config.channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([nn.ConvTranspose2d(now_ch, now_ch, 4, stride=2, padding=1), None]))

        self.out = nn.Sequential(nn.GroupNorm(32, now_ch), nn.SiLU(), nn.Conv2d(now_ch, config.out_channels, 3, padding=1))

    def forward(self, x, timesteps, context):
        t_emb = self.time_embed(timesteps)
        h = self.conv_in(x)
        skips = [h]
        for block, attn in self.down_blocks:
            if isinstance(block, nn.Conv2d): h = block(h)
            else:
                h = block(h, t_emb)
                if attn: h = attn(h, context)
            skips.append(h)
        
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        for block, attn in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d): h = block(h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                if attn: h = attn(h, context)
        return self.out(h)

# ===========================
# 7. INFERENCE HELPER
# ===========================
class DDIMScheduler:
    def __init__(self, num_train_steps=1000, num_inference_steps=50):
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(num_train_steps - 1, 0, num_inference_steps, dtype=torch.long)
        betas = torch.linspace(0.0001, 0.02, num_train_steps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def sample_prev_timestep(self, model_output, timestep, sample, eta=0.0):
        prev_timestep = timestep - self.num_train_steps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        return torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

class InferenceWrapper:
    def __init__(self, unet, vae, clip, tokenizer, config, device):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    @torch.no_grad()
    def sample(self, prompts, num_inference_steps=50, guidance_scale=7.5):
        self.unet.eval()
        self.vae.eval()
        self.clip.eval()
        
        batch_size = len(prompts)
        text_ids = torch.stack([self.tokenizer.encode(p) for p in prompts]).to(self.device)
        # Note: Calling text_encoder_forward because of new CLIP structure
        text_embeddings = self.clip.text_encoder_forward(text_ids).unsqueeze(1)
        
        uncond_embeddings = torch.zeros_like(text_embeddings)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn(batch_size, 4, 16, 16).to(self.device)
        scheduler = DDIMScheduler(1000, num_inference_steps)
        
        for t in scheduler.timesteps.to(self.device):
            latent_input = torch.cat([latents] * 2)
            t_input = torch.tensor([t] * (batch_size * 2), device=self.device)
            noise_pred = self.unet(latent_input, t_input, text_embeddings)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.sample_prev_timestep(noise_pred, t, latents)
            
        return (self.vae.decode(latents) + 1) / 2