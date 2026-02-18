import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import os
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from einops import rearrange
import time

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION (Must match training)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedConfig:
    # VAE
    vae_image_size: int = 512
    vae_in_channels: int = 3
    vae_latent_dim: int = 4
    vae_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512])
    
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

    # UNET
    unet_image_size: int = 64
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_model_channels: int = 192
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: Tuple[int] = (16, 8)
    unet_channel_mult: Tuple[int] = (1, 2, 2, 4)
    unet_dropout: float = 0.1
    unet_num_heads: int = 3
    unet_context_dim: int = 512
    unet_use_checkpoint: bool = False # Not needed for inference

    # INFERENCE
    num_diffusion_steps: int = 1000
    
config = UnifiedConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER CLASSES (Tokenizer & Scheduler)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTokenizer:
    def __init__(self, vocab_path: str, max_length: int = 77):
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
        # Load vocab
        if os.path.exists(vocab_path):
            t_data = torch.load(vocab_path, map_location="cpu")
            self.word2idx = t_data["word2idx"]
            self.idx2word = t_data["idx2word"]
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {vocab_path}")

    def encode(self, text: str) -> torch.Tensor:
        words = re.findall(r'\b\w+\b', text.lower())
        ids = [self.word2idx["<SOS>"]]
        for word in words[:self.max_length - 2]:
            ids.append(self.word2idx.get(word, self.word2idx["<UNK>"]))
        ids.append(self.word2idx["<EOS>"])
        ids += [self.word2idx["<PAD>"]] * (self.max_length - len(ids))
        return torch.tensor(ids[:self.max_length], dtype=torch.long)

class FlowMatchingScheduler:
    def __init__(self, num_train_steps: int = 1000):
        self.num_train_steps = num_train_steps

    def sample_prev_timestep(self, model_output, timestep, sample, num_inference_steps=20):
        """Euler integration step for Flow Matching"""
        dt = 1.0 / num_inference_steps
        prev_sample = sample + model_output * dt
        return prev_sample

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS (Copied from Notebook)
# ═══════════════════════════════════════════════════════════════════════════════

# --- VAE BLOCKS ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=4, hidden_dims=[64, 128, 256], out_channels=3):
        super().__init__()
        hidden_dims = list(reversed(hidden_dims))
        self.decoder_input = nn.Conv2d(latent_dim, hidden_dims[0], 3, padding=1)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]), nn.LeakyReLU(0.2)
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1), nn.Tanh()
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return self.final_layer(x)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder not strictly needed for inference, but needed to load state_dict without errors
        # (Mocking encoder to match keys if necessary, but full definition is safer)
        self.encoder = nn.Sequential(
             nn.Conv2d(config.vae_in_channels, 64, 3, 2, 1), # Simplified for definition
             # ... Full definition would go here, but we only strictly need decoder for inference
        ) 
        # To avoid complexity, we define the minimal structure to load the decoder weights
        # NOTE: For this script, we assume strict=False when loading VAE if we skip Encoder
        self.decoder = Decoder(latent_dim=config.vae_latent_dim, hidden_dims=config.vae_hidden_dims, out_channels=config.vae_in_channels)

    def decode(self, z):
        return self.decoder(z)

# --- CLIP BLOCKS ---
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(config.clip_vocab_size, config.clip_embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.clip_max_seq_length, config.clip_embed_dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(config.clip_embed_dim, config.clip_num_heads, config.clip_mlp_ratio, config.clip_dropout)
            for _ in range(config.clip_num_layers)
        ])
        self.norm = nn.LayerNorm(config.clip_embed_dim)

    def forward(self, text_ids):
        seq_len = text_ids.shape[1]
        x = self.token_embedding(text_ids) + self.pos_embedding[:, :seq_len, :]
        for block in self.transformer: x = block(x)
        x = self.norm(x)
        # Take SOS token (first token)
        return F.normalize(x[:, 0, :], dim=-1)

# --- UNET BLOCKS ---
class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
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

class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(channels),
                'attn1': MultiHeadAttention(channels, num_heads),
                'norm2': nn.LayerNorm(channels),
                'attn2': CrossAttention(channels, context_dim, num_heads),
                'norm3': nn.LayerNorm(channels),
                'mlp': nn.Sequential(nn.Linear(channels, channels*4), nn.GELU(), nn.Linear(channels*4, channels))
            })
        ])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
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
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
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

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_emb_dim = config.unet_model_channels * 4
        self.time_embed = nn.Sequential(
            TimestepEmbedding(config.unet_model_channels),
            nn.Linear(config.unet_model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.conv_in = nn.Conv2d(config.unet_in_channels, config.unet_model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList([])
        channels = [config.unet_model_channels]
        now_channels = config.unet_model_channels
        
        for level, mult in enumerate(config.unet_channel_mult):
            out_channels = config.unet_model_channels * mult
            for _ in range(config.unet_num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResBlock(now_channels, out_channels, time_emb_dim, config.unet_dropout),
                    SpatialTransformer(out_channels, config.unet_context_dim, config.unet_num_heads)
                    if config.unet_image_size // (2 ** level) in config.unet_attention_resolutions else None
                ]))
                now_channels = out_channels
                channels.append(now_channels)
            if level != len(config.unet_channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([
                    nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1), None
                ]))
                channels.append(now_channels)

        self.mid_block1 = ResBlock(now_channels, now_channels, time_emb_dim, config.unet_dropout)
        self.mid_attn = SpatialTransformer(now_channels, config.unet_context_dim, config.unet_num_heads)
        self.mid_block2 = ResBlock(now_channels, now_channels, time_emb_dim, config.unet_dropout)

        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(reversed(config.unet_channel_mult)):
            out_channels = config.unet_model_channels * mult
            for i in range(config.unet_num_res_blocks + 1):
                self.up_blocks.append(nn.ModuleList([
                    ResBlock(now_channels + channels.pop(), out_channels, time_emb_dim, config.unet_dropout),
                    SpatialTransformer(out_channels, config.unet_context_dim, config.unet_num_heads)
                    if config.unet_image_size // (2 ** (len(config.unet_channel_mult) - 1 - level)) in config.unet_attention_resolutions else None
                ]))
                now_channels = out_channels
            if level != len(config.unet_channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([
                    nn.ConvTranspose2d(now_channels, now_channels, 4, stride=2, padding=1), None
                ]))
        
        self.out = nn.Sequential(nn.GroupNorm(32, now_channels), nn.SiLU(), nn.Conv2d(now_channels, config.unet_out_channels, 3, padding=1))

    def forward(self, x, timesteps, context):
        t_emb = self.time_embed(timesteps)
        h = self.conv_in(x)
        skips = [h]
        for block, attn in self.down_blocks:
            if isinstance(block, nn.Conv2d): h = block(h)
            else:
                h = block(h, t_emb)
                if attn is not None: h = attn(h, context)
            skips.append(h)
            
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)
        
        for block, attn in self.up_blocks:
            if isinstance(block, nn.ConvTranspose2d): h = block(h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                if attn is not None: h = attn(h, context)
        return self.out(h)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class StableDiffusionPipeline:
    def __init__(self, tokenizer_path, vae_path, clip_path, unet_path):
        print("Loading models...")
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        
        # Load VAE
        # Note: We use strict=False because we might be missing the encoder keys if the VAE class
        # definition above is slightly simplified or if we only care about the decoder.
        self.vae = VAE().to(device)
        self.load_weights(self.vae, vae_path, strict=False) 
        self.vae.eval()

        # Load CLIP
        # We instantiate the full CLIP model structure implicitly or explicitly. 
        # Here we only need the Text Encoder. 
        self.text_encoder = TextEncoder().to(device)
        # To load, we likely need to load the full CLIP checkpoint and extract text_encoder keys
        self.load_clip_text_encoder(clip_path)
        self.text_encoder.eval()

        # Load UNet
        self.unet = UNet().to(device)
        self.load_weights(self.unet, unet_path)
        self.unet.eval()
        
        self.scheduler = FlowMatchingScheduler(config.num_diffusion_steps)
        print("✓ Pipeline Ready")

    def load_weights(self, model, path, strict=True):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Handle potential prefix issues (e.g. if saved with DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."): new_state_dict[k[7:]] = v
            else: new_state_dict[k] = v
            
        try:
            model.load_state_dict(new_state_dict, strict=strict)
        except Exception as e:
            print(f"⚠️ Warning loading {path}: {e}")

    def load_clip_text_encoder(self, path):
        """Extracts text_encoder keys from the full CLIP checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        text_encoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith("text_encoder."):
                text_encoder_dict[k.replace("text_encoder.", "")] = v
        
        self.text_encoder.load_state_dict(text_encoder_dict)

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 20, guidance_scale: float = 7.5, seed: int = 42):
        if seed is not None:
            torch.manual_seed(seed)
            
        # 1. Text Embeddings
        text_ids = self.tokenizer.encode(prompt).unsqueeze(0).to(device)
        text_emb = self.text_encoder(text_ids).unsqueeze(1) # [1, 1, 512]
        
        # Classifier Free Guidance
        uncond_ids = self.tokenizer.encode("").unsqueeze(0).to(device)
        uncond_emb = self.text_encoder(uncond_ids).unsqueeze(1)
        
        # Concatenate for batch processing [uncond, text]
        text_embeddings = torch.cat([uncond_emb, text_emb])
        
        # 2. Initial Noise (Latents)
        latents = torch.randn(
            1, config.unet_in_channels, 
            config.unet_image_size, config.unet_image_size
        ).to(device)
        
        # 3. Denoising Loop (Flow Matching)
        timesteps = torch.linspace(0, config.num_diffusion_steps - 1, steps).long().to(device)
        
        print(f"Generating: '{prompt}'")
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            t_input = torch.tensor([t] * 2, device=device)
            
            # Predict velocity
            velocity_pred = self.unet(latent_model_input, t_input, text_embeddings)
            
            # Perform Guidance
            vel_uncond, vel_text = velocity_pred.chunk(2)
            velocity = vel_uncond + guidance_scale * (vel_text - vel_uncond)
            
            # Step (Euler)
            latents = self.scheduler.sample_prev_timestep(velocity, t, latents, steps)

        # 4. Decode to Image
        image = self.vae.decode(latents)
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        
        return image.squeeze(0)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    unet_model_step = 36
    
    # Path setup
    paths = {
        "tokenizer": os.path.join('./models/tokenizer/', "tokenizer.pt"),
        "vae": os.path.join('./models/vae/', "vae_final.pt"),
        "clip": './models/clip/clip_final.pt',
        "unet": f'./models/unet/unet_step_0{unet_model_step}000.pt'
    }
    
    # Verify files exist
    if all(os.path.exists(p) for p in paths.values()):
        # Initialize Pipeline
        pipe = StableDiffusionPipeline(
            paths["tokenizer"], paths["vae"], paths["clip"], paths["unet"]
        )

        while True:
            # Run Inference
            startTime = time.time()
            prompt = input("Enter your prompt : ")

            if prompt == 'exit':
                break
            
            result_tensor = pipe.generate(prompt, steps=20, guidance_scale=7.5)
            endTime = time.time()

            # Save Result
            from torchvision.utils import save_image
            save_image(result_tensor, f"output_{prompt}_step_{unet_model_step}k.png")
            print(f"✓ Image saved to output_{prompt}_step_{unet_model_step}k.png")
            print(f"Took {endTime - startTime:.2f}s in generation")
    else:
        print("❌ Error: Some model files were not found. Check your paths in the 'MAIN EXECUTION' section.")
        for k, v in paths.items():
            print(f"  {k}: {v} -> {'Found' if os.path.exists(v) else 'MISSING'}")