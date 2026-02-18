import io
import base64
import os
import threading
import time
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from architecture import CLIP, VAE, UNet, TextEncoder, SimpleTokenizer, FlowMatchingScheduler
from config import MODEL_CONFIG as cfg, SERVER_CONFIG as srv
from logger import log


def _fmt(s: float) -> str:
    return f"{s*1000:.0f} ms" if s < 1 else f"{s:.2f} s"


def _validate_path(label: str, path) -> None:
    if not isinstance(path, str):
        raise TypeError(
            f"\n  ✗ {label} path is {type(path).__name__}, not str: {path!r}\n"
            f"  → Check config.py — no trailing commas!\n"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\n  ✗ {label} checkpoint not found: {path!r}\n"
            f"  → Edit config.py or set the env var.\n"
        )


def _extract_state(ckpt: dict, label: str) -> dict:
    """
    Pull state dict exactly like inference.py:
      checkpoint["model_state_dict"] if present, else the dict itself.
    EMA weights are intentionally NOT used — inference.py never uses them.
    """
    if not isinstance(ckpt, dict):
        return ckpt
    if "model_state_dict" in ckpt:
        log.info(f"    → using model_state_dict")
        return ckpt["model_state_dict"]
    log.info(f"    → treating as raw state dict")
    return ckpt


class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
        log.info(f"Device: {self.device}{gpu}")

        self.vae:          Optional[VAE]             = None
        self.text_encoder: Optional[TextEncoder]     = None
        self.unet:         Optional[UNet]            = None
        self.tokenizer:    Optional[SimpleTokenizer] = None
        self.scheduler = FlowMatchingScheduler(num_train_steps=cfg.num_diffusion_steps)

        self._ready = False
        self._lock  = threading.Lock()

    # ── Loaders ───────────────────────────────────────────────────────

    def _load_raw(self, label: str, path: str) -> dict:
        _validate_path(label, path)
        t0 = time.perf_counter()
        log.info(f"    Reading: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # noqa
        log.info(f"    read in {_fmt(time.perf_counter()-t0)}")
        return ckpt

    def _apply(self, model: torch.nn.Module, state: dict, label: str):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"    ⚠ {len(missing)} missing keys   e.g. {missing[:3]}")
        if unexpected:
            log.warning(f"    ⚠ {len(unexpected)} unexpected keys  e.g. {unexpected[:3]}")
        if not missing and not unexpected:
            log.info(f"    ✓ weights loaded perfectly (no missing/unexpected keys)")

    def _param_count(self, m: torch.nn.Module) -> str:
        return f"{sum(p.numel() for p in m.parameters())/1e6:.1f}M params"

    def _load_tokenizer(self):
        t0 = time.perf_counter()
        log.info("  [Tokenizer]")
        _validate_path("Tokenizer", srv.tokenizer_path)
        log.info(f"    Reading: {srv.tokenizer_path}")
        data = torch.load(srv.tokenizer_path, map_location="cpu", weights_only=False)  # noqa
        if not isinstance(data, dict) or "word2idx" not in data:
            raise ValueError(f"Bad tokenizer file — expected dict with 'word2idx'/'idx2word'")
        self.tokenizer = SimpleTokenizer(vocab_size=cfg.clip_vocab_size, max_length=cfg.clip_max_seq_length)
        self.tokenizer.word2idx    = data["word2idx"]
        self.tokenizer.idx2word    = data["idx2word"]
        self.tokenizer.vocab_built = True
        log.info(f"    ✓ vocab={len(self.tokenizer.word2idx):,}  |  {_fmt(time.perf_counter()-t0)}")

    def _load_vae(self):
        t0 = time.perf_counter()
        log.info("  [VAE]")
        ckpt  = self._load_raw("VAE", srv.vae_path)
        state = _extract_state(ckpt, "VAE")
        self.vae = VAE()
        self._apply(self.vae, state, "VAE")
        self.vae.to(self.device).eval()
        log.info(f"    ✓ {self._param_count(self.vae)}  |  {_fmt(time.perf_counter()-t0)}")

    def _load_clip(self):
        """
        Load only the text encoder from the CLIP checkpoint.
        Mirrors inference.py's load_clip_text_encoder() exactly:
        strips the 'text_encoder.' prefix from keys and loads into TextEncoder.
        """
        t0 = time.perf_counter()
        log.info("  [CLIP — text encoder only]")
        ckpt  = self._load_raw("CLIP", srv.clip_path)
        state = _extract_state(ckpt, "CLIP")

        # Extract text_encoder.* keys and strip the prefix
        te_state = {
            k.replace("text_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("text_encoder.")
        }
        if not te_state:
            raise ValueError(
                "No 'text_encoder.*' keys found in CLIP checkpoint.\n"
                "Keys in checkpoint: " + str(list(state.keys())[:10])
            )
        log.info(f"    → {len(te_state)} text_encoder keys extracted")

        self.text_encoder = TextEncoder()
        self._apply(self.text_encoder, te_state, "TextEncoder")
        self.text_encoder.to(self.device).eval()
        log.info(f"    ✓ {self._param_count(self.text_encoder)}  |  {_fmt(time.perf_counter()-t0)}")

    def _load_unet(self):
        t0 = time.perf_counter()
        log.info("  [UNet]")
        ckpt  = self._load_raw("UNet", srv.unet_path)
        state = _extract_state(ckpt, "UNet")
        self.unet = UNet()
        self._apply(self.unet, state, "UNet")
        self.unet.to(self.device).eval()
        log.info(f"    ✓ {self._param_count(self.unet)}  |  {_fmt(time.perf_counter()-t0)}")

    # ── Public API ────────────────────────────────────────────────────

    def load_models(self):
        total_t0 = time.perf_counter()
        log.info("=" * 60)
        log.info("Checkpoint paths:")
        log.info(f"  VAE       : {srv.vae_path}")
        log.info(f"  CLIP      : {srv.clip_path}")
        log.info(f"  UNet      : {srv.unet_path}")
        log.info(f"  Tokenizer : {srv.tokenizer_path}")
        log.info("=" * 60)

        self._load_tokenizer()
        self._load_vae()
        self._load_clip()
        self._load_unet()

        self._ready = True
        log.info("=" * 60)
        log.info(f"✓ All models loaded in {_fmt(time.perf_counter()-total_t0)}")
        log.info("  Server ready to generate images.")
        log.info("=" * 60)

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """prompt → (1, 1, 512)  — matches inference.py's text_encoder(text_ids).unsqueeze(1)"""
        token_ids = self.tokenizer.encode(prompt).unsqueeze(0).to(self.device)   # (1, 77)
        text_feat = self.text_encoder(token_ids)                                  # (1, 512)
        return text_feat.unsqueeze(1)                                             # (1, 1, 512)

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> Image.Image:
        decoded = self.vae.decode(latent)                   # (1, 3, H, W) in [-1, 1]
        decoded = ((decoded.clamp(-1, 1) + 1) / 2).squeeze(0).cpu()
        decoded = TF.resize(decoded, [srv.output_image_size, srv.output_image_size], antialias=True)
        return TF.to_pil_image(decoded)

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 20, cfg_scale: float = 7.5, seed: int = 42) -> str:
        if not self._ready:
            raise RuntimeError("Models not loaded.")

        steps     = max(srv.min_inference_steps, min(steps,     srv.max_inference_steps))
        cfg_scale = max(srv.min_cfg_scale,       min(cfg_scale, srv.max_cfg_scale))

        with self._lock:
            wall_t0 = time.perf_counter()
            log.info("─" * 60)
            log.info(f"GENERATE  '{prompt[:80]}{'…' if len(prompt)>80 else ''}'")
            log.info(f"          steps={steps}  cfg={cfg_scale}  seed={seed}")

            # 1. Text embeddings
            t0         = time.perf_counter()
            text_emb   = self._encode_prompt(prompt)             # (1, 1, 512)
            uncond_emb = self._encode_prompt("")                  # (1, 1, 512)
            combined   = torch.cat([uncond_emb, text_emb])       # (2, 1, 512)
            log.info(f"  [1/4] Prompt encoded         {_fmt(time.perf_counter()-t0)}")

            # 2. Initial noise
            t0 = time.perf_counter()
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)
            latents = torch.randn(
                1, cfg.unet_in_channels, cfg.unet_image_size, cfg.unet_image_size,
                device=self.device,
            )
            log.info(f"  [2/4] Noise sampled  {tuple(latents.shape)}  {_fmt(time.perf_counter()-t0)}")

            # 3. Denoising loop
            t0 = time.perf_counter()
            timesteps = torch.linspace(0, cfg.num_diffusion_steps - 1, steps, device=self.device).long()

            step_times = []
            pbar = tqdm(
                enumerate(timesteps), total=steps,
                desc="  Denoising", unit="step", ncols=72,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for i, t in pbar:
                s = time.perf_counter()
                # Batch uncond + cond together for efficiency
                latent_in = torch.cat([latents, latents])            # (2, C, H, W)
                t_in      = torch.tensor([t, t], device=self.device) # (2,)

                velocity = self.unet(latent_in, t_in, combined)

                v_uncond, v_cond = velocity.chunk(2)
                v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

                latents = self.scheduler.euler_step(v_guided, latents, steps)

                dt = time.perf_counter() - s
                step_times.append(dt)
                pbar.set_postfix({"ms/step": f"{dt*1000:.0f}"}, refresh=False)

            pbar.close()
            denoise_dur = time.perf_counter() - t0
            avg_ms      = sum(step_times) / len(step_times) * 1000
            log.info(f"  [3/4] Denoising done  total={_fmt(denoise_dur)}  avg={avg_ms:.0f}ms/step  {steps/denoise_dur:.1f}steps/s")

            # 4. VAE decode
            t0 = time.perf_counter()
            pil_img = self._decode_latent(latents)
            log.info(f"  [4/4] VAE decoded → {srv.output_image_size}×{srv.output_image_size}px  {_fmt(time.perf_counter()-t0)}")

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            log.info(f"  DONE  wall={_fmt(time.perf_counter()-wall_t0)}  size={len(buf.getvalue())//1024}KB")
            log.info("─" * 60)
            return data_uri


manager = ModelManager()