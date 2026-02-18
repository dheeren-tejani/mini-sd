# 🎨 Mini Stable Diffusion — Built From Scratch

> A complete, end-to-end text-to-image diffusion pipeline trained from scratch on a single T4 GPU. ~32% the parameter count of Stable Diffusion v1.5, implementing modern Flow Matching instead of DDPM.

---

## ⚠️ Honest Limitations (Read This First)

This model generates **mountains, sunsets, and landscape scenery** reasonably well. It **fails on essentially every other concept** — people, animals, objects, abstract ideas, text, you name it.

This is an unexpected outcome of our dataset choices during training. The LAION subset we used (~168K images) skewed heavily toward natural landscape photography, and our caption quality was insufficient to teach the model broader semantic understanding.

**We are starting a new project** that trains a fresh VAE, CLIP, and UNet (same architecture from scratch) on a curated, high-quality dataset of **2 million image-caption pairs** with significantly better caption diversity and quality. This repo documents v1 — the learning exercise.

---

## 📐 Architecture Overview

Mini SD is a three-stage pipeline, conceptually identical to Stable Diffusion but built entirely from scratch:

```
Text Prompt
    │
    ▼
┌─────────┐
│  CLIP   │  Text Encoder (ViT-B/32 equivalent)
│ Encoder │  Produces a 512-dim embedding
└────┬────┘
     │ conditioning (cross-attention)
     ▼
┌─────────┐      ┌─────────┐
│  UNet   │◄─────│  Noise  │  Pure Gaussian noise in latent space
│         │      └─────────┘
│ (Flow   │  Iteratively denoises over N steps
│Matching)│  using Euler integration
└────┬────┘
     │
     ▼
┌─────────┐
│   VAE   │  Decodes 64×64 latent → 512×512 RGB image
│ Decoder │
└─────────┘
     │
     ▼
512×512 PNG
```

### Component Details

| Component | Parameters | Architecture | Role |
| :--- | :--- | :--- | :--- |
| **VAE** | 3.13 M | ResNet-based encoder/decoder, 512px → 64px (8× compression), 4-channel latent | Compresses pixel space into a compact latent space for efficient diffusion |
| **CLIP** | 101.5 M | ViT-B/32 equivalent — 12-layer transformer, 512-dim embeddings, 77-token max sequence | Encodes text prompts into conditioning vectors via contrastive pretraining |
| **UNet** | 230.85 M | 192 base channels, channel multipliers (1,2,2,4), 2 ResBlocks per level, cross-attention at 16×16 and 8×8 resolutions | The diffusion "brain" — learns the velocity field to denoise latents |
| **Total** | **~335.5 M** | End-to-end pipeline | 32% the size of Stable Diffusion v1.5 (~1B total) |

### Flow Matching vs. DDPM

Unlike most tutorials that implement DDPM, this project uses **Flow Matching** (Lipman et al., 2023):

- **Forward process:** Linear interpolation — `x_t = (1-t) * noise + t * data`
- **Target:** The model learns velocity `v = data - noise`, not noise `ε`
- **Sampling:** Simple Euler integration — `x_{t+dt} = x_t + v * dt`
- **Benefit:** Straighter trajectories → high-quality samples in 20 steps vs. 50–1000 for DDPM

Classifier-Free Guidance (CFG) is implemented by concatenating unconditional and conditional embeddings in a single forward pass, then interpolating the predicted velocities.

---

## 🗂️ Repository Structure

```
mini-sd/
├── README.md
├── .gitignore
│
├── backend/
│   ├── architecture.py          # All model definitions: VAE, CLIP, UNet, Tokenizer, Scheduler
│   ├── config.py                # Unified config for model hyperparameters & server settings
│   ├── main.py                  # FastAPI server — /generate, /health, /config endpoints
│   ├── model_manager.py         # Checkpoint loading, inference pipeline, image encoding
│   ├── logger.py                # Structured logging utility
│   ├── standalone_inference.py  # Self-contained script for inference without the API server
│   ├── utils.py                 # Image ↔ base64 helpers, GPU memory info
│   └── requirements.txt         # Python dependencies
│
├── frontend/                    # Web UI (connects to FastAPI backend)
│
└── metadata/
    ├── training_script.ipynb    # Full Google Colab training notebook (VAE → CLIP → UNet)
    ├── dataset_samples.png      # Sample images from the LAION training subset
    ├── clip/                    # CLIP training metrics & sample outputs
    ├── unet/                    # UNet training metrics & generated samples
    └── vae/                     # VAE training metrics & reconstruction comparisons
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- A mid to high tier CPU (CUDA capable GPU with 4GB VRAM or more recommended for inference)
- Model checkpoints (VAE, CLIP, UNet, Tokenizer — not included in repo due to size)

### Installation

```bash
git clone https://github.com/dheeren-tejani/mini-sd
cd mini-sd/backend

pip install -r requirements.txt
```

`requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cu121` for CUDA 12.1 builds of PyTorch.

### Configure Checkpoint Paths

Edit `backend/config.py` or set environment variables:

```python
# In config.py — ServerConfig defaults:
vae_path:       "./models/vae/vae_final.pt"
clip_path:      "./models/clip/clip_final.pt"
unet_path:      "./models/unet/unet_step_017000.pt"
tokenizer_path: "./models/tokenizer/tokenizer.pt"
```

Or via environment variables:

```bash
export VAE_PATH="/path/to/vae_final.pt"
export CLIP_PATH="/path/to/clip_final.pt"
export UNET_PATH="/path/to/unet_step_017000.pt"
export TOKENIZER_PATH="/path/to/tokenizer.pt"
```

### Run the API Server

```bash
cd backend
uvicorn main:app --reload
# Server starts at http://127.0.0.1:8000

```

### Run the Frontend React Server

```bash
cd..
cd frontend
npm run dev
```

### OR

### Standalone Inference (no server)

```bash
cd backend
python standalone_inference.py
# Interactive prompt loop — type a prompt, get a saved PNG
```

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/generate` | Generate an image from a text prompt |
| `GET` | `/health` | Check if models are loaded and ready |
| `GET` | `/config` | Retrieve inference parameter ranges for the frontend |

**Example `/generate` request:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain lake at golden hour",
    "steps": 20,
    "cfg_scale": 7.5,
    "seed": 42
  }'
```

**Response:** A JSON object containing a base64-encoded PNG data-URI in the `image` field, ready to drop into an `<img src="...">` tag.

---

## 🏋️ Training

Training was done entirely on Google Colab (T4 GPU, 16GB VRAM) using the notebook at `metadata/training_script.ipynb`. Training is sequential — VAE first, then CLIP, then UNet.

### Dataset

- **Source:** [LAION Aesthetic subset (~168K images with captions)](https://huggingface.co/datasets/bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images)
- **Resolution:** 512×512 (pre-processed using [preprocess.py](./metadata/preprocess.py))

### Training Order & Key Settings

| Stage | Component | Batch Size | Effective Batch | Steps | Key Detail |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | VAE | 32 | 128 | Until convergence | Cyclical KL annealing, free bits = 0.5 |
| 2 | CLIP | 32 | 128 | Until convergence | Contrastive loss, temperature learned |
| 3 | UNet | 32 | 128 | ~17,000+ | Flow matching, CFG dropout = 10% |

All stages use:
- **Optimizer:** 8-bit AdamW (`bitsandbytes`) for VRAM efficiency
- **Scheduler:** Cosine LR with warmup
- **Mixed precision:** AMP (FP16 forward, FP32 loss)
- **Gradient clipping:** 0.5
- **EMA:** decay = 0.9999
- **GPU augmentations:** Kornia pipeline (normalization, flips, color jitter offloaded to GPU)

### Checkpoint Management

A sliding window keeps only the latest `max_checkpoints=2` step checkpoints to save Drive storage, while `_best.pt` and `_final.pt` are always preserved.

### Loss Curves

You can see the loss curves for VAE, CLIP, UNet here, [VAE Curves](./metadata/vae/vae_training_curves), [CLIP Curves](./metadata/clip/clip_training_curves), [UNet Curves](./metadata/unet/unet_training_curves).

---

## ⚙️ Model Configuration

All hyperparameters live in `backend/config.py` as two frozen dataclasses:

- **`ModelConfig`** — architecture dimensions (channels, layers, latent sizes, attention resolutions)
- **`ServerConfig`** — checkpoint paths, CORS origins, inference limits (max steps, CFG scale)

Key values at a glance:

```python
# VAE
vae_image_size = 512       # Input resolution
vae_latent_dim = 4         # Latent channels (matches UNet input)
vae_hidden_dims = [128, 256, 512]

# CLIP
clip_embed_dim = 512       # Also the UNet context dim
clip_num_layers = 12
clip_max_seq_length = 77

# UNet
unet_model_channels = 192
unet_channel_mult = (1, 2, 2, 4)
unet_attention_resolutions = (16, 8)  # Apply cross-attn at 16×16 and 8×8

# Inference
flow_inference_steps = 20   # Euler steps (20 is usually enough)
num_diffusion_steps = 1000  # Training timestep range
```

---

## 🧠 Known Issues & Design Decisions

**Why does it only work for landscapes?**
The LAION subset used for training had a strong bias toward natural scenery, and the captions were often too generic ("a beautiful image", "high quality photo") to teach meaningful text-image alignment. The model overfit to this distribution.

**Why is the UNet checkpoint at step 70,000?**
Training was done on free Colab T4 sessions with time limits. Step 70,000 represents the best checkpoint before the session expired. More training would likely improve results, but the dataset bias is the fundamental bottleneck.

**Why Flow Matching and not DDPM?**
Flow Matching converges faster, generates comparable quality in fewer inference steps, and the loss formulation is simpler. It was a deliberate design choice to learn modern techniques over the older DDPM formulation.

**EMA weights are intentionally NOT used at inference.**
The checkpoint saves both `model_state_dict` and `ema_model_state_dict`. The inference code uses `model_state_dict` exclusively, due to EMA weights being corrupted by a bug in the training script

**Samples being corrupted to pure noise.**
If you see the [VAE](./metadata/vae/samples/) and [UNet](./metadata/unet/samples/) samples, you will see pure noise in them, this is due to some bugs during their sampling phase, and they do not reflect their training behavior, it happenend due to some bug in VAE Scaling and UNet flow sampling, but we are unsure of the exact reason why it happenend, if you find the exact reason of it by analyzing the [Training Script](./metadata/training_script.ipynb), feel free to mail me at dheerennntejani@gmail.com

---

## 🔭 What's Next — v2 Plans

We are building a new version from scratch with the same architecture but addressing the root causes of v1's limitations:

- **2 million high-quality image-caption pairs** with diverse subjects (people, animals, objects, scenes, abstract concepts)
- **Rich, descriptive captions** generated or curated to carry real semantic signal
- **Longer training runs** with proper LR scheduling and evaluation
- **Better VAE** with higher latent quality (LPIPS loss component)

The goal is a model that actually generalizes beyond mountain photography.

---

## 📦 Dependencies

```
torch (cu121)
diffusers
transformers
bitsandbytes
einops
pillow
tqdm
kornia
fastapi
uvicorn
numpy
matplotlib
```

---

## 📄 License

This project is released for educational purposes. Model weights (when available) are for non-commercial research use only, consistent with the LAION dataset license terms.

---

## 🙏 Acknowledgements

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., 2023
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu et al., 2023
- [LAION-5B](https://laion.ai/blog/laion-5b/) — Open dataset used for training
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) — Architecture inspiration
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 8-bit optimizers enabling T4 training
- [Kornia](https://kornia.readthedocs.io/) — GPU-accelerated augmentation pipeline

The whole project, from start to end was made by **Made by Dheeren Tejani**, it was a great learning experience and paved a great path for phase 2