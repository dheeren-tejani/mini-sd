# ⚡ Toy Stable Diffusion: A Tiny Latent Diffusion Model (LDM)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kornia](https://img.shields.io/badge/Kornia-Augmentations-blue?logo=python&logoColor=white)](https://kornia.github.io/)
[![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Training_Phase_2-orange)]()

**Author:** Dheeren Tejani  
**Current Architecture:** 335.5M Total Parameters | 512x512 Resolution | 8x Compression

---

## 📖 Overview

**Toy Stable Diffusion** is a research project aiming to architect and train a fully functional Latent Diffusion Model (LDM) **from scratch** on consumer-grade hardware (Single T4 GPU).

Unlike standard implementations that rely on the Hugging Face `diffusers` library, this repository contains a custom-built PyTorch engine. The goal is to study the fundamental physics of generative AI, optimize training pipelines for constrained hardware.

This project is currently bridging the gap between "Toy" models and production-grade generative systems by proving that meaningful latent representations can be learned with <4M parameter VAEs through careful hyperparameter tuning.

---

## 🧪 Research & Experiments (The Evolution)

This repository documents the iterative research process. Instead of hiding failed experiments, we document them to show the path to optimization.

### **Phase 2: High-Fidelity 256px Model (Current Focus)** 🚀
* **Status:** Training in Progress (Run B)
* **Objective:** Eliminate "Posterior Collapse" and improve geometric stability.
* **Architecture:** Optimized VAE (8x Compression) + U-Net.
* **Key Engineering Decisions:**
    * **Compression Ratio:** Switched from 16x (Phase 1) to **8x**. This was the critical fix. The previous 16x bottleneck forced the lightweight VAE to discard too much spatial information.
    * **Cyclical KL Annealing:** Implemented a sinusoidal schedule for **beta** (KL Weight) to balance reconstruction accuracy with latent distribution regularity.
    * **GPU Augmentation:** Integrated `Kornia` to offload normalization and flipping to the GPU, removing CPU bottlenecks.
* **Current Results:** The VAE is graduating from "grey static" to clear geometric reconstruction (e.g., recognizable converging street lines, distinct silhouettes).

### **Phase 1: 128px Prototype (Legacy)** 📉
* **Goal:** Feasibility test on limited T4 hardware.
* **Outcome:** Model successfully trained but suffered from severe "waxy" artifacts.
* **Post-Mortem:**
    * **Failure Mode:** The 16x compression ratio (128px \to 8px latent) was too aggressive for a shallow ResNet encoder.
    * **Loss Analysis:** The model minimized loss by averaging pixel values (Posterior Collapse) rather than learning texture.
* **Artifacts:** [View Phase 1 Logs, Curves & Samples](./backend/v1_legacy)

---

## 🏗️ Model Architecture (The "Tiny" Stack)

We are running a highly optimized "Mini" architecture designed to fit entirely within 16GB VRAM while allowing for large batch sizes (128+).

| Component | Parameters | Architecture Details | Function |
| :--- | :--- | :--- | :--- |
| **VAE** | **3.13 M** | ResNet-based, 512px to 64px (8x) | Compresses images into latent space. |
| **CLIP** | **101.5 M** | ViT-B/32 equivalent | Encodes text prompts into embeddings. |
| **U-Net** | **230.85 M** | 192 base channels, 2 ResBlocks | The diffusion "brain" that denoises latents. |
| **Total** | **~335.5 M** | **End-to-End Pipeline** | **~32% the size of SD v1.5 (Total)** |

### Why these numbers matter
Standard Stable Diffusion models utilize ~1 Billion parameters. By shrinking the model to ~335M, we enable:
1.  **Mobile-Ready Scale:** This architecture is theoretically small enough to run on high-end mobile DSPs.
2.  **Fast Training:** We can iterate on architecture changes in hours/days rather than weeks.
3.  **Experimental Agility:** Allows for testing new layers (like **Rangeflow**) without burning thousands of dollars in compute.

---

## 📂 Project Structure

The codebase is versioned to support the migration from the Phase 1 prototype to the Phase 2 production model.

```text
ToyStableDiffusion/
├── backend/
│   ├── models/                 
│   │   ├── clip/
│   │   └── tokenizer/
│   │   └── unet/
│   │   └── vae/
│   └── app.py           # FastAPI entry point (points to v2)
│   └── model_utils.py  # New 8x Compression Architecture
│   └── requirements.txt
├── frontend/
├── metadata/
│   ├── v1/                 # 💀 Old 128px Prototype (Reference only)
│   │   ├── clip/
│   │   ├── unet/
│   │   ├── vae/
│   │   ├── app.py
│   │   ├── dataset_samples.png
│   │   ├── model_utils.py
│   │   ├── v1_training_script.ipynb
│   ├── v2/                 # 🚀 Current 512px Production Model
└── README.md

```

---

## ⚡ Key Engineering Features

### 1. The "Guerrilla" Training Loop

Designed for the unstable nature of Google Colab Free Tier:

* **Sliding Window Checkpointing:** Automatically keeps the last 3 checkpoints to save storage.
* **Auto-Resume:** The launcher automatically detects the latest `.pt` file and resumes training state (optimizer, scheduler, and epoch) exactly where it left off.

### 2. Kornia GPU Pipeline

We identified a CPU bottleneck in standard `torchvision` transforms.

* **Solution:** Moved Normalization (`(x - mean)/std`) and Augmentations to the GPU using Kornia.
* **Result:** Increased throughput from ~180 img/sec to **230 img/sec** on a T4.

Achieved near 100% GPU Utilization with a Dataloader/GPU throughput ratio of >5x, ensuring the T4 is never bottlenecked by CPU decoding.

---

## 🚀 Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch 2.0+ (CUDA recommended)
* 16GB VRAM (for training) or 4GB VRAM (for inference)

### Setup

```bash
git clone https://github.com/dheeren-tejani/mini-sd.git
cd mini-sd/backend
pip install -r requirements.txt
cd..
cd frontend
npm install

```

### Inference (Generate an Image)

The server defaults to the optimized **v2** model.

```bash
# Start the backend server
uvicorn backend.app:app --reload

```

Then start the frontend server and run

```bash
npm run dev

```

Once running, navigate to http://localhost:3000 to interact with the diffusion interface.

---

## 📊 Roadmap

* [x] **Phase 1:** Proof of concept (128px) - *Completed*
* [ ] **Phase 2 - VAE:** Optimization (8x Compression) - *In Progress*
* [ ] **Phase 2 - CLIP:** Training Alignment - *Upcoming*
* [ ] **Phase 2 - U-Net:** Latent Diffusion Training - *Upcoming*

---

## 🤝 Contributing & Citation

This is an active research project. If you are interested in **On-Device Generative AI** or **Interval Arithmetic for Diffusion**, feel free to open an issue or PR.

If you use this codebase for your research, please cite:

```bibtex
@misc{tejani2025toystablediffusion,
  author = {Tejani, Dheeren},
  title = {Toy Stable Diffusion: Architectural Optimization for Low-Resource LDMs},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}

```