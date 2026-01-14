import torch
import io
import base64
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms as T

# IMPORT YOUR BRAIN 🧠
# Assumes 'model_utils.py' is in the same directory
from model_utils import (
    UNet, VAE, CLIP, 
    UNetConfig, VAEConfig, CLIPConfig, 
    InferenceWrapper, SimpleTokenizer
)

# Initialize App
app = FastAPI(title="RangeFlow AI API")

# Setup CORS (Allows Frontend to talk to Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# 1. LOAD MODEL (Global Setup)
# ===========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing RangeFlow on {DEVICE}...")

# 1. Initialize Architecture
unet = UNet(UNetConfig()).to(DEVICE)
vae = VAE(VAEConfig()).to(DEVICE)
clip = CLIP(CLIPConfig()).to(DEVICE)

# 2. LOAD VAE WEIGHTS (Crucial!)
try:
    # Update path to where you stored your VAE checkpoint
    vae_ckpt = torch.load("./models/vae/vae_best.pt", map_location=DEVICE)
    # Handle if it's a full checkpoint dict or just state_dict
    if 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    print("✅ VAE Weights Loaded")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load VAE: {e}")

# 3. LOAD CLIP WEIGHTS (Crucial!)
try:
    # Update path to where you stored your CLIP checkpoint
    clip_ckpt = torch.load("./models/clip/clip_best.pt", map_location=DEVICE)
    if 'model_state_dict' in clip_ckpt:
        clip.load_state_dict(clip_ckpt['model_state_dict'])
    else:
        clip.load_state_dict(clip_ckpt)
    print("✅ CLIP Weights Loaded")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load CLIP: {e}")

# 4. LOAD UNET (The one you are already loading)
try:
    # Use the FIXED checkpoint you verified
    unet_path = "./models/unet/unet_step_056000_FIXED.pt" 
    unet_ckpt = torch.load(unet_path, map_location=DEVICE)
    
    # ✅ EMA WEIGHTS LOGIC
    if 'ema_model_state_dict' in unet_ckpt:
        print(f"✨ Loading EMA Weights from {unet_path} (Smoother!)")
        unet.load_state_dict(unet_ckpt['ema_model_state_dict'])
    elif 'model_state_dict' in unet_ckpt:
        print(f"⚠️ EMA not found in {unet_path}, loading standard weights.")
        unet.load_state_dict(unet_ckpt['model_state_dict'])
    else:
        unet.load_state_dict(unet_ckpt)
        
    print("✅ UNet Weights Loaded")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load UNet: {e}")

tokenizer = SimpleTokenizer()
try:
    tokenizer_data = torch.load("./models/tokenizer/tokenizer.pt", map_location=DEVICE)
    tokenizer.word2idx = tokenizer_data["word2idx"]
    tokenizer.idx2word = tokenizer_data["idx2word"]
    print("✅ Tokenizer Vocab Loaded")
except:
    print("⚠️ Warning: 'tokenizer.pt' not found. Using empty/default vocab.")

# Create the Inference Helper
params = InferenceWrapper(unet, vae, clip, tokenizer, UNetConfig(), DEVICE)

# ===========================
# 2. DEFINE INPUT SCHEMA
# ===========================
class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 30
    cfg_scale: float = 7.5
    seed: int = 42

# ===========================
# 3. API ENDPOINT
# ===========================
@app.post("/generate")
def generate(req: GenerateRequest):
    print(f"🎨 Generating: '{req.prompt}' (Steps: {req.steps}, CFG: {req.cfg_scale}, Seed: {req.seed})")
    
    # Set Seed
    torch.manual_seed(req.seed)
    
    try:
        # Run Inference
        with torch.no_grad():
            images = params.sample(
                [req.prompt], 
                num_inference_steps=req.steps,
                guidance_scale=req.cfg_scale
            )
        
        # Process Image (Tensor -> Base64)
        img_tensor = images[0].cpu().clamp(0, 1)
        pil_img = T.ToPILImage()(img_tensor)
        
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": f"data:image/png;base64,{img_str}"}

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)