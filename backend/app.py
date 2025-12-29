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

# Initialize Architecture
unet = UNet(UNetConfig()).to(DEVICE)
vae = VAE(VAEConfig()).to(DEVICE)
clip = CLIP(CLIPConfig()).to(DEVICE)

# Load the FIXED Checkpoint
# ⚠️ UPDATE THIS PATH to match your actual file location
CHECKPOINT_PATH = "./unet_step_056000_FIXED.pt" 

try:
    print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    unet.load_state_dict(ckpt['model_state_dict'])
    print("✅ Weights Loaded Successfully")
except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    # We continue so the app starts, but generation will fail if not fixed

# Initialize Tokenizer & Wrapper
# If you have 'tokenizer.pt', load it. Otherwise SimpleTokenizer uses defaults.
tokenizer = SimpleTokenizer()
try:
    tokenizer_data = torch.load("tokenizer.pt", map_location=DEVICE)
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