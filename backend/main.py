import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import SERVER_CONFIG as srv, MODEL_CONFIG as mcfg
from logger import log
from model_manager import manager


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — load models at startup
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models when the server starts."""
    import traceback
    log.info("Toy SD backend starting…")
    try:
        manager.load_models()
    except Exception as e:
        log.error("=" * 60)
        log.error("FATAL — Model loading failed. Server will start but")
        log.error("cannot generate images until this is fixed.")
        log.error("")
        log.error(f"Error: {e}")
        log.error("")
        log.error("Full traceback:")
        for line in traceback.format_exc().splitlines():
            log.error(f"  {line}")
        log.error("=" * 60)
        log.error("Fix the paths in config.py, then restart the server.")
        log.error("=" * 60)
        # Server stays alive so /health returns {model_ready: false}
        # giving you a clear signal without cryptic uvicorn crashes.
    yield
    log.info("Toy SD backend shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Toy Stable Diffusion API",
    description="Flow Matching image generation with CLIP + VAE + UNet.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=srv.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512, description="Text prompt")
    steps: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of inference steps (1–50). More = slower but better.",
    )
    cfg_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Classifier-free guidance scale (1–20).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2_147_483_647,
        description="Random seed for reproducibility.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v.strip()

    model_config = {"json_schema_extra": {
        "example": {
            "prompt": "a serene mountain lake at sunset, photorealistic",
            "steps": 20,
            "cfg_scale": 7.5,
            "seed": 42,
        }
    }}


class GenerateResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG data-URI")
    prompt: str
    steps: int
    cfg_scale: float
    seed: int
    generation_time_s: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    device: str


class ConfigResponse(BaseModel):
    max_steps: int
    min_steps: int
    max_cfg_scale: float
    min_cfg_scale: float
    default_steps: int
    default_cfg_scale: float
    default_seed: int
    output_image_size: int
    supported_dimensions: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# Middleware — request timing
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    # Let CORS middleware handle OPTIONS preflight without interference
    if request.method == "OPTIONS":
        return await call_next(request)
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """
    Readiness probe.
    The frontend sidebar shows 'Ready' / 'Processing…' based on this.
    """
    import torch
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok" if manager.is_ready else "loading",
        model_ready=manager.is_ready,
        device=device_str,
    )


@app.get("/config", response_model=ConfigResponse, tags=["meta"])
async def get_config():
    """
    Expose inference parameter ranges so the frontend sliders stay in sync.
    """
    return ConfigResponse(
        max_steps=srv.max_inference_steps,
        min_steps=srv.min_inference_steps,
        max_cfg_scale=srv.max_cfg_scale,
        min_cfg_scale=srv.min_cfg_scale,
        default_steps=mcfg.flow_inference_steps,
        default_cfg_scale=7.5,
        default_seed=42,
        output_image_size=srv.output_image_size,
        # Frontend currently offers two dimension options
        supported_dimensions=["512x512", "768x768"],
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_image(req: GenerateRequest):
    """
    Generate an image from a text prompt using Flow Matching diffusion.

    Returns a base64-encoded PNG data-URI in the `image` field.
    Paste it directly into `<img src="…">` or use it as-is in the Canvas.
    """
    if not manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please try again in a few seconds.",
        )

    log.info(
        f"[/generate] prompt='{req.prompt[:80]}' "
        f"steps={req.steps} cfg={req.cfg_scale} seed={req.seed}"
    )

    t0 = time.perf_counter()
    try:
        # Run inference in a thread-pool executor so the event loop isn't blocked
        import asyncio

        loop = asyncio.get_event_loop()
        image_data_uri = await loop.run_in_executor(
            None,
            lambda: manager.generate(
                prompt=req.prompt,
                steps=req.steps,
                cfg_scale=req.cfg_scale,
                seed=req.seed,
            ),
        )
    except RuntimeError as e:
        log.error(f"Generation runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.exception(f"Unexpected error during generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}",
        )

    elapsed = time.perf_counter() - t0
    log.info(f"[/generate] done in {elapsed:.2f}s")

    return GenerateResponse(
        image=image_data_uri,
        prompt=req.prompt,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        generation_time_s=round(elapsed, 3),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Global error handler
# ══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Dev entrypoint
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=srv.host,
        port=srv.port,
        reload=False,
        log_level="info",
    )