import io
import base64
import logging
from contextlib import asynccontextmanager

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_state = {"status": "loading", "pipe_t2i": None, "pipe_i2i": None}

MODEL_ID = "black-forest-labs/FLUX.1-schnell"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FLUX.1-schnell pipeline — first run downloads ~12GB...")
    try:
        pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
        pipe.enable_sequential_cpu_offload()

        # Build img2img from same weights
        pipe_i2i = FluxImg2ImgPipeline.from_pipe(pipe)

        model_state["pipe_t2i"] = pipe
        model_state["pipe_i2i"] = pipe_i2i
        model_state["status"] = "ready"
        logger.info("FLUX.1-schnell loaded and ready.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_state["status"] = "error"

    yield

    # Shutdown: free VRAM
    logger.info("Shutting down — freeing VRAM...")
    model_state["pipe_t2i"] = None
    model_state["pipe_i2i"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": model_state["status"]}


def generate_image_response(image: Image.Image) -> JSONResponse:
    """Convert a PIL Image to a base64 JSON response."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return JSONResponse({"imageUrl": f"data:image/png;base64,{b64}"})


@app.post("/generate")
def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
):
    if model_state["status"] != "ready":
        return JSONResponse({"error": "Model is not ready"}, status_code=503)

    pipe = model_state["pipe_t2i"]
    try:
        result = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
        )
        return generate_image_response(result.images[0])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse({"error": "GPU out of memory. Try again."}, status_code=500)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate-img2img")
def generate_img2img(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.9),
    image: UploadFile = File(...),
):
    if model_state["status"] != "ready":
        return JSONResponse({"error": "Model is not ready"}, status_code=503)

    pipe = model_state["pipe_i2i"]
    try:
        raw = image.file.read()
        init_image = Image.open(io.BytesIO(raw)).convert("RGB").resize((512, 512))

        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=4,
            guidance_scale=0.0,
        )
        return generate_image_response(result.images[0])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return JSONResponse({"error": "GPU out of memory. Try again."}, status_code=500)
    except Exception as e:
        logger.error(f"img2img generation failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
