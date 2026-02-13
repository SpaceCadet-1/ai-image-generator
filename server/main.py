"""FastAPI server — replaces Express server/index.js.

Same API contract:
  POST /api/generate  (multipart/form-data)
    - prompt: str (required)
    - referenceImage: file (optional)
    - faceStrength: float 0.0-1.0 (optional, default 0.6)
  Returns: { "imageUrl": "data:image/png;base64,..." }
"""

# Use the OS certificate store (fixes corporate SSL inspection)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from config import UPLOADS_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE, HOST, PORT, print_config
import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    print_config()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading models (this may take a minute on first run)...")
    pipeline.load_models()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="AI Image Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/generate")
async def generate_image(
    prompt: str = Form(...),
    referenceImage: UploadFile | None = File(None),
    faceStrength: float = Form(0.6),
):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    temp_path: Path | None = None

    try:
        if referenceImage and referenceImage.filename:
            # Validate file extension
            ext = Path(referenceImage.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="Only PNG, JPG, and WebP files are allowed",
                )

            # Validate file size by reading content
            contents = await referenceImage.read()
            if len(contents) > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail="File is too large. Maximum size is 25MB.",
                )

            # Save to temp file and open as PIL Image
            temp_path = UPLOADS_DIR / f"upload_{id(referenceImage)}{ext}"
            temp_path.write_bytes(contents)

            ref_image = Image.open(temp_path).convert("RGB")

            # Clamp face strength
            strength = max(0.0, min(1.0, faceStrength))

            image_url = pipeline.generate_face_likeness(
                prompt=prompt.strip(),
                reference_image=ref_image,
                face_strength=strength,
            )
        else:
            image_url = pipeline.generate_text_to_image(prompt=prompt.strip())

        return {"imageUrl": image_url}

    except ValueError as e:
        logger.error(f"ValueError: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
