"""Device-aware configuration for model selection and paths."""

import os
import torch
from pathlib import Path

# Base paths
SERVER_DIR = Path(__file__).parent
MODELS_DIR = SERVER_DIR / "models"
UPLOADS_DIR = SERVER_DIR / "uploads"

# Device detection: override with DEVICE_MODE=cpu or DEVICE_MODE=cuda
_env_mode = os.environ.get("DEVICE_MODE", "").lower()
if _env_mode == "cuda" and torch.cuda.is_available():
    DEVICE = "cuda"
elif _env_mode == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_GPU = DEVICE == "cuda"

# Model selection based on device
if USE_GPU:
    BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    IMAGE_SIZE = 1024
    NUM_INFERENCE_STEPS = 30
    DTYPE = torch.float16
else:
    BASE_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    IMAGE_SIZE = 512
    NUM_INFERENCE_STEPS = 25
    DTYPE = torch.float32

# InstantID face-likeness (SDXL + GPU only)
INSTANTID_REPO = "InstantX/InstantID"
INSTANTID_CONTROLNET_SUBFOLDER = "ControlNetModel"
INSTANTID_ADAPTER_FILENAME = "ip-adapter.bin"

# InsightFace ONNX models for face detection/recognition
INSIGHTFACE_MODEL_DIR = MODELS_DIR / "insightface"

# Local model cache directory (downloaded by scripts/download_models.py)
LOCAL_MODEL_DIR = MODELS_DIR / "diffusers"

# Upload constraints
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Server
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "3001"))


def print_config():
    """Print current configuration for debugging."""
    print(f"Device:          {DEVICE}")
    print(f"Base model:      {BASE_MODEL_ID}")
    print(f"Image size:      {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"Dtype:           {DTYPE}")
    if USE_GPU:
        print(f"Face method:     InstantID ({INSTANTID_REPO})")
    else:
        print(f"Face method:     None (CPU mode)")
    print(f"Models dir:      {MODELS_DIR}")
