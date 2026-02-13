"""Model loading and inference for text-to-image and face-likeness generation.

Two modes:
  - text-to-image: Standard Stable Diffusion generation from a text prompt.
  - face-likeness: IP-Adapter FaceID Plus V2 — generates images preserving
    a person's identity from a reference photo.

Two device targets:
  - CPU (dev): SD 1.5, 512x512, float32
  - CUDA (prod): SDXL, 1024x1024, float16
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import (
    DEVICE,
    USE_GPU,
    DTYPE,
    BASE_MODEL_ID,
    IMAGE_SIZE,
    NUM_INFERENCE_STEPS,
    FACE_ADAPTER_REPO,
    FACE_ADAPTER_FILENAME,
    FACE_LORA_FILENAME,
    CLIP_IMAGE_ENCODER_ID,
    INSIGHTFACE_MODEL_DIR,
    LOCAL_MODEL_DIR,
    MODELS_DIR,
)
import face_analysis

logger = logging.getLogger(__name__)

# Module-level state — loaded once at startup
_pipe = None
_face_loaded = False


def _load_base_pipeline():
    """Load the base Stable Diffusion pipeline."""
    global _pipe

    if USE_GPU:
        from diffusers import StableDiffusionXLPipeline

        logger.info("Loading SDXL pipeline on CUDA...")
        _pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=LOCAL_MODEL_DIR,
        ).to(DEVICE)
    else:
        from diffusers import StableDiffusionPipeline

        logger.info("Loading SD 1.5 pipeline on CPU...")
        _pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=LOCAL_MODEL_DIR,
        ).to(DEVICE)

    # GPU memory optimization
    if USE_GPU:
        _pipe.enable_model_cpu_offload()

    logger.info("Base pipeline loaded.")


def _load_face_detection():
    """Load ONNX face detection and recognition models."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if USE_GPU
        else ["CPUExecutionProvider"]
    )
    logger.info("Loading face detection/recognition ONNX models...")
    face_analysis.load(str(INSIGHTFACE_MODEL_DIR), providers)
    logger.info("Face models loaded.")


def _load_face_adapter():
    """Load IP-Adapter FaceID Plus V2 onto the pipeline."""
    global _pipe, _face_loaded

    from huggingface_hub import hf_hub_download
    from transformers import CLIPVisionModelWithProjection

    logger.info("Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_IMAGE_ENCODER_ID,
        torch_dtype=DTYPE,
        cache_dir=LOCAL_MODEL_DIR,
    ).to(DEVICE)

    # Set image encoder on pipeline before loading IP-Adapter
    _pipe.image_encoder = image_encoder

    # Download adapter weights if not already local
    adapter_path = MODELS_DIR / "ip-adapter-faceid" / FACE_ADAPTER_FILENAME
    lora_path = MODELS_DIR / "ip-adapter-faceid" / FACE_LORA_FILENAME

    if not adapter_path.exists():
        logger.info(f"Downloading {FACE_ADAPTER_FILENAME}...")
        hf_hub_download(
            repo_id=FACE_ADAPTER_REPO,
            filename=FACE_ADAPTER_FILENAME,
            local_dir=MODELS_DIR / "ip-adapter-faceid",
        )
    if not lora_path.exists():
        logger.info(f"Downloading {FACE_LORA_FILENAME}...")
        hf_hub_download(
            repo_id=FACE_ADAPTER_REPO,
            filename=FACE_LORA_FILENAME,
            local_dir=MODELS_DIR / "ip-adapter-faceid",
        )

    logger.info("Loading IP-Adapter FaceID Plus V2...")

    # Load LoRA weights
    _pipe.load_lora_weights(
        str(lora_path.parent),
        weight_name=FACE_LORA_FILENAME,
    )
    _pipe.fuse_lora(lora_scale=0.5)

    # Load IP-Adapter (image_encoder_folder=None since we set it above)
    _pipe.load_ip_adapter(
        str(adapter_path.parent),
        subfolder="",
        weight_name=FACE_ADAPTER_FILENAME,
        image_encoder_folder=None,
    )

    _face_loaded = True
    logger.info("Face adapter loaded.")


def load_models():
    """Load all models at startup. Call once from main.py lifespan."""
    _load_base_pipeline()
    _load_face_detection()
    _load_face_adapter()
    logger.info(f"All models loaded on {DEVICE}.")


def _extract_face_embedding(image: Image.Image) -> np.ndarray:
    """Extract face embedding from a PIL image."""
    img_array = np.array(image)
    # face_analysis expects BGR
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_bgr = img_array[:, :, ::-1]
    else:
        img_bgr = img_array

    return face_analysis.get_face_embedding(img_bgr)


def _encode_clip_features(image: Image.Image) -> torch.Tensor:
    """Encode a PIL image through CLIP to get hidden states for FaceID Plus.

    Returns tensor of shape (1, 257, 1280) — CLIP ViT-H-14 hidden states.
    """
    pixel_values = _pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)
    clip_output = _pipe.image_encoder(pixel_values, output_hidden_states=True)
    return clip_output.hidden_states[-2]


def _set_clip_embeds(clip_hidden: torch.Tensor | None):
    """Set CLIP embeddings on the FaceID Plus projection layer.

    The IPAdapterFaceIDPlusImageProjection layer reads self.clip_embeds
    during its forward pass. This must be set before each pipeline call.

    Args:
        clip_hidden: CLIP hidden states (1, 257, 1280) or None for zeros.
    """
    proj_layer = _pipe.unet.encoder_hid_proj.image_projection_layers[0]

    if clip_hidden is None:
        # For text-only: zeros. Shape must be (batch, 1, 257, 1280) where
        # batch=2 for CFG (negative + positive).
        proj_layer.clip_embeds = torch.zeros(
            2, 1, 257, 1280, dtype=DTYPE, device=DEVICE
        )
    else:
        # For face mode: negative (zeros) + positive (actual CLIP features)
        clip_hidden = clip_hidden.unsqueeze(0)  # (1, 1, 257, 1280)
        neg_clip = torch.zeros_like(clip_hidden)
        proj_layer.clip_embeds = torch.cat([neg_clip, clip_hidden], dim=0)  # (2, 1, 257, 1280)

    proj_layer.shortcut = True  # FaceID Plus V2 uses shortcut connection


def _save_image(image: Image.Image) -> str:
    """Save a PIL Image to the output directory and return its filename.

    Uses JPEG for smaller file sizes (~200KB vs ~4MB PNG).
    """
    import uuid
    from config import UPLOADS_DIR

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = UPLOADS_DIR / filename
    image.save(filepath, format="JPEG", quality=90)
    logger.info(f"Saved image: {filepath} ({filepath.stat().st_size / 1024:.0f} KB)")
    return filename


def generate_text_to_image(
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
) -> str:
    """Generate an image from a text prompt. Returns a base64 data URI."""
    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # Disable face adapter influence for text-only generation
    if _face_loaded:
        _pipe.set_ip_adapter_scale(0.0)
        _set_clip_embeds(None)

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    # Pass zero face ID embeddings (required even when scale is 0)
    if _face_loaded:
        zero_embed = torch.zeros(2, 1, 512, dtype=DTYPE, device=DEVICE)
        kwargs["ip_adapter_image_embeds"] = [zero_embed]

    logger.info(f"Generating text-to-image ({IMAGE_SIZE}x{IMAGE_SIZE}, {NUM_INFERENCE_STEPS} steps)...")
    result = _pipe(**kwargs)
    image = result.images[0]

    return _save_image(image)


def generate_face_likeness(
    prompt: str,
    reference_image: Image.Image,
    face_strength: float = 0.6,
    negative_prompt: str = "",
    seed: int | None = None,
) -> str:
    """Generate an image preserving face likeness from a reference photo.

    Args:
        prompt: Text description of the desired image.
        reference_image: PIL Image containing a face.
        face_strength: How strongly to preserve the face (0.0-1.0).
        negative_prompt: Things to avoid in the generated image.
        seed: Optional random seed for reproducibility.

    Returns:
        Base64 data URI of the generated image.
    """
    # Extract face ID embedding (512-dim from InsightFace)
    face_embed = _extract_face_embedding(reference_image)
    face_embed_single = torch.from_numpy(face_embed).unsqueeze(0).unsqueeze(0).to(DEVICE, dtype=DTYPE)
    neg_embed = torch.zeros_like(face_embed_single)
    face_embed_tensor = torch.cat([neg_embed, face_embed_single], dim=0)  # (2, 1, 512)

    # Encode reference image through CLIP for the Plus V2 projection layer
    clip_hidden = _encode_clip_features(reference_image)
    _set_clip_embeds(clip_hidden)

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # Set IP-Adapter influence strength
    _pipe.set_ip_adapter_scale(face_strength)

    kwargs = {
        "prompt": prompt,
        "ip_adapter_image_embeds": [face_embed_tensor],
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    logger.info(
        f"Generating face-likeness ({IMAGE_SIZE}x{IMAGE_SIZE}, "
        f"{NUM_INFERENCE_STEPS} steps, strength={face_strength})..."
    )
    result = _pipe(**kwargs)
    image = result.images[0]

    return _save_image(image)
