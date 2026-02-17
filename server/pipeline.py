"""Model loading and inference for text-to-image and face-likeness generation.

Two modes:
  - text-to-image: Standard Stable Diffusion generation from a text prompt.
  - face-likeness: InstantID — generates images preserving a person's identity
    from a reference photo using ControlNet + IP-Adapter.

Two device targets:
  - CPU (dev): SD 1.5, 512x512, float32 (text-only, no face support)
  - CUDA (prod): SDXL + InstantID, 1024x1024, float16
"""

import logging
import math
from pathlib import Path

import cv2
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
    INSTANTID_REPO,
    INSTANTID_CONTROLNET_SUBFOLDER,
    INSTANTID_ADAPTER_FILENAME,
    INSIGHTFACE_MODEL_DIR,
    LOCAL_MODEL_DIR,
    MODELS_DIR,
)
import face_analysis

logger = logging.getLogger(__name__)

# Module-level state
_pipe = None


def _draw_face_kps(image_size: tuple[int, int], kps: np.ndarray) -> Image.Image:
    """Draw 5 face keypoints on a blank canvas for ControlNet conditioning.

    Draws colored circles at each keypoint and lines connecting each to the nose.
    This is the standard InstantID conditioning image format.

    Args:
        image_size: (width, height) of the output image.
        kps: 5x2 array of keypoint coordinates (left eye, right eye, nose,
             left mouth corner, right mouth corner).

    Returns:
        PIL Image with keypoints drawn on black background.
    """
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    limb_seq = [[0, 2], [1, 2], [3, 2], [4, 2]]  # Connect each point to nose
    stickwidth = 4

    w, h = image_size
    out_img = np.zeros([h, w, 3], dtype=np.uint8)
    kps = np.array(kps)

    # Draw limbs (connections to nose)
    for limb in limb_seq:
        color = color_list[limb[0]]
        x = kps[limb][:, 0]
        y = kps[limb][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stickwidth),
            int(angle), 0, 360, 1,
        )
        out_img = cv2.fillConvexPoly(out_img, polygon, color)

    # Draw keypoint circles
    for idx, kp in enumerate(kps):
        color = color_list[idx]
        out_img = cv2.circle(out_img, (int(kp[0]), int(kp[1])), 10, color, -1)

    return Image.fromarray(out_img)


def _load_base_pipeline():
    """Load the base Stable Diffusion pipeline."""
    global _pipe

    if USE_GPU:
        from diffusers import ControlNetModel
        from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

        logger.info("Loading InstantID ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            INSTANTID_REPO,
            subfolder=INSTANTID_CONTROLNET_SUBFOLDER,
            torch_dtype=DTYPE,
            cache_dir=LOCAL_MODEL_DIR,
        )

        logger.info("Loading SDXL + InstantID pipeline on CUDA...")
        _pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnet,
            torch_dtype=DTYPE,
            cache_dir=LOCAL_MODEL_DIR,
        ).to(DEVICE)

        # Load InstantID IP-Adapter weights
        adapter_path = MODELS_DIR / "instantid" / INSTANTID_ADAPTER_FILENAME
        if not adapter_path.exists():
            logger.info("Downloading InstantID IP-Adapter weights...")
            from huggingface_hub import hf_hub_download

            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=INSTANTID_REPO,
                filename=INSTANTID_ADAPTER_FILENAME,
                local_dir=adapter_path.parent,
            )

        logger.info("Loading InstantID IP-Adapter weights...")
        _pipe.load_ip_adapter_instantid(str(adapter_path))

        # Verify IP-Adapter loaded correctly
        ip_count = sum(1 for p in _pipe.unet.attn_processors.values()
                       if type(p).__name__.startswith("IPAttn"))
        has_proj = hasattr(_pipe, "image_proj_model")
        proj_params = sum(p.numel() for p in _pipe.image_proj_model.parameters()) if has_proj else 0
        logger.info(f"IP-Adapter verification: {ip_count} IP processors, "
                     f"image_proj_model={'yes' if has_proj else 'MISSING'} ({proj_params:,} params)")

        # Load face detection/recognition ONNX models
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Loading face detection/recognition ONNX models...")
        face_analysis.load(str(INSIGHTFACE_MODEL_DIR), providers)
        logger.info("Face models loaded.")
    else:
        from diffusers import StableDiffusionPipeline

        logger.info("Loading SD 1.5 pipeline on CPU...")
        _pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=LOCAL_MODEL_DIR,
        ).to(DEVICE)

    logger.info("Pipeline loaded.")


def load_models():
    """Load pipeline at startup."""
    _load_base_pipeline()
    logger.info(f"Models loaded on {DEVICE}.")


def _save_image(image: Image.Image) -> str:
    """Save a PIL Image to the output directory and return its filename."""
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
    """Generate an image from a text prompt. Returns the saved filename."""
    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    if USE_GPU:
        # InstantID pipeline requires face inputs — pass zeros with scales=0
        kwargs["image_embeds"] = torch.zeros(1, 512, dtype=DTYPE, device=DEVICE)
        kwargs["image"] = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "black")
        kwargs["controlnet_conditioning_scale"] = 0.0
        kwargs["ip_adapter_scale"] = 0.0

    logger.info(f"Generating text-to-image ({IMAGE_SIZE}x{IMAGE_SIZE}, {NUM_INFERENCE_STEPS} steps)...")
    result = _pipe(**kwargs)
    image = result.images[0]

    return _save_image(image)


def generate_face_likeness(
    prompt: str,
    reference_image: Image.Image,
    face_strength: float = 0.8,
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
        Filename of the generated image.
    """
    if not USE_GPU:
        raise ValueError(
            "Face likeness requires GPU mode (SDXL + InstantID). "
            "This feature is not available on CPU."
        )

    # Extract face embedding and landmarks from reference image
    img_array = np.array(reference_image)
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        img_bgr = img_array[:, :, ::-1]
    else:
        img_bgr = img_array

    face_embed, landmarks = face_analysis.get_face_info(img_bgr)
    logger.info(f"Face embedding: norm={np.linalg.norm(face_embed):.1f}")

    # Prepare face embedding tensor — shape (1, 512)
    face_embed_tensor = torch.from_numpy(face_embed).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Scale landmarks from reference image coords to output image size
    h, w = img_bgr.shape[:2]
    scaled_kps = landmarks.copy()
    scaled_kps[:, 0] = scaled_kps[:, 0] * IMAGE_SIZE / w
    scaled_kps[:, 1] = scaled_kps[:, 1] * IMAGE_SIZE / h
    face_kps_image = _draw_face_kps((IMAGE_SIZE, IMAGE_SIZE), scaled_kps)

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    # Scale balance: ip_adapter_scale drives identity preservation,
    # controlnet_conditioning_scale drives spatial/pose guidance.
    ip_scale = face_strength
    cn_scale = face_strength

    kwargs = {
        "prompt": prompt,
        "image_embeds": face_embed_tensor,
        "image": face_kps_image,
        "controlnet_conditioning_scale": cn_scale,
        "ip_adapter_scale": ip_scale,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt

    logger.info(
        f"Generating face-likeness ({IMAGE_SIZE}x{IMAGE_SIZE}, "
        f"{NUM_INFERENCE_STEPS} steps, ip_scale={ip_scale:.2f}, cn_scale={cn_scale:.2f})..."
    )
    result = _pipe(**kwargs)
    image = result.images[0]

    return _save_image(image)
