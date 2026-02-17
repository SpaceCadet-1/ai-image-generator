"""Download all required models for local image generation.

Usage:
    python scripts/download_models.py          # CPU mode (SD 1.5, ~4 GB)
    python scripts/download_models.py --gpu     # GPU mode (adds SDXL + InstantID, ~12 GB)
"""

import argparse
import sys
from pathlib import Path

# Use the OS certificate store (fixes corporate SSL inspection)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Add server/ to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))


def download_base_model(model_id: str, cache_dir: Path):
    """Download the Stable Diffusion base model."""
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    print(f"\n[1/3] Downloading base model: {model_id}")
    if "xl" in model_id.lower():
        StableDiffusionXLPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype="auto",
        )
    else:
        StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype="auto",
        )
    print(f"  -> Saved to {cache_dir}")


def download_instantid_models(models_dir: Path, cache_dir: Path):
    """Download InstantID ControlNet and IP-Adapter weights."""
    from diffusers import ControlNetModel
    from huggingface_hub import hf_hub_download

    print("\n[2/3] Downloading InstantID models from InstantX/InstantID")

    # ControlNet (uses diffusers cache)
    print("  -> Downloading ControlNet...")
    ControlNetModel.from_pretrained(
        "InstantX/InstantID",
        subfolder="ControlNetModel",
        cache_dir=cache_dir,
    )

    # IP-Adapter weights (local dir for direct file access)
    dest_dir = models_dir / "instantid"
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("  -> Downloading IP-Adapter weights (ip-adapter.bin)...")
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir=dest_dir,
    )
    print(f"  -> Saved to {dest_dir}")


def download_insightface(model_dir: Path):
    """Download InsightFace antelopev2 ONNX models for face detection/recognition."""
    from huggingface_hub import snapshot_download
    import zipfile

    print(f"\n[3/3] Downloading InsightFace antelopev2")
    model_dir.mkdir(parents=True, exist_ok=True)

    antelope_dir = model_dir / "models" / "antelopev2"
    if antelope_dir.exists() and any(antelope_dir.glob("*.onnx")):
        print("  -> Already downloaded, skipping")
        return

    # Download the zip using snapshot_download
    downloaded_dir = snapshot_download(
        repo_id="MonsterMMORPG/tools",
        allow_patterns=["antelopev2.zip"],
        local_dir=model_dir / "_tmp_download",
    )
    zip_path = Path(downloaded_dir) / "antelopev2.zip"

    print(f"  -> Extracting to {antelope_dir}")
    antelope_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(model_dir / "models")

    # Clean up temp download
    import shutil
    shutil.rmtree(model_dir / "_tmp_download", ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Download models for AI Image Generator")
    parser.add_argument("--gpu", action="store_true", help="Download SDXL + InstantID models for GPU (adds ~8 GB)")
    parser.add_argument("--cpu-only", action="store_true", help="Download only SD 1.5 models for CPU (~4 GB)")
    args = parser.parse_args()

    from config import (
        MODELS_DIR,
        LOCAL_MODEL_DIR,
        INSIGHTFACE_MODEL_DIR,
    )

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which models to download
    if args.gpu:
        base_models = [
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ]
    else:
        base_models = ["stable-diffusion-v1-5/stable-diffusion-v1-5"]

    # Download all components
    for model_id in base_models:
        download_base_model(model_id, LOCAL_MODEL_DIR)

    if args.gpu:
        download_instantid_models(MODELS_DIR, LOCAL_MODEL_DIR)

    download_insightface(INSIGHTFACE_MODEL_DIR)

    print("\n--- Download complete ---")
    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"Total size: {total / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
