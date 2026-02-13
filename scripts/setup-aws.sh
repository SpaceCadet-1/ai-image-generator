#!/bin/bash
# setup-aws.sh — One-time setup on AWS Deep Learning AMI (Ubuntu 24.04, PyTorch 2.9)
# Instance: g5.xlarge (A10G 24GB VRAM)
#
# Usage: bash scripts/setup-aws.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo ""
echo "================================="
echo "  AI Image Generator — AWS Setup"
echo "================================="
echo ""
echo "Repo: $REPO_DIR"

# ── 1. Activate the Deep Learning AMI's PyTorch conda env ──
echo ""
echo "[1/4] Activating PyTorch conda environment..."
source activate pytorch 2>/dev/null || conda activate pytorch
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── 2. Install Python dependencies (torch is already installed with CUDA) ──
echo ""
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip

# Install everything except torch (already on the AMI with CUDA support)
pip install \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    python-multipart==0.0.20 \
    Pillow==11.1.0 \
    "diffusers>=0.32.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "safetensors>=0.4.0" \
    "peft>=0.10.0" \
    "opencv-python-headless>=4.8.0" \
    "huggingface-hub>=0.23.0" \
    numpy

# Use onnxruntime-gpu instead of onnxruntime (for CUDA execution provider)
pip install "onnxruntime-gpu>=1.17.0"

# ── 3. Download models (~11.5 GB for GPU mode) ──
echo ""
echo "[3/4] Downloading models (SDXL + FaceID + CLIP + InsightFace)..."
echo "       This will download ~11.5 GB on first run."
cd "$REPO_DIR"
python scripts/download_models.py --gpu

# ── 4. Install Node.js + frontend dependencies ──
echo ""
echo "[4/4] Setting up frontend..."
if ! command -v node &>/dev/null; then
    echo "Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
echo "Node $(node --version), npm $(npm --version)"

cd "$REPO_DIR/client"
npm install

# ── Done ──
echo ""
echo "================================="
echo "  Setup complete!"
echo "================================="
echo ""
echo "To start:  bash start.sh"
echo "To stop:   bash stop.sh"
echo ""
echo "From your laptop, connect with:"
echo "  ssh -i <key>.pem -L 5173:localhost:5173 -L 3001:localhost:3001 ubuntu@<ip>"
echo "  Then open http://localhost:5173"
echo ""
