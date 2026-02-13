#!/bin/bash
# setup-aws.sh — One-time setup on AWS Deep Learning AMI (Ubuntu 24.04)
# Instance: g5.xlarge (A10G 24GB VRAM, CUDA 13.0)
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

# ── 1. Create Python venv ──
echo ""
echo "[1/4] Setting up Python virtual environment..."
cd "$REPO_DIR/server"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

python --version
echo "venv: $VIRTUAL_ENV"

# ── 2. Install Python dependencies ──
echo ""
echo "[2/4] Installing Python dependencies..."

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install remaining deps
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

# onnxruntime-gpu for CUDA face detection
pip install "onnxruntime-gpu>=1.17.0"

# ── 3. Download models (~11.5 GB for GPU mode) ──
echo ""
echo "[3/4] Downloading models (SDXL + FaceID + CLIP + InsightFace)..."
echo "       This will download ~11.5 GB on first run."
cd "$REPO_DIR"
.venv/bin/python scripts/download_models.py --gpu || server/.venv/bin/python scripts/download_models.py --gpu

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
echo "Connect from your laptop with:"
echo "  .\scripts\connect.ps1 i-068e8e5b4c7fe01e9"
echo "  Then open http://localhost:5173"
echo ""
