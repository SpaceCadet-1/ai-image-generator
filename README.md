# AI Image Generator

A fully offline web application that generates images from text descriptions using local Stable Diffusion models. Supports **face-likeness preservation** — upload a reference photo and generate new images that maintain the person's identity.

## Features

- Text-to-image generation with style presets and mood controls
- Face-likeness mode using IP-Adapter FaceID Plus V2
- Adjustable face strength slider (0.0-1.0)
- Runs entirely offline — no API keys needed
- Dual-device support: CPU (dev) and GPU (prod)

## Prerequisites

- Python 3.10+
- Node.js 18+ (for React frontend)
- ~4 GB disk space for models (CPU mode)
- ~12 GB disk space for models (GPU mode)

## Setup

### 1. Python environment

```bash
cd server
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download models (~4 GB for CPU, ~12 GB with GPU)

```bash
# CPU only (SD 1.5 — for development)
python scripts/download_models.py

# CPU + GPU (adds SDXL — for production)
python scripts/download_models.py --gpu
```

### 3. Install frontend dependencies

```bash
cd client
npm install
```

### 4. Run the application

Start the backend server:

```bash
cd server
uvicorn main:app --host 0.0.0.0 --port 3001
```

Start the frontend (in a new terminal):

```bash
cd client
npm run dev
```

### 5. Use the application

1. Open http://localhost:3000 in your browser
2. Enter an image description
3. Optionally upload a reference face photo
4. Click "Generate Image"
5. Wait for the local model to generate (~45-90 sec on CPU, ~5-15 sec on GPU)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE_MODE` | auto-detected | Force `cpu` or `cuda` |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `3001` | Server port |

## Project Structure

```
ai-image-generator/
├── client/                  # React frontend (Vite)
│   └── src/
│       ├── App.jsx          # Main component
│       └── App.css          # Styles
├── server/                  # Python FastAPI backend
│   ├── main.py              # FastAPI server
│   ├── pipeline.py          # Model loading & inference
│   ├── config.py            # Device-aware configuration
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # Downloaded model weights (gitignored)
│   └── uploads/             # Temp upload directory (gitignored)
├── scripts/
│   └── download_models.py   # Model download script
└── server_legacy/           # Archived Express/OpenAI server
    ├── index.js
    └── package.json
```

## Performance

| Environment | Mode | Resolution | Time per Image |
|-------------|------|-----------|----------------|
| CPU (dev) | SD 1.5 text-only | 512x512 | 30-60 sec |
| CPU (dev) | SD 1.5 + FaceID | 512x512 | 45-90 sec |
| A10G GPU (prod) | SDXL text-only | 1024x1024 | 5-10 sec |
| A10G GPU (prod) | SDXL + FaceID | 1024x1024 | 8-15 sec |
