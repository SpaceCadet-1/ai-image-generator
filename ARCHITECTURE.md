# AI Image Generator - Architecture Reference

## Stack

| Layer | Technology | Port |
|-------|-----------|------|
| Frontend | React 18 + Vite 5 | 5173 (dev) / served by FastAPI (prod) |
| Backend | Python FastAPI + Uvicorn | 3001 |
| AI (CPU) | Stable Diffusion 1.5 (text-only) | — |
| AI (GPU) | Stable Diffusion XL + InstantID (ControlNet + IP-Adapter) | — |
| Face Detection | Custom ONNX (SCRFD + ArcFace, antelopev2) | — |

No database, authentication, or external API calls. Fully offline after model download.

## Project Structure

```
ai-image-generator/
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── client/                     # React frontend (Vite)
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx            # Entry point
│       ├── App.jsx             # Single-component app with prompt builder UI
│       └── App.css             # Dark theme, gradient styling, responsive
├── server/                     # Python FastAPI backend
│   ├── main.py                 # FastAPI server — replaces Express index.js
│   ├── pipeline.py             # Model loading + inference (text & face modes)
│   ├── config.py               # Device-aware config (CPU/GPU auto-selection)
│   ├── requirements.txt        # Python dependencies
│   ├── pipeline_stable_diffusion_xl_instantid.py  # InstantID community pipeline
│   ├── ip_adapter/             # IP-Adapter module (resampler, attention processor)
│   ├── models/                 # Downloaded weights (gitignored)
│   │   ├── diffusers/          # HuggingFace model cache
│   │   ├── instantid/          # InstantID IP-Adapter weights
│   │   └── insightface/        # Face ONNX models (SCRFD + ArcFace)
│   └── uploads/                # Generated images as JPEG (gitignored)
├── scripts/
│   ├── download_models.py      # Downloads all models from HuggingFace
│   ├── setup-aws.sh            # One-time AWS instance setup
│   ├── install-service.sh      # Install as systemd service
│   └── connect.ps1             # SSM tunnel from Windows laptop
└── server_legacy/              # Archived Express/OpenAI server
    ├── index.js
    └── package.json
```

## Server

### Endpoint

`POST /api/generate` (multipart/form-data)

**Request fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | yes | Text description of the image |
| `referenceImage` | file | no | Face reference photo (PNG/JPG/WebP, max 25MB) |
| `faceStrength` | float | no | Face likeness strength 0.0-1.0 (default 0.8) |

**Success response (200):**
```json
{ "imageUrl": "/generated/abc123.jpg" }
```

**Error responses:**
- `400` — prompt missing, no face detected, invalid file type/size
- `500` — model inference failure
- Response body: `{ "detail": "<error message>" }`

### Generation Modes

1. **Text-to-image** — no reference image uploaded. Uses base SD pipeline.
2. **Face-likeness** (GPU only) — reference image uploaded. Extracts face embedding + landmarks via custom ONNX models, feeds them through InstantID (ControlNet + IP-Adapter) to preserve identity in the generated image.

### Device-Aware Configuration (`config.py`)

Automatically selects model variant based on hardware:

| Setting | CPU (dev) | CUDA (prod) |
|---------|-----------|-------------|
| Base model | SD 1.5 | SDXL |
| Image size | 512x512 | 1024x1024 |
| Inference steps | 25 | 30 |
| Dtype | float32 | float16 |
| Face method | None | InstantID (ControlNet + IP-Adapter) |

Override with `DEVICE_MODE=cpu` or `DEVICE_MODE=cuda` environment variable.

## Model Architecture

### InstantID (GPU only)

Face-likeness preservation using two parallel signals:
1. **Face embedding** — 512-dim identity vector from ArcFace (raw, unnormalized, norm ~23). Projected by a Perceiver Resampler (4 layers, 16 latent queries, 512→2048) then injected via IP-Adapter decoupled cross-attention (70 processors in SDXL UNet).
2. **Face keypoints** — 5-point landmarks (eyes, nose, mouth corners) drawn on a blank canvas. Fed to a ControlNet for spatial/pose guidance.

The `faceStrength` parameter (0.0-1.0) controls both `ip_adapter_scale` (identity) and `controlnet_conditioning_scale` (pose). Higher values = closer likeness, lower values = more creative freedom.

**Important:** ArcFace embeddings must NOT be L2-normalized. The Resampler expects raw embeddings with norm ~20-25. Normalizing to unit norm reduces the face signal by ~23x.

### Face Detection (`face_analysis.py`)

Custom ONNX module replacing the `insightface` Python package (which requires MSVC on Windows):
- **SCRFD** (`scrfd_10g_bnkps.onnx`) — face detection with 5 landmarks
- **ArcFace** (`glintr100.onnx`) — 512-dim face embedding extraction
- Both run via `onnxruntime` (CUDA or CPU providers)

### Model Downloads

| Component | Size | Used by |
|-----------|------|---------|
| SD 1.5 base | ~2 GB | CPU mode |
| SDXL base | ~6.9 GB | GPU mode |
| InstantID ControlNet | ~1.7 GB | GPU mode |
| InstantID IP-Adapter (`ip-adapter.bin`) | ~1.7 GB | GPU mode |
| InsightFace antelopev2 (SCRFD + ArcFace) | ~300 MB | GPU mode |

## Client

**File:** `client/src/App.jsx` — single React component

### State

| State | Type | Purpose |
|-------|------|---------|
| `subject` | string | Main image description text |
| `styleKeywords` | string | Style modifier keywords |
| `selectedPreset` | object/null | Currently active style preset |
| `mood` | string | Selected mood/atmosphere |
| `avoid` | string | Negative prompt (things to exclude) |
| `imageUrl` | string | Generated image URL (`/generated/<file>.jpg`) |
| `loading` | boolean | Request in-flight flag |
| `error` | string | Error message display |
| `generatedPrompt` | string | Final assembled prompt shown to user |
| `referenceImage` | File/null | Uploaded face reference photo |
| `imagePreview` | string | Data URI preview of reference image |
| `faceStrength` | number | Face likeness strength (0.0-1.0) |

### UI Sections

1. **Subject** — textarea for the main image description
2. **Reference Image** — drag-and-drop / file picker with preview
3. **Face Strength** — range slider (appears when reference image is uploaded)
4. **Style Presets** — 12 clickable buttons with curated keywords
5. **Style Keywords** — freeform text input
6. **Mood / Atmosphere** — 8 toggleable pill buttons
7. **What to Avoid** — text input for negative prompt terms
8. **Generate Image** — button that assembles the prompt and calls the API

### Data Flow

```
User fills form
    ↓
buildPrompt() assembles final prompt string
    ↓
POST /api/generate  (FormData: prompt, referenceImage?, faceStrength?)
    ↓
FastAPI validates → pipeline.generate_*() → Stable Diffusion inference
    ↓
Image saved as JPEG to server/uploads/ → returns { imageUrl: "/generated/<file>.jpg" }
    ↓
Displayed in <img> tag (served by FastAPI StaticFiles mount)
```

## Dependencies

### Server (Python)

| Package | Purpose |
|---------|---------|
| fastapi | HTTP server framework |
| uvicorn | ASGI server |
| python-multipart | Form/file upload parsing |
| torch | PyTorch deep learning framework |
| diffusers | HuggingFace diffusion model pipelines |
| transformers | Tokenizer/text encoder support |
| accelerate | Model offloading and memory optimization |
| opencv-python-headless | Face keypoint drawing, image alignment |
| einops | Tensor reshaping (used by IP-Adapter Resampler) |
| onnxruntime | ONNX model inference for face detection/recognition |
| Pillow | Image processing |
| huggingface-hub | Model downloading |
| truststore | Corporate SSL inspection bypass |

### Client (Node.js)

| Package | Version | Purpose |
|---------|---------|---------|
| react | ^18.2.0 | UI framework |
| react-dom | ^18.2.0 | DOM rendering |
| vite | ^5.0.8 | Build tool / dev server |
| @vitejs/plugin-react | ^4.2.1 | React JSX support |

## Running the Application

```bash
# 1. Set up Python environment
cd server && python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download models (~4 GB for CPU)
python ../scripts/download_models.py

# 3. Start backend
uvicorn main:app --host 0.0.0.0 --port 3001

# 4. Start frontend (new terminal)
cd client && npm install && npm run dev

# 5. Open http://localhost:5173 (dev) or http://localhost:3001 (prod)
```
