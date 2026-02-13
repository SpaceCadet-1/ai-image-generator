# AI Image Generator - Architecture Reference

## Stack

| Layer | Technology | Port |
|-------|-----------|------|
| Frontend | React 18 + Vite 5 | 3000 |
| Backend | Python FastAPI + Uvicorn | 3001 |
| AI (CPU) | Stable Diffusion 1.5 + IP-Adapter FaceID Plus V2 | — |
| AI (GPU) | Stable Diffusion XL + IP-Adapter FaceID Plus V2 | — |
| Face Detection | InsightFace (antelopev2, ONNX) | — |

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
│   ├── models/                 # Downloaded weights (gitignored)
│   │   ├── diffusers/          # HuggingFace model cache
│   │   ├── ip-adapter-faceid/  # FaceID adapter + LoRA weights
│   │   └── insightface/        # InsightFace ONNX models
│   └── uploads/                # Temp upload directory (gitignored)
├── scripts/
│   └── download_models.py      # Downloads all models from HuggingFace
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
| `faceStrength` | float | no | Face likeness strength 0.0-1.0 (default 0.6) |

**Success response (200):**
```json
{ "imageUrl": "data:image/png;base64,..." }
```

**Error responses:**
- `400` — prompt missing, no face detected, invalid file type/size
- `500` — model inference failure
- Response body: `{ "detail": "<error message>" }`

### Generation Modes

1. **Text-to-image** — no reference image uploaded. Uses base SD pipeline.
2. **Face-likeness** — reference image uploaded. Extracts face embedding via InsightFace, feeds it through IP-Adapter FaceID Plus V2 to preserve identity in the generated image.

### Device-Aware Configuration (`config.py`)

Automatically selects model variant based on hardware:

| Setting | CPU (dev) | CUDA (prod) |
|---------|-----------|-------------|
| Base model | SD 1.5 | SDXL |
| Image size | 512x512 | 1024x1024 |
| Inference steps | 25 | 30 |
| Dtype | float32 | float16 |
| Face adapter | sd15 variant | sdxl variant |

Override with `DEVICE_MODE=cpu` or `DEVICE_MODE=cuda` environment variable.

## Model Architecture

### IP-Adapter FaceID Plus V2

Combines two signals for face-likeness preservation:
1. **InsightFace embedding** — 512-dim face identity vector from antelopev2 ONNX model
2. **CLIP image features** — visual features from CLIP ViT-H-14 encoder

These are injected into the diffusion process via:
- **LoRA weights** — fine-tuned cross-attention layers (fused at scale 0.5)
- **IP-Adapter** — decoupled cross-attention for image prompt conditioning

The `faceStrength` parameter (0.0-1.0) controls the IP-Adapter scale — higher values produce closer likeness but less prompt adherence.

### Model Downloads

| Component | Size | Used by |
|-----------|------|---------|
| SD 1.5 base | ~2 GB | CPU mode |
| SDXL base | ~6.9 GB | GPU mode |
| IP-Adapter FaceID Plus V2 (SD 1.5) | ~208 MB | CPU mode |
| IP-Adapter FaceID Plus V2 (SDXL) | ~400 MB | GPU mode |
| CLIP ViT-H-14 image encoder | ~1.7 GB | Both |
| InsightFace antelopev2 | ~300 MB | Both |

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
| `imageUrl` | string | Generated image (base64 data URI) |
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
POST http://localhost:3001/api/generate  (FormData: prompt, referenceImage?, faceStrength?)
    ↓
FastAPI validates → pipeline.generate_*() → Stable Diffusion inference
    ↓
Returns { imageUrl: "data:image/png;base64,..." } → displayed in <img> tag
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
| transformers | CLIP model loading |
| accelerate | Model offloading and memory optimization |
| insightface | Face detection and embedding extraction |
| onnxruntime | ONNX model inference for InsightFace |
| Pillow | Image processing |
| huggingface-hub | Model downloading |

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

# 5. Open http://localhost:3000
```
