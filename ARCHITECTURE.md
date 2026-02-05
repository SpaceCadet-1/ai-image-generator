# AI Image Generator - Architecture Reference

## Stack

| Layer | Technology | Port |
|-------|-----------|------|
| Frontend | React 18 + Vite 5 | 3000 |
| Backend | Express 4 (ES modules) | 3001 |
| AI | OpenAI SDK v4 → DALL-E 3 | — |

No database, authentication, or state persistence.

## Project Structure

```
ai-image-generator/
├── .gitignore               # node_modules, .env, dist, .DS_Store, *.log
├── README.md                # Setup instructions
├── ARCHITECTURE.md          # This file
├── client/                  # React frontend (Vite)
│   ├── package.json         # react, react-dom, vite, @vitejs/plugin-react
│   ├── vite.config.js       # Dev server on port 3000
│   ├── index.html           # HTML shell
│   └── src/
│       ├── main.jsx         # Entry point — renders <App /> into #root
│       ├── App.jsx          # Single-component app with prompt builder UI
│       └── App.css          # Dark theme, gradient styling, responsive
└── server/                  # Express backend
    ├── package.json         # express, cors, dotenv, openai
    ├── index.js             # API server with single endpoint
    └── .env                 # OPENAI_API_KEY (gitignored)
```

## Server

**File:** `server/index.js`

### Endpoint

`POST /api/generate`

**Request body:**
```json
{ "prompt": "a sunset over mountains, watercolor style" }
```

**Success response (200):**
```json
{ "imageUrl": "https://oaidalleapiprodscus.blob.core.windows.net/..." }
```

**Error responses:**
- `400` — prompt missing or empty: `{ "error": "Prompt is required" }`
- `500` — OpenAI API failure: `{ "error": "<error message>" }`

### Configuration

- CORS enabled for all origins
- OpenAI client initialized from `process.env.OPENAI_API_KEY`
- DALL-E 3 model, 1024x1024 resolution, 1 image per request
- Image URLs returned by OpenAI are temporary (~1 hour expiry)

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
| `imageUrl` | string | Generated image URL from API |
| `loading` | boolean | Request in-flight flag |
| `error` | string | Error message display |
| `generatedPrompt` | string | Final assembled prompt shown to user |

### UI Sections

1. **Subject** — textarea for the main image description
2. **Style Presets** — 12 clickable buttons, each maps to curated keywords:
   - Photorealistic, Digital Art, Watercolor, Oil Painting, Anime, Cinematic, 3D Render, Pixel Art, Sketch, Fantasy, Minimalist, Vintage
3. **Style Keywords** — freeform text input; editing clears the active preset
4. **Mood / Atmosphere** — 8 toggleable pill buttons:
   - Bright & Cheerful, Dark & Moody, Peaceful & Calm, Dramatic & Intense, Mysterious, Whimsical & Playful, Elegant & Sophisticated, Gritty & Raw
5. **What to Avoid** — text input for negative prompt terms
6. **Generate Image** — button that assembles the prompt and calls the API
7. **Results** — displays the assembled prompt text and rendered image

### Prompt Assembly

`buildPrompt()` concatenates parts with commas:
```
<subject>, <styleKeywords>, <mood> mood
```
If "avoid" text is present, appends:
```
. Without: <avoid text>
```

Example output:
```
a castle on a hill, watercolor painting, soft edges, artistic, painted, dark & moody mood. Without: people, text
```

### Data Flow

```
User fills form
    ↓
buildPrompt() assembles final prompt string
    ↓
POST http://localhost:3001/api/generate  { prompt }
    ↓
Express validates → OpenAI DALL-E 3 API call
    ↓
Returns { imageUrl } → displayed in <img> tag
```

## Styling

**File:** `client/src/App.css`

- Dark gradient background: `#1a1a2e → #16213e`
- Gradient heading: cyan (`#00d4ff`) → purple (`#7b2cbf`)
- Glassmorphic form card with subtle border
- Preset buttons: grid layout, highlight on selection
- Mood buttons: pill-shaped, toggle on/off
- Generate button: full-width gradient with hover lift effect
- Image output: rounded corners with cyan glow shadow
- Responsive breakpoint at 600px (3-column preset grid, smaller padding)

## Running the Application

```bash
# 1. Install dependencies
cd server && npm install
cd ../client && npm install

# 2. Configure API key
# Create server/.env with: OPENAI_API_KEY=sk-...

# 3. Start backend (terminal 1)
cd server && npm start

# 4. Start frontend (terminal 2)
cd client && npm run dev

# 5. Open http://localhost:3000
```

## Dependencies

### Server
| Package | Version | Purpose |
|---------|---------|---------|
| express | ^4.18.2 | HTTP server |
| cors | ^2.8.5 | Cross-origin requests |
| dotenv | ^16.3.1 | Environment variable loading |
| openai | ^4.20.1 | OpenAI API client |

### Client
| Package | Version | Purpose |
|---------|---------|---------|
| react | ^18.2.0 | UI framework |
| react-dom | ^18.2.0 | DOM rendering |
| vite | ^5.0.8 | Build tool / dev server |
| @vitejs/plugin-react | ^4.2.1 | React JSX support |
