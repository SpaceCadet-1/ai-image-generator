import express from 'express';
import cors from 'cors';
import OpenAI, { toFile } from 'openai';
import dotenv from 'dotenv';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const uploadsDir = path.join(__dirname, 'uploads');

if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

const app = express();
const PORT = 3001;
const LOCAL_SERVER = 'http://localhost:8000';

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

const storage = multer.diskStorage({
  destination: uploadsDir,
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${Math.round(Math.random() * 1e9)}${path.extname(file.originalname)}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ['image/png', 'image/jpeg', 'image/webp'];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Only PNG, JPG, and WebP files are allowed'));
    }
  },
});

// --- OpenAI generation helper ---
async function generateOpenAI(prompt, filePath, originalname, mimetype) {
  if (!openai) {
    throw new Error('OpenAI API key is not configured. Set OPENAI_API_KEY in server/.env');
  }
  if (filePath) {
    const imageFile = await toFile(
      fs.createReadStream(filePath),
      originalname,
      { type: mimetype }
    );
    const response = await openai.images.edit({
      model: 'gpt-image-1',
      image: imageFile,
      prompt,
      n: 1,
      size: '1024x1024',
    });
    const base64 = response.data[0].b64_json;
    return `data:image/png;base64,${base64}`;
  }

  const response = await openai.images.generate({
    model: 'dall-e-3',
    prompt,
    n: 1,
    size: '1024x1024',
  });
  return response.data[0].url;
}

// --- Local SDXL generation helper ---
const SDXL_QUALITY_SUFFIX = ', masterpiece, best quality, highly detailed, sharp focus, professional';
const SDXL_DEFAULT_NEGATIVE = 'worst quality, low quality, blurry, deformed, distorted, disfigured, bad anatomy, watermark, text, signature';

async function generateLocal(prompt, filePath) {
  // Split "Without: ..." into negative_prompt
  let positivePrompt = prompt;
  let negativePrompt = '';
  const withoutMatch = prompt.match(/\.\s*Without:\s*(.+)$/i);
  if (withoutMatch) {
    negativePrompt = withoutMatch[1].trim();
    positivePrompt = prompt.slice(0, withoutMatch.index).trim();
  }

  // Boost SDXL prompts with quality keywords
  positivePrompt += SDXL_QUALITY_SUFFIX;
  negativePrompt = negativePrompt
    ? `${SDXL_DEFAULT_NEGATIVE}, ${negativePrompt}`
    : SDXL_DEFAULT_NEGATIVE;

  const endpoint = filePath ? '/generate-img2img' : '/generate';
  const formData = new FormData();
  formData.append('prompt', positivePrompt);
  formData.append('negative_prompt', negativePrompt);
  if (filePath) {
    formData.append('image', new Blob([fs.readFileSync(filePath)]), 'image.png');
  }

  let resp;
  try {
    resp = await fetch(`${LOCAL_SERVER}${endpoint}`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(120_000), // 2 minute timeout
    });
  } catch (err) {
    if (err.name === 'TimeoutError') {
      throw new Error('Local GPU generation timed out after 2 minutes. The model may be overloaded — try again.');
    }
    throw new Error(
      `Cannot connect to local GPU server at ${LOCAL_SERVER}. Is it running? Start it with: cd local-server && .venv\\Scripts\\python run.py`
    );
  }
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'Local generation failed');
  return data.imageUrl;
}

// --- Local server health proxy ---
app.get('/api/local-status', async (req, res) => {
  try {
    const resp = await fetch(`${LOCAL_SERVER}/health`);
    const data = await resp.json();
    res.json(data);
  } catch {
    res.json({ status: 'offline' });
  }
});

// --- Main generate endpoint ---
app.post('/api/generate', upload.single('referenceImage'), async (req, res) => {
  const prompt = req.body.prompt;
  const mode = req.body.mode || 'api';

  if (!prompt || prompt.trim() === '') {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  let tempFilePath = req.file?.path;

  try {
    let imageUrl;

    if (mode === 'local') {
      imageUrl = await generateLocal(prompt, tempFilePath);
    } else {
      imageUrl = await generateOpenAI(
        prompt,
        tempFilePath,
        req.file?.originalname,
        req.file?.mimetype
      );
    }

    res.json({ imageUrl });
  } catch (error) {
    console.error('Error generating image:', error);
    res.status(500).json({
      error: error.message || 'Failed to generate image',
    });
  } finally {
    if (tempFilePath) {
      fs.unlink(tempFilePath, (err) => {
        if (err) console.error('Failed to clean up temp file:', err);
      });
    }
  }
});

// Multer error handling
app.use((err, req, res, next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File is too large. Maximum size is 25MB.' });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err.message === 'Only PNG, JPG, and WebP files are allowed') {
    return res.status(400).json({ error: err.message });
  }
  next(err);
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
