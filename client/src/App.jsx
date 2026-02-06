import { useState, useRef, useEffect } from 'react';

const STYLE_PRESETS = [
  { name: 'Photorealistic', keywords: 'photorealistic, highly detailed, 8k, professional photography' },
  { name: 'Digital Art', keywords: 'digital art, vibrant colors, detailed illustration' },
  { name: 'Watercolor', keywords: 'watercolor painting, soft edges, artistic, painted' },
  { name: 'Oil Painting', keywords: 'oil painting, classical art style, textured brushstrokes' },
  { name: 'Anime', keywords: 'anime style, manga, Japanese animation, cel shaded' },
  { name: 'Cinematic', keywords: 'cinematic, movie scene, dramatic lighting, film still' },
  { name: '3D Render', keywords: '3D render, octane render, CGI, highly detailed' },
  { name: 'Pixel Art', keywords: 'pixel art, retro game style, 16-bit, pixelated' },
  { name: 'Sketch', keywords: 'pencil sketch, hand drawn, detailed line art' },
  { name: 'Fantasy', keywords: 'fantasy art, magical, ethereal, dreamlike' },
  { name: 'Minimalist', keywords: 'minimalist, simple, clean lines, modern design' },
  { name: 'Vintage', keywords: 'vintage, retro, nostalgic, aged photograph' },
];

const MOOD_OPTIONS = [
  'Bright & Cheerful',
  'Dark & Moody',
  'Peaceful & Calm',
  'Dramatic & Intense',
  'Mysterious',
  'Whimsical & Playful',
  'Elegant & Sophisticated',
  'Gritty & Raw',
];

const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const MAX_FILE_SIZE = 25 * 1024 * 1024;

function App() {
  const [subject, setSubject] = useState('');
  const [styleKeywords, setStyleKeywords] = useState('');
  const [selectedPreset, setSelectedPreset] = useState(null);
  const [mood, setMood] = useState('');
  const [avoid, setAvoid] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [referenceImage, setReferenceImage] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [mode, setMode] = useState('api');
  const [localStatus, setLocalStatus] = useState('offline');
  const fileInputRef = useRef(null);

  useEffect(() => {
    const checkLocalStatus = async () => {
      try {
        const resp = await fetch('http://localhost:3001/api/local-status');
        const data = await resp.json();
        setLocalStatus(data.status);
      } catch {
        setLocalStatus('offline');
      }
    };
    checkLocalStatus();
    const interval = setInterval(checkLocalStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handlePresetClick = (preset) => {
    if (selectedPreset?.name === preset.name) {
      setSelectedPreset(null);
      setStyleKeywords('');
    } else {
      setSelectedPreset(preset);
      setStyleKeywords(preset.keywords);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Only PNG, JPG, and WebP files are allowed');
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError('File is too large. Maximum size is 25MB.');
      return;
    }

    setError('');
    setReferenceImage(file);

    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setReferenceImage(null);
    setImagePreview('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const buildPrompt = () => {
    let parts = [];

    if (subject.trim()) {
      parts.push(subject.trim());
    }

    if (styleKeywords.trim()) {
      parts.push(styleKeywords.trim());
    }

    if (mood) {
      parts.push(`${mood.toLowerCase()} mood`);
    }

    let prompt = parts.join(', ');

    if (avoid.trim()) {
      prompt += `. Without: ${avoid.trim()}`;
    }

    return prompt;
  };

  const handleGenerate = async () => {
    const prompt = buildPrompt();

    if (!prompt) {
      setError('Please describe what you want to generate');
      return;
    }

    setLoading(true);
    setError('');
    setImageUrl('');
    setGeneratedPrompt(prompt);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('mode', mode);
      if (referenceImage) {
        formData.append('referenceImage', referenceImage);
      }

      const response = await fetch('http://localhost:3001/api/generate', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate image');
      }

      setImageUrl(data.imageUrl);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getModelNote = () => {
    if (mode === 'local') {
      return referenceImage
        ? 'Using SDXL img2img (local GPU)'
        : 'Using SDXL text-to-image (local GPU)';
    }
    return referenceImage ? 'Using gpt-image-1 (image edit mode)' : null;
  };

  const modelNote = getModelNote();

  return (
    <div className="container">
      <h1>AI Image Generator</h1>
      <p className="subtitle">Fine-tune your image with detailed controls</p>

      <div className="mode-toggle">
        <button
          className={`mode-btn ${mode === 'api' ? 'active' : ''}`}
          onClick={() => setMode('api')}
          disabled={loading}
        >
          OpenAI API
        </button>
        <button
          className={`mode-btn ${mode === 'local' ? 'active' : ''}`}
          onClick={() => setMode('local')}
          disabled={loading || localStatus === 'offline'}
          title={localStatus === 'offline' ? 'Local GPU server is not running' : ''}
        >
          Local GPU
        </button>
        <span className={`mode-status ${localStatus}`}>
          {localStatus === 'ready' ? 'GPU Ready' :
           localStatus === 'loading' ? 'Loading Model...' :
           'GPU Offline'}
        </span>
      </div>

      <div className="form-section">
        <div className="input-group">
          <label>What do you want to see?</label>
          <textarea
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            placeholder="A golden retriever playing in autumn leaves..."
            rows={3}
            disabled={loading}
          />
        </div>

        <div className="input-group">
          <label>Reference Image (optional)</label>
          <input
            type="file"
            ref={fileInputRef}
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => handleFileSelect(e.target.files[0])}
            style={{ display: 'none' }}
          />
          {!imagePreview ? (
            <div
              className="upload-area"
              onClick={() => !loading && fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <span className="upload-icon">+</span>
              <p>Drag & drop an image here, or click to browse</p>
              <p className="upload-hint">PNG, JPG, or WebP — max 25MB</p>
            </div>
          ) : (
            <div className="preview-container">
              <img src={imagePreview} alt="Reference preview" className="reference-preview" />
              <button
                className="remove-image-btn"
                onClick={handleRemoveImage}
                disabled={loading}
                title="Remove image"
              >
                &times;
              </button>
            </div>
          )}
          {modelNote && (
            <p className="model-note">{modelNote}</p>
          )}
        </div>

        <div className="input-group">
          <label>Style Presets</label>
          <div className="presets-grid">
            {STYLE_PRESETS.map((preset) => (
              <button
                key={preset.name}
                className={`preset-btn ${selectedPreset?.name === preset.name ? 'active' : ''}`}
                onClick={() => handlePresetClick(preset)}
                disabled={loading}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        <div className="input-group">
          <label>Style Keywords</label>
          <input
            type="text"
            value={styleKeywords}
            onChange={(e) => {
              setStyleKeywords(e.target.value);
              setSelectedPreset(null);
            }}
            placeholder="Add your own style keywords..."
            disabled={loading}
          />
        </div>

        <div className="input-group">
          <label>Mood / Atmosphere</label>
          <div className="mood-options">
            {MOOD_OPTIONS.map((m) => (
              <button
                key={m}
                className={`mood-btn ${mood === m ? 'active' : ''}`}
                onClick={() => setMood(mood === m ? '' : m)}
                disabled={loading}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="input-group">
          <label>What to Avoid</label>
          <input
            type="text"
            value={avoid}
            onChange={(e) => setAvoid(e.target.value)}
            placeholder="blurry, text, watermarks, extra limbs..."
            disabled={loading}
          />
        </div>

        <button className="generate-btn" onClick={handleGenerate} disabled={loading}>
          {loading
            ? mode === 'local' ? 'Generating on GPU...' : 'Generating...'
            : 'Generate Image'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>{mode === 'local' ? 'Running SDXL on your GPU...' : 'Creating your image...'}</p>
        </div>
      )}

      {generatedPrompt && !loading && (
        <div className="prompt-preview">
          <strong>Generated Prompt:</strong> {generatedPrompt}
        </div>
      )}

      {imageUrl && (
        <div className="image-section">
          <img src={imageUrl} alt={subject} />
        </div>
      )}
    </div>
  );
}

export default App;
