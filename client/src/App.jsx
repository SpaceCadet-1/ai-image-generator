import { useState } from 'react';

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

  const handlePresetClick = (preset) => {
    if (selectedPreset?.name === preset.name) {
      setSelectedPreset(null);
      setStyleKeywords('');
    } else {
      setSelectedPreset(preset);
      setStyleKeywords(preset.keywords);
    }
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
      const response = await fetch('http://localhost:3001/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
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

  return (
    <div className="container">
      <h1>AI Image Generator</h1>
      <p className="subtitle">Fine-tune your image with detailed controls</p>

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
          {loading ? 'Generating...' : 'Generate Image'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Creating your image...</p>
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
