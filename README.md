# AI Image Generator

A web application that generates images from text descriptions using OpenAI's DALL-E API.

## Setup

### 1. Install dependencies

```bash
# Install server dependencies
cd server
npm install

# Install client dependencies
cd ../client
npm install
```

### 2. Configure API key

Edit `server/.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the application

Start the backend server:
```bash
cd server
npm start
```

Start the frontend (in a new terminal):
```bash
cd client
npm run dev
```

### 4. Use the application

1. Open http://localhost:3000 in your browser
2. Enter an image description in the textarea
3. Click "Generate Image"
4. Wait for the AI to create your image

## Project Structure

```
ai-image-generator/
├── client/           # React frontend (Vite)
│   └── src/
│       ├── App.jsx   # Main component
│       └── App.css   # Styles
└── server/           # Express backend
    ├── index.js      # API server
    └── .env          # API key (not in git)
```
