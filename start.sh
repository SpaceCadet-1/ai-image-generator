#!/bin/bash
# start.sh - Start the server (backend + built frontend on port 3001)
# Usage: bash start.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "================================="
echo "  AI Image Generator - Starting"
echo "================================="
echo ""

# Build frontend if not already built
if [ ! -d "$REPO_DIR/client/dist" ]; then
    echo "Building frontend..."
    cd "$REPO_DIR/client"
    npm run build
fi

# Start FastAPI (serves both API and frontend)
cd "$REPO_DIR/server"
echo "[SERVER] Starting on http://localhost:3001"
echo ""
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 3001
