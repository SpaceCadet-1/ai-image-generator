#!/bin/bash
# start.sh — Start backend + frontend on Linux/AWS
# Usage: bash start.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$REPO_DIR/.pids"

# Activate conda pytorch env (Deep Learning AMI)
source activate pytorch 2>/dev/null || conda activate pytorch 2>/dev/null || true

echo ""
echo "================================="
echo "  AI Image Generator — Starting"
echo "================================="
echo ""

# Start FastAPI backend
cd "$REPO_DIR/server"
python -m uvicorn main:app --host 0.0.0.0 --port 3001 &
API_PID=$!
echo "[API] FastAPI started (PID $API_PID) on http://localhost:3001"

# Start Vite frontend
cd "$REPO_DIR/client"
npx vite --host 0.0.0.0 &
WEB_PID=$!
echo "[WEB] Vite started (PID $WEB_PID) on http://localhost:5173"

# Save PIDs for stop.sh
echo "$API_PID" > "$PIDFILE"
echo "$WEB_PID" >> "$PIDFILE"

echo ""
echo "PIDs saved to .pids — run 'bash stop.sh' to stop."
echo ""
echo "From your laptop:"
echo "  ssh -i <key>.pem -L 5173:localhost:5173 -L 3001:localhost:3001 ubuntu@<ip>"
echo "  Then open http://localhost:5173"
echo ""

# Wait for both — Ctrl+C kills this script, then stop.sh cleans up
trap "bash $REPO_DIR/stop.sh; exit 0" INT TERM
wait
