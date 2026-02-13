#!/bin/bash
# start.sh — Start backend + frontend on Linux/AWS
# Usage: bash start.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$REPO_DIR/.pids"

echo ""
echo "================================="
echo "  AI Image Generator — Starting"
echo "================================="
echo ""

# Start FastAPI backend (uses venv)
cd "$REPO_DIR/server"
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 3001 &
API_PID=$!
echo "[API] FastAPI started (PID $API_PID) on http://localhost:3001"

# Start Vite frontend
cd "$REPO_DIR/client"
npx vite --host 0.0.0.0 &
WEB_PID=$!
echo "[WEB] Vite started (PID $WEB_PID) on http://localhost:3000"

# Save PIDs for stop.sh
echo "$API_PID" > "$PIDFILE"
echo "$WEB_PID" >> "$PIDFILE"

echo ""
echo "PIDs saved to .pids — run 'bash stop.sh' to stop."
echo ""
echo "Connect from your laptop:"
echo "  .\\scripts\\connect.ps1 <instance-id>"
echo "  Then open http://localhost:3000"
echo ""

# Wait for both — Ctrl+C kills this script, then stop.sh cleans up
trap "bash $REPO_DIR/stop.sh; exit 0" INT TERM
wait
