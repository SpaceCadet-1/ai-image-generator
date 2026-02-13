#!/bin/bash
# stop.sh — Stop backend + frontend on Linux/AWS
# Usage: bash stop.sh

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$REPO_DIR/.pids"

echo ""
echo "Stopping AI Image Generator..."

stopped=0

# Kill from saved PIDs
if [ -f "$PIDFILE" ]; then
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            ((stopped++))
        fi
    done < "$PIDFILE"
    rm -f "$PIDFILE"
fi

# Also kill anything on ports 3001 and 5173 (fallback)
for port in 3001 5173; do
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing process on port $port (PID $pid)"
            kill "$pid" 2>/dev/null || true
            ((stopped++))
        fi
    done
done

if [ "$stopped" -eq 0 ]; then
    echo "  No running processes found."
else
    echo "  Stopped $stopped process(es)."
fi
echo ""
