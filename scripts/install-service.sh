#!/bin/bash
# install-service.sh - Install the app as a systemd service
# Usage: sudo bash scripts/install-service.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cat > /etc/systemd/system/ai-image-generator.service <<EOF
[Unit]
Description=AI Image Generator (FastAPI + SDXL)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$REPO_DIR/server
ExecStart=$REPO_DIR/server/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 3001
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-image-generator
systemctl start ai-image-generator

echo ""
echo "Service installed and started."
echo ""
echo "Commands:"
echo "  sudo systemctl status ai-image-generator   # check status"
echo "  sudo journalctl -u ai-image-generator -f    # view logs"
echo "  sudo systemctl restart ai-image-generator   # restart"
echo "  sudo systemctl stop ai-image-generator      # stop"
echo ""
