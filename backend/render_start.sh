#!/usr/bin/env bash
# =====================================================================
# Render Start Script — Single Service, Both Backends
# rootDir: backend/
#
# Render assigns $PORT for the main web process (HFT backend).
# api_server runs on PORT+1 (ML predictions, internal only).
# Frontend calls HFT backend ($PORT) for everything including proxied ML.
# =====================================================================
set -e

HFT_PORT="${PORT:-10000}"
API_PORT=$((HFT_PORT + 1))

# Point api_server at HFT backend on same machine
export HFT2_BACKEND_URL="http://127.0.0.1:${HFT_PORT}"
export PORT="${API_PORT}"  # api_server reads PORT env var via config.py

# Writable dirs on Render (ephemeral filesystem)
export DATA_DIR="/tmp/data"
export LOGS_DIR="/tmp/logs"
mkdir -p /tmp/logs /tmp/data/cache /tmp/data/features /tmp/data/logs /tmp/stock_analysis

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo " Trade Bot Unified Backend"
echo " HFT backend (main) -> port ${HFT_PORT}"
echo " ML  backend (side) -> port ${API_PORT}"
echo "=========================================="

# Start ML api_server in background (predictions only)
echo "[api_server] Starting on port ${API_PORT}..."
cd "${SCRIPT_DIR}"
python api_server.py &
API_PID=$!

# Small delay to avoid port race
sleep 3

# Restore PORT to HFT port — uvicorn reads it here
export PORT="${HFT_PORT}"

# Start HFT backend (foreground — Render health checks this)
echo "[web_backend] Starting on port ${HFT_PORT}..."
cd "${SCRIPT_DIR}/hft2/backend"
exec uvicorn web_backend:app --host 0.0.0.0 --port "${HFT_PORT}" --workers 1
