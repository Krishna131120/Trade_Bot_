#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-5000}"
exec python web_backend.py --host 0.0.0.0 --port "$PORT"
