#!/usr/bin/env bash
set -e
PORT="${PORT:-10000}"
exec uvicorn api_server:app --host 0.0.0.0 --port "$PORT"
