#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/demo-app/backend"
FRONTEND_DIR="$ROOT_DIR/demo-app/frontend"

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Frontend dependencies missing. Run: cd $FRONTEND_DIR && npm install"
  exit 1
fi

if [ ! -d "$BACKEND_DIR/.venv" ]; then
  echo "Backend virtualenv missing. Run setup from demo-app/docs/RUNBOOK.md"
  exit 1
fi

if [ ! -f "$BACKEND_DIR/.venv/bin/activate" ]; then
  echo "Backend virtualenv is invalid: $BACKEND_DIR/.venv"
  exit 1
fi

echo "Starting backend on http://127.0.0.1:8000"
(
  cd "$BACKEND_DIR"
  source .venv/bin/activate
  uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) &
BACK_PID=$!

echo "Starting frontend on http://127.0.0.1:5173"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --port 5173
) &
FRONT_PID=$!

cleanup() {
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$BACK_PID" "$FRONT_PID"
