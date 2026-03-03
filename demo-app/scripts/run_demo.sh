#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="$ROOT_DIR/demo-app/backend"
FRONTEND_DIR="$ROOT_DIR/demo-app/frontend"

HOST="${DEMO_HOST:-127.0.0.1}"
BACKEND_PORT="${DEMO_BACKEND_PORT:-8000}"
FRONTEND_PORT="${DEMO_FRONTEND_PORT:-5173}"

STATE_DIR="${TMPDIR:-/tmp}/angiography-demo"
BACKEND_PID_FILE="$STATE_DIR/backend.pid"
FRONTEND_PID_FILE="$STATE_DIR/frontend.pid"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|restart|status]

Commands:
  start    Start backend + frontend (default)
  stop     Stop demo processes started from this repo
  restart  Stop then start
  status   Show process and endpoint status

Environment overrides:
  DEMO_HOST
  DEMO_BACKEND_PORT
  DEMO_FRONTEND_PORT
EOF
}

is_running_pid() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

listeners_on_port() {
  local port="$1"
  lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true
}

port_is_free() {
  local port="$1"
  [ -z "$(listeners_on_port "$port")" ]
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local timeout_seconds="${3:-30}"
  local elapsed=0

  while [ "$elapsed" -lt "$timeout_seconds" ]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  echo "$label did not become ready within ${timeout_seconds}s: $url"
  return 1
}

kill_pid_gracefully() {
  local pid="$1"
  local label="$2"
  local elapsed=0

  if ! is_running_pid "$pid"; then
    return 0
  fi

  echo "Stopping $label (PID $pid)"
  kill "$pid" 2>/dev/null || true

  while is_running_pid "$pid" && [ "$elapsed" -lt 8 ]; do
    sleep 1
    elapsed=$((elapsed + 1))
  done

  if is_running_pid "$pid"; then
    echo "Force stopping $label (PID $pid)"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

known_backend_pids() {
  pgrep -f "$BACKEND_DIR/.venv/bin/uvicorn app.main:app --host $HOST --port $BACKEND_PORT" || true
}

known_frontend_pids() {
  pgrep -f "$FRONTEND_DIR/node_modules/.bin/vite --strictPort --host $HOST --port $FRONTEND_PORT" || true
}

ensure_prereqs() {
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "Frontend dependencies missing. Run: cd $FRONTEND_DIR && npm install"
    exit 1
  fi

  if [ ! -d "$BACKEND_DIR/.venv" ] || [ ! -f "$BACKEND_DIR/.venv/bin/activate" ]; then
    echo "Backend virtualenv missing or invalid. See demo-app/docs/RUNBOOK.md"
    exit 1
  fi
}

stop_demo() {
  mkdir -p "$STATE_DIR"

  if [ -f "$BACKEND_PID_FILE" ]; then
    kill_pid_gracefully "$(cat "$BACKEND_PID_FILE")" "backend"
  fi
  if [ -f "$FRONTEND_PID_FILE" ]; then
    kill_pid_gracefully "$(cat "$FRONTEND_PID_FILE")" "frontend"
  fi

  for pid in $(known_backend_pids); do
    kill_pid_gracefully "$pid" "backend"
  done
  for pid in $(known_frontend_pids); do
    kill_pid_gracefully "$pid" "frontend"
  done

  rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
}

show_status() {
  local backend_url="http://$HOST:$BACKEND_PORT/api/health"
  local models_url="http://$HOST:$BACKEND_PORT/api/models"
  local frontend_url="http://$HOST:$FRONTEND_PORT"

  echo "Ports:"
  if port_is_free "$BACKEND_PORT"; then
    echo "- backend port $BACKEND_PORT: free"
  else
    echo "- backend port $BACKEND_PORT: in use by PID(s) $(listeners_on_port "$BACKEND_PORT" | tr '\n' ' ')"
  fi

  if port_is_free "$FRONTEND_PORT"; then
    echo "- frontend port $FRONTEND_PORT: free"
  else
    echo "- frontend port $FRONTEND_PORT: in use by PID(s) $(listeners_on_port "$FRONTEND_PORT" | tr '\n' ' ')"
  fi

  echo "Endpoints:"
  if curl -fsS "$backend_url" >/dev/null 2>&1; then
    echo "- backend: UP ($backend_url)"
    echo "- models:"
    curl -fsS "$models_url" || true
    echo
  else
    echo "- backend: DOWN ($backend_url)"
  fi

  if curl -fsS "$frontend_url" >/dev/null 2>&1; then
    echo "- frontend: UP ($frontend_url)"
  else
    echo "- frontend: DOWN ($frontend_url)"
  fi
}

start_demo() {
  ensure_prereqs
  mkdir -p "$STATE_DIR"

  if ! port_is_free "$BACKEND_PORT"; then
    echo "Backend port $BACKEND_PORT is busy (PID(s): $(listeners_on_port "$BACKEND_PORT" | tr '\n' ' '))."
    echo "Run: $(basename "$0") stop"
    exit 1
  fi

  if ! port_is_free "$FRONTEND_PORT"; then
    echo "Frontend port $FRONTEND_PORT is busy (PID(s): $(listeners_on_port "$FRONTEND_PORT" | tr '\n' ' '))."
    echo "Run: $(basename "$0") stop"
    exit 1
  fi

  echo "Starting backend on http://$HOST:$BACKEND_PORT"
  (
    cd "$BACKEND_DIR"
    source .venv/bin/activate
    exec uvicorn app.main:app --host "$HOST" --port "$BACKEND_PORT" --reload
  ) &
  local back_pid=$!
  echo "$back_pid" >"$BACKEND_PID_FILE"

  echo "Starting frontend on http://$HOST:$FRONTEND_PORT"
  (
    cd "$FRONTEND_DIR"
    exec npm run dev -- --strictPort --host "$HOST" --port "$FRONTEND_PORT"
  ) &
  local front_pid=$!
  echo "$front_pid" >"$FRONTEND_PID_FILE"

  cleanup() {
    kill_pid_gracefully "$back_pid" "backend"
    kill_pid_gracefully "$front_pid" "frontend"
    rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
  }
  trap cleanup EXIT INT TERM

  wait_for_http "http://$HOST:$BACKEND_PORT/api/health" "Backend"
  wait_for_http "http://$HOST:$FRONTEND_PORT" "Frontend"

  echo
  echo "Demo is ready:"
  echo "- Backend:  http://$HOST:$BACKEND_PORT"
  echo "- Frontend: http://$HOST:$FRONTEND_PORT"
  echo "- Press Ctrl+C to stop both."
  echo

  while true; do
    if ! is_running_pid "$back_pid"; then
      wait "$back_pid"
      exit $?
    fi
    if ! is_running_pid "$front_pid"; then
      wait "$front_pid"
      exit $?
    fi
    sleep 1
  done
}

command="${1:-start}"
case "$command" in
start|up)
  start_demo
  ;;
stop|down)
  stop_demo
  ;;
restart)
  stop_demo
  start_demo
  ;;
status)
  show_status
  ;;
-h|--help|help)
  usage
  ;;
*)
  echo "Unknown command: $command"
  usage
  exit 1
  ;;
esac
