#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$APP_DIR/.." && pwd)"
BACKEND_DIR="$APP_DIR/backend"
FRONTEND_DIR="$APP_DIR/frontend"
BACKEND_REQUIREMENTS="$BACKEND_DIR/requirements.txt"

HOST="${DEMO_HOST:-127.0.0.1}"
BACKEND_PORT="${DEMO_BACKEND_PORT:-8000}"
FRONTEND_PORT="${DEMO_FRONTEND_PORT:-5173}"
DATA_DIR="${DEMO_DATA_DIR:-$APP_DIR/data/patients}"
MODEL_PATH="${DEMO_MODEL_PATH:-$ROOT_DIR/YOLO26s/weights/best.pt}"
AUTO_SETUP="${DEMO_AUTO_SETUP:-1}"
AUTO_USE_MOCK_MODEL="${DEMO_AUTO_USE_MOCK_MODEL:-1}"
SOURCE_ROOT="${DEMO_SOURCE_ROOT:-}"
PYTHON_BIN_OVERRIDE="${DEMO_PYTHON_BIN:-}"

STATE_KEY="$(printf '%s' "$ROOT_DIR" | cksum | awk '{print $1}')"
STATE_DIR="${TMPDIR:-/tmp}/angiography-demo-$STATE_KEY"
BACKEND_PID_FILE="$STATE_DIR/backend.pid"
FRONTEND_PID_FILE="$STATE_DIR/frontend.pid"

SELECTED_PYTHON=""
BACKEND_PYTHON=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|restart|status|bootstrap|doctor]

Commands:
  start    Start backend + frontend (default)
  stop     Stop demo processes started from this repo
  restart  Stop then start
  status   Show process and endpoint status
  bootstrap  Install/update dependencies and validate demo data/model
  doctor   Validate environment without changing anything

Environment overrides:
  DEMO_HOST
  DEMO_BACKEND_PORT
  DEMO_FRONTEND_PORT
  DEMO_API_HOST
  DEMO_DATA_DIR
  DEMO_MODEL_PATH
  DEMO_USE_MOCK_MODEL
  DEMO_AUTO_USE_MOCK_MODEL
  DEMO_SOURCE_ROOT
  DEMO_PYTHON_BIN
  DEMO_AUTO_SETUP
EOF
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_truthy() {
  case "${1:-}" in
  1 | true | TRUE | yes | YES | on | ON)
    return 0
    ;;
  *)
    return 1
    ;;
  esac
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

public_host() {
  if [ "$HOST" = "0.0.0.0" ]; then
    echo "127.0.0.1"
    return
  fi
  if [ "$HOST" = "::" ]; then
    echo "localhost"
    return
  fi
  echo "$HOST"
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

select_python() {
  local candidates=()

  if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    candidates=("$PYTHON_BIN_OVERRIDE")
  else
    candidates=(python3.11 python3)
  fi

  local candidate=""
  local version=""
  local major=""
  local minor=""

  for candidate in "${candidates[@]}"; do
    if ! command_exists "$candidate"; then
      continue
    fi
    version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [ -z "$version" ]; then
      continue
    fi
    IFS='.' read -r major minor <<<"$version"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
      SELECTED_PYTHON="$candidate"
      if [ "$minor" -ne 11 ]; then
        echo "Note: using Python $version via $candidate (3.11 recommended)."
      fi
      return 0
    fi
  done

  if [ -n "$PYTHON_BIN_OVERRIDE" ]; then
    fail "DEMO_PYTHON_BIN=$PYTHON_BIN_OVERRIDE is unavailable or not Python >=3.10."
  fi
  fail "Python >=3.10 is required. Install python3.11, or set DEMO_PYTHON_BIN to a valid interpreter."
}

ensure_base_commands() {
  local missing=()
  local cmd=""
  for cmd in curl lsof pgrep; do
    if ! command_exists "$cmd"; then
      missing+=("$cmd")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    fail "Missing required command(s): ${missing[*]}"
  fi
}

ensure_node_runtime() {
  command_exists node || fail "Node.js is required. Install Node 20.19+ (or >=22.12)."
  command_exists npm || fail "npm is required. Install Node.js with npm."

  local raw_version
  raw_version="$(node -v 2>/dev/null || true)"
  raw_version="${raw_version#v}"

  local major minor patch
  IFS='.' read -r major minor patch <<<"$raw_version"
  major="${major:-0}"
  minor="${minor:-0}"

  if [ "$major" -lt 20 ] \
    || [ "$major" -eq 21 ] \
    || { [ "$major" -eq 20 ] && [ "$minor" -lt 19 ]; } \
    || { [ "$major" -eq 22 ] && [ "$minor" -lt 12 ]; }; then
    fail "Node $(node -v) is incompatible with Vite 7. Use Node 20.19+ or >=22.12."
  fi
}

ensure_backend_env() {
  local mode="$1"

  select_python

  if [ ! -x "$BACKEND_DIR/.venv/bin/python" ]; then
    if [ "$mode" = "check" ]; then
      fail "Backend venv missing at $BACKEND_DIR/.venv. Run: $(basename "$0") bootstrap"
    fi
    echo "Creating backend virtualenv with $SELECTED_PYTHON"
    "$SELECTED_PYTHON" -m venv "$BACKEND_DIR/.venv"
  fi

  BACKEND_PYTHON="$BACKEND_DIR/.venv/bin/python"
  [ -x "$BACKEND_PYTHON" ] || fail "Backend venv is invalid at $BACKEND_DIR/.venv."

  local req_checksum
  req_checksum="$(cksum "$BACKEND_REQUIREMENTS" | awk '{print $1 ":" $2}')"
  local req_stamp="$BACKEND_DIR/.venv/.requirements.cksum"

  local needs_install=0
  local reason=""

  if [ ! -f "$BACKEND_DIR/.venv/bin/uvicorn" ]; then
    needs_install=1
    reason="uvicorn is missing"
  elif ! "$BACKEND_PYTHON" -c "import fastapi, uvicorn, ultralytics, torch" >/dev/null 2>&1; then
    needs_install=1
    reason="Python dependencies are missing or broken"
  elif [ ! -f "$req_stamp" ] || [ "$(cat "$req_stamp" 2>/dev/null || true)" != "$req_checksum" ]; then
    needs_install=1
    reason="requirements.txt changed"
  fi

  if [ "$needs_install" -eq 1 ]; then
    if [ "$mode" = "check" ]; then
      fail "Backend dependencies are not ready ($reason). Run: $(basename "$0") bootstrap"
    fi
    echo "Installing backend dependencies ($reason)..."
    "$BACKEND_PYTHON" -m pip install --upgrade pip >/dev/null
    "$BACKEND_PYTHON" -m pip install -r "$BACKEND_REQUIREMENTS"
    printf '%s' "$req_checksum" >"$req_stamp"
  fi
}

ensure_frontend_env() {
  local mode="$1"

  ensure_node_runtime

  local lock_checksum
  lock_checksum="$(cksum "$FRONTEND_DIR/package-lock.json" | awk '{print $1 ":" $2}')"
  local lock_stamp="$FRONTEND_DIR/node_modules/.deps.cksum"

  local needs_install=0
  local reason=""

  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    needs_install=1
    reason="node_modules is missing"
  elif [ ! -f "$lock_stamp" ]; then
    needs_install=1
    reason="frontend dependency stamp missing"
  elif [ "$(cat "$lock_stamp" 2>/dev/null || true)" != "$lock_checksum" ]; then
    needs_install=1
    reason="package-lock.json changed"
  elif [ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]; then
    needs_install=1
    reason="vite binary is missing"
  fi

  if [ "$needs_install" -eq 1 ]; then
    if [ "$mode" = "check" ]; then
      fail "Frontend dependencies are not ready ($reason). Run: $(basename "$0") bootstrap"
    fi
    echo "Installing frontend dependencies ($reason)..."
    (
      cd "$FRONTEND_DIR"
      npm install
      printf '%s' "$lock_checksum" >"$lock_stamp"
    )
  fi
}

validate_model_path() {
  if is_truthy "${DEMO_USE_MOCK_MODEL:-0}"; then
    echo "Model mode: mock (DEMO_USE_MOCK_MODEL=${DEMO_USE_MOCK_MODEL:-1})"
    return
  fi

  if [ -f "$MODEL_PATH" ]; then
    return
  fi

  if is_truthy "$AUTO_USE_MOCK_MODEL"; then
    export DEMO_USE_MOCK_MODEL=1
    echo "Model not found at $MODEL_PATH. Falling back to mock model (DEMO_USE_MOCK_MODEL=1)."
    return
  fi

  fail "Model weights missing at $MODEL_PATH. Set DEMO_MODEL_PATH, or set DEMO_USE_MOCK_MODEL=1."
}

validate_data_manifest() {
  local mode="$1"
  local validation_output

  if validation_output="$(
    cd "$BACKEND_DIR"
    DEMO_DATA_DIR="$DATA_DIR" "$BACKEND_PYTHON" - <<'PY' 2>&1
from pathlib import Path
import os
import sys

from app.data import ManifestValidationError, PatientStore

data_dir = Path(os.environ["DEMO_DATA_DIR"]).resolve()
try:
    store = PatientStore(data_dir)
except ManifestValidationError as error:
    print(error)
    sys.exit(1)

print(f"{len(store.summaries())} patient(s)")
PY
  )"; then
    echo "Patient data: ready ($validation_output)"
    return
  fi

  if [ "$mode" = "fix" ] && [ -n "$SOURCE_ROOT" ] && [ -d "$SOURCE_ROOT" ]; then
    echo "Patient data invalid: $validation_output"
    echo "Preparing patient data from DEMO_SOURCE_ROOT=$SOURCE_ROOT ..."
    "$SELECTED_PYTHON" "$APP_DIR/scripts/prepare_patient_data.py" \
      --source-root "$SOURCE_ROOT" \
      --output-root "$DATA_DIR"

    if validation_output="$(
      cd "$BACKEND_DIR"
      DEMO_DATA_DIR="$DATA_DIR" "$BACKEND_PYTHON" - <<'PY' 2>&1
from pathlib import Path
import os
import sys

from app.data import ManifestValidationError, PatientStore

data_dir = Path(os.environ["DEMO_DATA_DIR"]).resolve()
try:
    store = PatientStore(data_dir)
except ManifestValidationError as error:
    print(error)
    sys.exit(1)

print(f"{len(store.summaries())} patient(s)")
PY
    )"; then
      echo "Patient data: ready ($validation_output)"
      return
    fi
  fi

  fail "Patient data invalid: $validation_output
Fix by running:
  $SELECTED_PYTHON \"$APP_DIR/scripts/prepare_patient_data.py\" --source-root /ABS/PATH/TO/CURATED_DATA --output-root \"$DATA_DIR\""
}

validate_sam_precomputed_readiness() {
  local readiness_output

  if readiness_output="$(
    cd "$BACKEND_DIR"
    DEMO_DATA_DIR="$DATA_DIR" "$BACKEND_PYTHON" - <<'PY' 2>&1
from pathlib import Path
import os

from app.data import ManifestValidationError, PatientStore

data_dir = Path(os.environ["DEMO_DATA_DIR"]).resolve()
try:
    store = PatientStore(data_dir)
except ManifestValidationError as error:
    print(f"unable to evaluate (manifest invalid): {error}")
    raise SystemExit(1)

ready, reason = store.is_model_prediction_ready("sam_vmnet_arcade", "arcade")
if ready:
    print("ready")
else:
    print(reason or "not ready")
    raise SystemExit(1)
PY
  )"; then
    echo "SAM-VMNet precomputed masks: ready ($readiness_output)"
    return
  fi

  echo "SAM-VMNet precomputed masks: unavailable ($readiness_output)"
  echo "  Tip: run demo-app/scripts/prepare_arcade_data.py + demo-app/scripts/precompute_sam_vmnet_masks.py"
}

ensure_prereqs() {
  local mode="$1"

  ensure_base_commands
  ensure_backend_env "$mode"
  ensure_frontend_env "$mode"
  validate_model_path
  validate_data_manifest "$mode"
  validate_sam_precomputed_readiness
}

bootstrap_demo() {
  echo "Bootstrapping demo dependencies and data checks..."
  ensure_prereqs "fix"
  echo "Bootstrap complete."
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
  pgrep -f "$BACKEND_DIR/.venv/bin/uvicorn app.main:app" || true
}

known_frontend_pids() {
  pgrep -f "$FRONTEND_DIR/node_modules/.bin/vite" || true
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
  local probe_host
  probe_host="$(public_host)"

  local backend_url="http://$probe_host:$BACKEND_PORT/api/health"
  local models_url="http://$probe_host:$BACKEND_PORT/api/models"
  local frontend_url="http://$probe_host:$FRONTEND_PORT"

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

doctor_demo() {
  echo "Running environment checks (read-only)..."
  ensure_prereqs "check"
  echo "All checks passed."
  show_status
}

start_demo() {
  if is_truthy "$AUTO_SETUP"; then
    ensure_prereqs "fix"
  else
    ensure_prereqs "check"
  fi
  mkdir -p "$STATE_DIR"

  local public
  public="$(public_host)"
  local api_host="${DEMO_API_HOST:-$public}"
  local api_base_url="${VITE_API_BASE_URL:-http://$api_host:$BACKEND_PORT}"
  local frontend_origin="${DEMO_FRONTEND_ORIGIN:-http://$public:$FRONTEND_PORT}"

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
    export DEMO_DATA_DIR="$DATA_DIR"
    export DEMO_MODEL_PATH="$MODEL_PATH"
    export DEMO_FRONTEND_ORIGIN="$frontend_origin"
    exec uvicorn app.main:app --host "$HOST" --port "$BACKEND_PORT" --reload
  ) &
  local back_pid=$!
  echo "$back_pid" >"$BACKEND_PID_FILE"

  echo "Starting frontend on http://$HOST:$FRONTEND_PORT"
  (
    cd "$FRONTEND_DIR"
    export VITE_API_BASE_URL="$api_base_url"
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

  wait_for_http "http://$public:$BACKEND_PORT/api/health" "Backend"
  wait_for_http "http://$public:$FRONTEND_PORT" "Frontend"

  echo
  echo "Demo is ready:"
  echo "- Backend:  http://$public:$BACKEND_PORT"
  echo "- Frontend: http://$public:$FRONTEND_PORT"
  echo "- API base: $api_base_url"
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
bootstrap)
  bootstrap_demo
  ;;
doctor)
  doctor_demo
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
