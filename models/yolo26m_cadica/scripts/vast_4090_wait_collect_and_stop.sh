#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
VAST_BIN="${VAST_BIN:-/tmp/vastai-cli/bin/vastai}"
RUN_NAME="${RUN_NAME:-cadica_selected_seed42_4090}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/Angiography}"
REMOTE_RUN_DIR="${REMOTE_ROOT}/models/yolo26m_cadica/runs/${RUN_NAME}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-models/yolo26m_cadica/runs/${RUN_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

SSH_URL_RAW="$("${VAST_BIN}" ssh-url "${INSTANCE_ID}" | sed 's#ssh://##')"
SSH_DEST="${SSH_URL_RAW%%:*}"
SSH_PORT="${SSH_URL_RAW##*:}"
RSYNC_SSH="ssh -i ${SSH_KEY} -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new"

echo "Watching remote training on instance ${INSTANCE_ID}"
while true; do
  STATUS="$(ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_DEST}" "python3 - <<'PY'
from pathlib import Path
run_dir = Path('${REMOTE_RUN_DIR}')
pid_path = run_dir / 'train.pid'
results_csv = run_dir / 'results.csv'
last_pt = run_dir / 'weights' / 'last.pt'
if pid_path.exists():
    pid = pid_path.read_text().strip()
else:
    pid = ''
train_alive = False
if pid:
    try:
        import os
        os.kill(int(pid), 0)
        train_alive = True
    except Exception:
        train_alive = False
print('alive' if train_alive else 'stopped')
print('results' if results_csv.exists() else 'no_results')
print('weights' if last_pt.exists() else 'no_weights')
PY")"

  TRAIN_STATE="$(printf '%s\n' "${STATUS}" | sed -n '1p')"
  RESULTS_STATE="$(printf '%s\n' "${STATUS}" | sed -n '2p')"
  WEIGHTS_STATE="$(printf '%s\n' "${STATUS}" | sed -n '3p')"
  echo "$(date '+%Y-%m-%d %H:%M:%S') remote state: ${TRAIN_STATE}, ${RESULTS_STATE}, ${WEIGHTS_STATE}"

  if [[ "${TRAIN_STATE}" == "stopped" && "${WEIGHTS_STATE}" == "weights" ]]; then
    break
  fi

  sleep "${POLL_SECONDS}"
done

mkdir -p "${LOCAL_RUN_DIR}"
rsync -az -e "${RSYNC_SSH}" \
  "${SSH_DEST}:${REMOTE_RUN_DIR}/" \
  "${LOCAL_RUN_DIR}/"

echo "Remote artifacts copied to ${LOCAL_RUN_DIR}"
"${VAST_BIN}" destroy instance "${INSTANCE_ID}"
echo "Destroyed Vast.ai instance ${INSTANCE_ID}"
