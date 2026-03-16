#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
RUN_NAME="${RUN_NAME:-cadica_selected_seed42_4090}"
VAST_BIN="${VAST_BIN:-/tmp/vastai-cli/bin/vastai}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

SSH_URL_RAW="$("${VAST_BIN}" ssh-url "${INSTANCE_ID}" | sed 's#ssh://##')"
SSH_DEST="${SSH_URL_RAW%%:*}"
SSH_PORT="${SSH_URL_RAW##*:}"
ssh -t -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_DEST}" \
  "cd /workspace/Angiography && python3 models/yolo26m_cadica/scripts/monitor_training.py --run-dir models/yolo26m_cadica/runs/${RUN_NAME} --epochs 300 --interval 15"
