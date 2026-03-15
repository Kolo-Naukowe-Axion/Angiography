#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
RUN_NAME="${RUN_NAME:-cadica_selected_seed42_4090}"
VAST_BIN="${VAST_BIN:-/tmp/vastai-cli/bin/vastai}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

SSH_URL="$("${VAST_BIN}" ssh-url "${INSTANCE_ID}" | sed 's#ssh://##')"
ssh -t "${SSH_URL}" \
  "cd /workspace/Angiography && python3 models/yolo26x/scripts/monitor_training.py --run-dir models/yolo26x/runs/${RUN_NAME} --epochs 300 --interval 15"
