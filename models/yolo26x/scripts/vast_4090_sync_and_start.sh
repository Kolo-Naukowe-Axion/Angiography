#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
VAST_BIN="${VAST_BIN:-/tmp/vastai-cli/bin/vastai}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/Angiography}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

if [[ ! -x "${VAST_BIN}" ]]; then
  echo "vastai CLI not found at ${VAST_BIN}"
  exit 2
fi

SSH_URL="$("${VAST_BIN}" ssh-url "${INSTANCE_ID}" | sed 's#ssh://##')"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  ./ "${SSH_URL}:${REMOTE_ROOT}/"

rsync -azL --delete \
  datasets/cadica/derived/yolo26_selected_seed42/ \
  "${SSH_URL}:${REMOTE_ROOT}/datasets/cadica/derived/yolo26_selected_seed42/"

ssh "${SSH_URL}" "cd ${REMOTE_ROOT} && bash models/yolo26x/scripts/run_vast_4090_pipeline.sh"
