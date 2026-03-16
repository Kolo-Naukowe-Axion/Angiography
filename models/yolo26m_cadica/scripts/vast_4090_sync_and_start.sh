#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${1:-}"
VAST_BIN="${VAST_BIN:-/tmp/vastai-cli/bin/vastai}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/Angiography}"
REMOTE_MODEL_ROOT="${REMOTE_ROOT}/models/yolo26m_cadica"
REMOTE_DATASET_ROOT="${REMOTE_ROOT}/datasets/cadica/derived/yolo26_selected_seed42"
REMOTE_SEED_DIR="${REMOTE_ROOT}/models/yolo26m_cadica/checkpoints"
LOCAL_SEED_MODEL="${LOCAL_SEED_MODEL:-models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 <instance_id>"
  exit 1
fi

if [[ ! -x "${VAST_BIN}" ]]; then
  echo "vastai CLI not found at ${VAST_BIN}"
  exit 2
fi

SSH_URL_RAW="$("${VAST_BIN}" ssh-url "${INSTANCE_ID}" | sed 's#ssh://##')"
SSH_DEST="${SSH_URL_RAW%%:*}"
SSH_PORT="${SSH_URL_RAW##*:}"
RSYNC_SSH="ssh -i ${SSH_KEY} -p ${SSH_PORT} -o StrictHostKeyChecking=accept-new"

ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_DEST}" \
  "mkdir -p ${REMOTE_MODEL_ROOT} ${REMOTE_ROOT}/datasets/cadica/derived ${REMOTE_SEED_DIR}"

rsync -az --delete -e "${RSYNC_SSH}" \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'runs/' \
  models/yolo26m_cadica/ "${SSH_DEST}:${REMOTE_MODEL_ROOT}/"

rsync -azL --delete -e "${RSYNC_SSH}" \
  datasets/cadica/derived/yolo26_selected_seed42/ \
  "${SSH_DEST}:${REMOTE_DATASET_ROOT}/"

if [[ -f "${LOCAL_SEED_MODEL}" ]]; then
  ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_DEST}" "mkdir -p ${REMOTE_SEED_DIR}"
  rsync -az -e "${RSYNC_SSH}" \
    "${LOCAL_SEED_MODEL}" \
    "${SSH_DEST}:${REMOTE_SEED_DIR}/local_last.pt"
fi

ssh -i "${SSH_KEY}" -p "${SSH_PORT}" -o StrictHostKeyChecking=accept-new "${SSH_DEST}" \
  "cd ${REMOTE_ROOT} && bash models/yolo26m_cadica/scripts/run_vast_4090_pipeline.sh"
