#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/Angiography}"
RUN_NAME="${RUN_NAME:-cadica_selected_seed42_4090}"
RUN_DIR="${ROOT_DIR}/models/yolo26x/runs/${RUN_NAME}"
DATA_YAML="${DATA_YAML:-${ROOT_DIR}/datasets/cadica/derived/yolo26_selected_seed42/data.yaml}"
EPOCHS="${EPOCHS:-300}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-12}"
IOU_EVERY="${IOU_EVERY:-10}"

cd "${ROOT_DIR}"

python3 -m pip install -r models/yolo26x/requirements-vast.txt
python3 models/yolo26m_cadica/scripts/verify_cadica_selected.py --dataset-root datasets/cadica/derived/yolo26_selected_seed42

mkdir -p "${RUN_DIR}"

nohup python3 -u models/yolo26x/train.py \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --device 0 \
  --project "${ROOT_DIR}/models/yolo26x/runs" \
  --name "${RUN_NAME}" \
  > "${RUN_DIR}/train.log" 2>&1 &
echo $! > "${RUN_DIR}/train.pid"

nohup python3 -u models/yolo26x/scripts/periodic_iou_eval.py \
  --run-dir "${RUN_DIR}" \
  --data "${DATA_YAML}" \
  --split val \
  --device 0 \
  --imgsz "${IMGSZ}" \
  --every "${IOU_EVERY}" \
  --epochs "${EPOCHS}" \
  > "${RUN_DIR}/periodic_iou.log" 2>&1 &
echo $! > "${RUN_DIR}/periodic_iou.pid"

echo "Training started."
echo "Run dir: ${RUN_DIR}"
echo "Train PID: $(cat "${RUN_DIR}/train.pid")"
echo "IoU monitor PID: $(cat "${RUN_DIR}/periodic_iou.pid")"
echo "Tail logs with:"
echo "  tail -f ${RUN_DIR}/train.log"
echo "  tail -f ${RUN_DIR}/periodic_iou.log"
