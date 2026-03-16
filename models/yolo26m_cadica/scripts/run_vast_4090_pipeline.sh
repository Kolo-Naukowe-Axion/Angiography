#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/Angiography}"
RUN_NAME="${RUN_NAME:-cadica_selected_seed42_4090}"
RUN_DIR="${ROOT_DIR}/models/yolo26m_cadica/runs/${RUN_NAME}"
DATASET_ROOT="${ROOT_DIR}/datasets/cadica/derived/yolo26_selected_seed42"
DATA_YAML="${DATA_YAML:-${DATASET_ROOT}/data.yaml}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/models/yolo26m_cadica/checkpoints/local_last.pt}"
EPOCHS="${EPOCHS:-300}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-50}"
IOU_EVERY="${IOU_EVERY:-10}"

cd "${ROOT_DIR}"

python3 -m pip install -r models/yolo26m_cadica/requirements-vast.txt
python3 models/yolo26m_cadica/scripts/rewrite_data_yaml.py --dataset-root "${DATASET_ROOT}"
python3 models/yolo26m_cadica/scripts/verify_prepared_dataset.py --dataset-root "${DATASET_ROOT}"

mkdir -p "${RUN_DIR}"
rm -f "${RUN_DIR}/results.csv" "${RUN_DIR}/iou_metrics.csv" "${RUN_DIR}/train.log" "${RUN_DIR}/periodic_iou.log"
rm -f "${RUN_DIR}/train.pid" "${RUN_DIR}/periodic_iou.pid"
rm -rf "${RUN_DIR}/weights"

MODEL_ARGS=(--model "yolo26m.pt")
if [[ -f "${SEED_MODEL}" ]]; then
  MODEL_ARGS=(--model "${SEED_MODEL}")
  echo "Seeding 4090 training from checkpoint: ${SEED_MODEL}"
else
  echo "No seed checkpoint found. Starting from yolo26m.pt"
fi

nohup python3 -u models/yolo26m_cadica/train.py \
  --data "${DATA_YAML}" \
  --epochs "${EPOCHS}" \
  --imgsz "${IMGSZ}" \
  --batch "${BATCH}" \
  --workers "${WORKERS}" \
  --patience "${PATIENCE}" \
  --device 0 \
  --project "${ROOT_DIR}/models/yolo26m_cadica/runs" \
  --name "${RUN_NAME}" \
  --cache \
  "${MODEL_ARGS[@]}" \
  > "${RUN_DIR}/train.log" 2>&1 &
echo $! > "${RUN_DIR}/train.pid"

nohup python3 -u models/yolo26m_cadica/scripts/periodic_iou_eval.py \
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
