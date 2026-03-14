#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SKIP_SETUP="${SKIP_SETUP:-0}"
GPU_ID="${GPU_ID:-0}"
AUDIT_POLICY="${AUDIT_POLICY:-auto-rebuild}"
GROUP_LEVEL="${GROUP_LEVEL:-patient}"
SEED="${SEED:-42}"

DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../../datasets/arcade}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${DATA_ROOT}/downloads}"
VESSEL_ROOT="${VESSEL_ROOT:-${DATA_ROOT}/data/vessel}"
REPORT_DIR="${REPORT_DIR:-${DATA_ROOT}/data/vessel_meta}"

BRANCH1_EPOCHS="${BRANCH1_EPOCHS:-160}"
BRANCH2_EPOCHS="${BRANCH2_EPOCHS:-160}"
BRANCH1_BATCH="${BRANCH1_BATCH:-128}"
BRANCH2_BATCH="${BRANCH2_BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

BRANCH1_WORK_DIR="${BRANCH1_WORK_DIR:-${SCRIPT_DIR}/runs/branch1_h100}"
BRANCH2_WORK_DIR="${BRANCH2_WORK_DIR:-${SCRIPT_DIR}/runs/branch2_h100}"
MEDSAM_PATH="${MEDSAM_PATH:-${SCRIPT_DIR}/pre_trained_weights/medsam_vit_b.pth}"

if [[ "${SKIP_SETUP}" != "1" ]]; then
  bash "${SCRIPT_DIR}/setup_vast.sh"
fi

source "${SCRIPT_DIR}/.venv/bin/activate"

mkdir -p "${DATA_ROOT}" "${REPORT_DIR}" "${BRANCH1_WORK_DIR}" "${BRANCH2_WORK_DIR}"

echo "=== [1/5] Prepare ARCADE dataset ==="
python "${SCRIPT_DIR}/scripts/prepare_arcade_for_samvmnet.py" \
  --zenodo-record 10390295 \
  --download-root "${DOWNLOAD_ROOT}" \
  --output-vessel-root "${VESSEL_ROOT}" \
  --audit-policy "${AUDIT_POLICY}" \
  --group-level "${GROUP_LEVEL}" \
  --seed "${SEED}" \
  --report-dir "${REPORT_DIR}" \
  --overwrite

run_branch1() {
  local bs="$1"
  echo "Running Branch1 with batch_size=${bs}"
  python "${SCRIPT_DIR}/train_branch1.py" \
    --batch_size "${bs}" \
    --gpu_id "${GPU_ID}" \
    --epochs "${BRANCH1_EPOCHS}" \
    --work_dir "${BRANCH1_WORK_DIR}" \
    --data_path "${VESSEL_ROOT}" \
    --num_workers "${NUM_WORKERS}" \
    --amp \
    --amp_dtype "${AMP_DTYPE}" \
    --grad_accum_steps 1 \
    --tf32
}

echo "=== [2/5] Train Branch1 with OOM fallback ==="
BRANCH1_SUCCESS=0
for bs in "${BRANCH1_BATCH}" 96 64 48 32 24 16 8; do
  if run_branch1 "${bs}"; then
    BRANCH1_SUCCESS=1
    break
  fi
  echo "Branch1 failed with batch_size=${bs}, trying lower batch size..."
  sleep 2
done

if [[ "${BRANCH1_SUCCESS}" != "1" ]]; then
  echo "Branch1 training failed for all fallback batch sizes."
  exit 10
fi

BRANCH1_BEST="$(ls -1t "${BRANCH1_WORK_DIR}/checkpoints"/best-epoch*.pth 2>/dev/null | head -n 1 || true)"
if [[ -z "${BRANCH1_BEST}" ]]; then
  echo "Could not locate Branch1 best checkpoint in ${BRANCH1_WORK_DIR}/checkpoints"
  exit 11
fi

echo "=== [3/5] Export Branch1 test predictions -> test/pred_masks ==="
python "${SCRIPT_DIR}/test.py" \
  --data_path "${VESSEL_ROOT}" \
  --pretrained_weight "${BRANCH1_BEST}" \
  --device "cuda:${GPU_ID}" \
  --pred_masks_dir "${VESSEL_ROOT}/test/pred_masks"

run_branch2() {
  local bs="$1"
  echo "Running Branch2 with batch_size=${bs}"
  python "${SCRIPT_DIR}/train_branch2.py" \
    --batch_size "${bs}" \
    --gpu_id "${GPU_ID}" \
    --epochs "${BRANCH2_EPOCHS}" \
    --work_dir "${BRANCH2_WORK_DIR}" \
    --data_path "${VESSEL_ROOT}" \
    --medsam_path "${MEDSAM_PATH}" \
    --branch1_model_path "${BRANCH1_BEST}" \
    --num_workers "${NUM_WORKERS}" \
    --amp \
    --amp_dtype "${AMP_DTYPE}" \
    --grad_accum_steps 1 \
    --tf32 \
    --test_mask_subdir pred_masks
}

echo "=== [4/5] Train Branch2 with OOM fallback ==="
BRANCH2_SUCCESS=0
for bs in "${BRANCH2_BATCH}" 48 32 24 16 8 4; do
  if run_branch2 "${bs}"; then
    BRANCH2_SUCCESS=1
    break
  fi
  echo "Branch2 failed with batch_size=${bs}, trying lower batch size..."
  sleep 2
done

if [[ "${BRANCH2_SUCCESS}" != "1" ]]; then
  echo "Branch2 training failed for all fallback batch sizes."
  exit 12
fi

BRANCH2_BEST="$(ls -1t "${BRANCH2_WORK_DIR}/checkpoints"/best-epoch*.pth 2>/dev/null | head -n 1 || true)"

echo "=== [5/5] Done ==="
echo "Branch1 best: ${BRANCH1_BEST}"
echo "Branch2 best: ${BRANCH2_BEST:-not-found}"
echo "Data root: ${VESSEL_ROOT}"
echo "Reports: ${REPORT_DIR}"
