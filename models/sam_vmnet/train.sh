SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/../../datasets/arcade/data/vessel}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/result_branch2}"
MEDSAM_PATH="${MEDSAM_PATH:-${SCRIPT_DIR}/pre_trained_weights/medsam_vit_b.pth}"
BRANCH1_MODEL_PATH="${BRANCH1_MODEL_PATH:-${SCRIPT_DIR}/result_branch1/checkpoints/best-epoch160-loss0.2211.pth}"

# ==========================================
# Branch 1 Training Script
# workdir: results for saving models and logs
# data_path: path to the training dataset
# ==========================================
# python train_branch1.py \
#    --batch_size 8 \
#    --gpu_id "3" \
#    --epochs 200 \
#    --work_dir "./result_branch1/" \
#    --data_path "../datasets/arcade/data/vessel/"

# ==========================================
# Branch 2 Training Script
# workdir: results for saving models and logs important!!! Different path from branch1
# data_path: path to the training dataset
# medsam_path: path to the checkpoint of MedSAM
# branch1_model_path: path to the checkpoint of Branch 1 Pure-VM-UNet
# ==========================================
python "${SCRIPT_DIR}/train_branch2.py" \
    --batch_size 4 \
    --gpu_id "3" \
    --epochs 5 \
    --work_dir "${WORK_DIR}" \
    --data_path "${DATA_PATH}" \
    --medsam_path "${MEDSAM_PATH}" \
    --branch1_model_path "${BRANCH1_MODEL_PATH}"
