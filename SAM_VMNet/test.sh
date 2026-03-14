SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==========================================
# Branch 1 Testing Script
# data_path: path to the training dataset
# pretrained_weight: checkpoint of the branch1 pure VM-UNet
# output_dir: path to the prediction of test set
# ==========================================
python "${SCRIPT_DIR}/test.py" \
    --data_path "${DATA_PATH:-${SCRIPT_DIR}/../datasets/arcade/data/vessel}" \
    --pretrained_weight "${PRETRAINED_WEIGHT:-${SCRIPT_DIR}/result_branch1/checkpoints/best-epoch160-loss0.2211.pth}" \
    --device "${DEVICE:-cuda:0}" \
    --pred_masks_dir "${PRED_MASKS_DIR:-${SCRIPT_DIR}/../datasets/arcade/data/vessel/test/pred_masks}"
