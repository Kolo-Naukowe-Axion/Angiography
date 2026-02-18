#!/bin/bash
# ============================================================
# SAM-VMNet: Full Training Pipeline
# ============================================================
# Runs the complete training pipeline end-to-end:
#   1. Branch 1: Pure VM-UNet training (200 epochs)
#   2. Generate predicted masks for test set (needed for feature extraction)
#   3. Feature extraction: MedSAM features for all splits
#   4. Branch 2: SAM-VMNet training (100 epochs)
#
# Prerequisites:
#   - Run setup_vastai.sh first
#   - Run prepare_mendeley.py to prepare the dataset
#   - Pre-trained weights in ./pre_trained_weights/
#
# Usage:
#   ./run_training.sh                    # Full pipeline
#   ./run_training.sh --skip_branch1     # Skip Branch 1 (if already trained)
#   ./run_training.sh --quick_test       # Quick test (1 epoch each)
# ============================================================

set -e  # Exit on any error

# --------------------------------------------------
# Configuration
# --------------------------------------------------
GPU_ID="0"
DATA_PATH="./data/vessel/"
MEDSAM_PATH="./pre_trained_weights/medsam_vit_b.pth"

# Branch 1 settings
B1_BATCH_SIZE=16
B1_EPOCHS=200
B1_WORK_DIR="./result_branch1/"

# Branch 2 settings
B2_BATCH_SIZE=8
B2_EPOCHS=100
B2_WORK_DIR="./result_branch2/"

# Parse arguments
SKIP_BRANCH1=false
QUICK_TEST=false

for arg in "$@"; do
    case $arg in
        --skip_branch1)
            SKIP_BRANCH1=true
            shift
            ;;
        --quick_test)
            QUICK_TEST=true
            B1_EPOCHS=1
            B2_EPOCHS=1
            B1_BATCH_SIZE=2
            B2_BATCH_SIZE=2
            shift
            ;;
    esac
done

echo "============================================"
echo "  SAM-VMNet Training Pipeline"
echo "============================================"
echo "  GPU: ${GPU_ID}"
echo "  Data: ${DATA_PATH}"
echo "  Branch 1: ${B1_EPOCHS} epochs, batch ${B1_BATCH_SIZE}"
echo "  Branch 2: ${B2_EPOCHS} epochs, batch ${B2_BATCH_SIZE}"
if [ "$QUICK_TEST" = true ]; then
    echo "  Mode: QUICK TEST"
fi
echo "============================================"
echo ""

# --------------------------------------------------
# Verify data exists
# --------------------------------------------------
echo "[Pre-check] Verifying dataset..."
if [ ! -d "${DATA_PATH}train/images" ] || [ ! -d "${DATA_PATH}train/masks" ]; then
    echo "ERROR: Dataset not found at ${DATA_PATH}"
    echo "Please run prepare_mendeley.py first:"
    echo "  python prepare_mendeley.py --data_dir /path/to/mendeley/ --output_dir ${DATA_PATH}"
    exit 1
fi

TRAIN_COUNT=$(ls "${DATA_PATH}train/images/" | wc -l | tr -d ' ')
VAL_COUNT=$(ls "${DATA_PATH}val/images/" | wc -l | tr -d ' ')
TEST_COUNT=$(ls "${DATA_PATH}test/images/" | wc -l | tr -d ' ')
echo "  Train: ${TRAIN_COUNT} images"
echo "  Val: ${VAL_COUNT} images"
echo "  Test: ${TEST_COUNT} images"
echo ""

# --------------------------------------------------
# Step 1: Branch 1 - Pure VM-UNet
# --------------------------------------------------
if [ "$SKIP_BRANCH1" = false ]; then
    echo "============================================"
    echo "  Step 1/4: Branch 1 Training (VM-UNet)"
    echo "============================================"
    START_TIME=$(date +%s)

    python train_branch1.py \
        --batch_size ${B1_BATCH_SIZE} \
        --gpu_id "${GPU_ID}" \
        --epochs ${B1_EPOCHS} \
        --work_dir "${B1_WORK_DIR}" \
        --data_path "${DATA_PATH}"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo ""
    echo "  Branch 1 completed in $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
    echo ""
else
    echo "[Skipping Branch 1 training]"
fi

# --------------------------------------------------
# Step 2: Generate predicted masks for test set
# --------------------------------------------------
echo "============================================"
echo "  Step 2/4: Generating Test Set Predictions"
echo "============================================"

# Find the best Branch 1 checkpoint
B1_BEST_CKPT=$(ls -t ${B1_WORK_DIR}checkpoints/best-epoch*.pth 2>/dev/null | head -1)

if [ -z "$B1_BEST_CKPT" ]; then
    # Fallback: try best.pth
    B1_BEST_CKPT="${B1_WORK_DIR}checkpoints/best.pth"
fi

if [ ! -f "$B1_BEST_CKPT" ]; then
    echo "ERROR: No Branch 1 checkpoint found in ${B1_WORK_DIR}checkpoints/"
    echo "Please ensure Branch 1 training completed successfully."
    exit 1
fi

echo "  Using checkpoint: ${B1_BEST_CKPT}"

python test.py \
    --data_path "${DATA_PATH}" \
    --pretrained_weight "${B1_BEST_CKPT}" \
    --device "cuda:${GPU_ID}" \
    --output_dir "${DATA_PATH}test"

echo "  Test predictions generated."
echo ""

# --------------------------------------------------
# Step 3: Feature Extraction (MedSAM)
# --------------------------------------------------
echo "============================================"
echo "  Step 3/4: MedSAM Feature Extraction"
echo "============================================"
echo "  (This is handled automatically by train_branch2.py)"
echo ""

# --------------------------------------------------
# Step 4: Branch 2 - SAM-VMNet
# --------------------------------------------------
echo "============================================"
echo "  Step 4/4: Branch 2 Training (SAM-VMNet)"
echo "============================================"
START_TIME=$(date +%s)

python train_branch2.py \
    --batch_size ${B2_BATCH_SIZE} \
    --gpu_id "${GPU_ID}" \
    --epochs ${B2_EPOCHS} \
    --work_dir "${B2_WORK_DIR}" \
    --data_path "${DATA_PATH}" \
    --medsam_path "${MEDSAM_PATH}" \
    --branch1_model_path "${B1_BEST_CKPT}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "  Branch 2 completed in $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo ""

# --------------------------------------------------
# Summary
# --------------------------------------------------
echo "============================================"
echo "  Training Pipeline Complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  Branch 1 checkpoints: ${B1_WORK_DIR}checkpoints/"
echo "  Branch 2 checkpoints: ${B2_WORK_DIR}checkpoints/"
echo "  Branch 1 logs: ${B1_WORK_DIR}log/"
echo "  Branch 2 logs: ${B2_WORK_DIR}log/"
echo ""

B2_BEST_CKPT=$(ls -t ${B2_WORK_DIR}checkpoints/best-epoch*.pth 2>/dev/null | head -1)
if [ -n "$B2_BEST_CKPT" ]; then
    echo "  Best Branch 2 model: ${B2_BEST_CKPT}"
fi
echo ""
