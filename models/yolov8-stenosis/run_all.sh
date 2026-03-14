#!/bin/bash
# Full pipeline: train → evaluate → benchmark → export ONNX
# Run this once on a GPU server and come back to results.
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh --batch 32

set -e
cd "$(dirname "$0")"

BATCH=${1:-32}
DEVICE=0
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo " STEP 1/4  Training  (batch=$BATCH)"
echo "============================================"
python train.py --batch "$BATCH" --device "$DEVICE"

# Find best weights from the latest run
BEST_PT=$(find runs -name "best.pt" -newer yolov8m.pt 2>/dev/null | head -1)
if [ -z "$BEST_PT" ]; then
    BEST_PT=$(find runs -name "best.pt" | sort | tail -1)
fi
echo ""
echo "Best weights: $BEST_PT"

echo ""
echo "============================================"
echo " STEP 2/4  Evaluation on test set"
echo "============================================"
python evaluate.py --weights "$BEST_PT" --data dataset/data.yaml --output "$RESULTS_DIR/metrics.json"

echo ""
echo "============================================"
echo " STEP 3/4  Speed benchmark + ONNX export"
echo "============================================"
python scripts/benchmark_speed.py \
    --weights "$BEST_PT" \
    --export-onnx \
    --output "$RESULTS_DIR/benchmark.json"

echo ""
echo "============================================"
echo " STEP 4/4  Copying artifacts to results/"
echo "============================================"
RUN_DIR=$(dirname $(dirname "$BEST_PT"))
cp "$RUN_DIR/results.csv"           "$RESULTS_DIR/" 2>/dev/null || true
cp "$RUN_DIR/confusion_matrix.png"  "$RESULTS_DIR/" 2>/dev/null || true
cp "$RUN_DIR/PR_curve.png"          "$RESULTS_DIR/" 2>/dev/null || true
cp "$RUN_DIR/F1_curve.png"          "$RESULTS_DIR/" 2>/dev/null || true
cp "$RUN_DIR/results.png"           "$RESULTS_DIR/" 2>/dev/null || true
cp "$BEST_PT"                        "$RESULTS_DIR/best.pt"

ONNX_PT="${BEST_PT%.pt}.onnx"
if [ -f "$ONNX_PT" ]; then
    cp "$ONNX_PT" "$RESULTS_DIR/best.onnx"
fi

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
echo "Results in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR/"
