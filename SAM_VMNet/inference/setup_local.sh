#!/bin/bash
# Setup local inference environment for Mac (Apple Silicon M4 Pro)
# Uses Python 3.13 — PyTorch doesn't have 3.14 wheels yet
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/inference/venv"

echo "=== SAM-VMNet Local Inference Setup ==="

# Use Python 3.13 (PyTorch compatible)
PYTHON=$(command -v python3.13 || command -v python3)
echo "Using Python: $PYTHON ($($PYTHON --version))"

# Create venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "Python in venv: $(python --version)"

# Install dependencies
echo "Installing PyTorch (MPS-enabled)..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing other dependencies..."
pip install numpy pillow opencv-python-headless einops timm scikit-image matplotlib medpy thop

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run inference:"
echo "  source inference/venv/bin/activate"
echo "  python inference/predict.py --image path/to/angiogram.png"
echo ""
echo "For Branch 2 (SAM-VMNet, better quality):"
echo "  python inference/predict.py --image path/to/angiogram.png --branch 2"
echo ""
echo "For batch processing:"
echo "  python inference/predict.py --image_dir path/to/images/ --output_dir results/"
