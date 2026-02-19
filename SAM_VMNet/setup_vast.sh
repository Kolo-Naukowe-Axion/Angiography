#!/bin/bash
# Setup script for vast.ai RTX 5090 instance
# Run: bash setup_vast.sh

set -e
echo "=== SAM-VMNet + Stenosis Detection Setup (vast.ai) ==="

# System deps
apt-get update && apt-get install -y git wget unzip libgl1-mesa-glx libglib2.0-0 2>/dev/null || true

# Clone repo
if [ ! -d "SAM-VMNet" ]; then
    git clone https://github.com/qimingfan10/SAM-VMNet.git
    cd SAM-VMNet
else
    cd SAM-VMNet
    git pull
fi

# Python deps (core only - skip MATLAB-specific torch version constraints)
pip install --upgrade pip
pip install numpy opencv-python scikit-image scipy matplotlib jupyterlab ipywidgets tqdm

# PyTorch for RTX 5090 (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Additional deps for SAM-VMNet model
pip install monai SimpleITK nibabel timm==0.4.12 tensorboardX scikit-learn thop h5py medpy

# Copy stenosis detection Python port
echo "=== Stenosis detection (Python port) ready ==="
echo "=== Setup complete! ==="
echo ""
echo "To run: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
