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

# DCA1 dataset setup
echo ""
echo "=== DCA1 Dataset Setup ==="
pip install kaggle
echo "NOTE: Set Kaggle credentials before running setup_dca1.py:"
echo "  export KAGGLE_USERNAME=your_username"
echo "  export KAGGLE_KEY=your_key"
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "Kaggle credentials detected, setting up DCA1 dataset..."
    python setup_dca1.py
else
    echo "Kaggle credentials not set. Run manually: python setup_dca1.py"
fi

echo "=== Setup complete! ==="
echo ""
echo "To run: jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser"
echo ""
echo "DCA1 training command:"
echo "  python SAM_VMNet/train_branch1.py \\"
echo "      --config dca1 --batch_size 4 --gpu_id 0 --epochs 200 \\"
echo "      --work_dir ./result_dca1_branch1/ \\"
echo "      --data_path ./data/dca1/ \\"
echo "      --pretrained_ckpt ./pre_trained_weights/best-epoch142-loss0.3230.pth"
