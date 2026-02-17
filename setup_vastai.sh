#!/bin/bash
# ============================================================
# SAM-VMNet: vast.ai Instance Setup Script
# ============================================================
# Run this on a fresh vast.ai instance to set up everything.
#
# Recommended instance specs:
#   GPU: 1x RTX 4090 (24 GB VRAM)
#   RAM: 32 GB
#   Disk: 50 GB
#   CUDA: 11.7 or 11.8
#   Docker: pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
#
# Usage:
#   chmod +x setup_vastai.sh
#   ./setup_vastai.sh
# ============================================================

set -e  # Exit on any error

REPO_URL="https://github.com/qimingfan10/SAM-VMNet.git"
WORK_DIR="/workspace/SAM-VMNet"
WEIGHTS_DIR="${WORK_DIR}/pre_trained_weights"

# Google Drive file IDs for pre-trained weights
VMAMBA_GDRIVE_ID="1XL7JuacjoZCr8w2b0c8CaQn8b0hREblk"
MEDSAM_GDRIVE_ID="1O5IVkcVxd2RtOcZEKuTR3WkOBiosHBfz"

echo "============================================"
echo "  SAM-VMNet vast.ai Setup"
echo "============================================"
echo ""

# --------------------------------------------------
# Step 1: System packages
# --------------------------------------------------
echo "[1/7] Installing system packages..."
apt-get update -qq && apt-get install -y -qq git wget unzip > /dev/null 2>&1
echo "  Done."

# --------------------------------------------------
# Step 2: Clone repository
# --------------------------------------------------
echo "[2/7] Cloning SAM-VMNet repository..."
if [ -d "$WORK_DIR" ]; then
    echo "  Repository already exists at ${WORK_DIR}, pulling latest..."
    cd "$WORK_DIR" && git pull
else
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi
echo "  Done."

# --------------------------------------------------
# Step 3: Install Python dependencies
# --------------------------------------------------
echo "[3/7] Installing Python dependencies..."
cd "$WORK_DIR"

# Install main requirements (skip CUDA-compiled packages for now)
pip install --quiet \
    monai matplotlib scikit-image "SimpleITK>=2.2.1" nibabel tqdm scipy \
    opencv-python tensorboardX scikit-learn thop h5py medpy \
    packaging "timm==0.4.12" chardet yacs termcolor submitit \
    gdown

# torch 1.13.0 should already be in the Docker image, but verify
python -c "import torch; assert torch.__version__.startswith('1.13'), f'Need torch 1.13.x, got {torch.__version__}'" 2>/dev/null || {
    echo "  Installing PyTorch 1.13.0..."
    pip install --quiet torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
        --extra-index-url https://download.pytorch.org/whl/cu117
}
echo "  Done."

# --------------------------------------------------
# Step 4: Install CUDA-dependent packages
# --------------------------------------------------
echo "[4/7] Installing CUDA-dependent packages (causal_conv1d, mamba_ssm)..."

# triton
pip install --quiet "triton==2.0.0" 2>/dev/null || {
    echo "  Warning: triton 2.0.0 install failed, trying latest..."
    pip install --quiet triton
}

# causal_conv1d
pip install --quiet "causal_conv1d==1.0.0" 2>/dev/null || {
    echo "  Building causal_conv1d from source..."
    pip install --quiet "causal_conv1d==1.0.0" --no-build-isolation
}

# mamba_ssm
pip install --quiet "mamba_ssm==1.0.1" 2>/dev/null || {
    echo "  Building mamba_ssm from source..."
    pip install --quiet "mamba_ssm==1.0.1" --no-build-isolation
}
echo "  Done."

# --------------------------------------------------
# Step 5: Download pre-trained weights
# --------------------------------------------------
echo "[5/7] Downloading pre-trained weights..."
mkdir -p "$WEIGHTS_DIR"

# VMamba backbone
if [ ! -f "${WEIGHTS_DIR}/vmamba_tiny_e292.pth" ]; then
    echo "  Downloading VMamba backbone weights..."
    gdown --id "$VMAMBA_GDRIVE_ID" -O "${WEIGHTS_DIR}/vmamba_tiny_e292.pth"
else
    echo "  VMamba weights already exist, skipping."
fi

# MedSAM
if [ ! -f "${WEIGHTS_DIR}/medsam_vit_b.pth" ]; then
    echo "  Downloading MedSAM weights..."
    gdown --id "$MEDSAM_GDRIVE_ID" -O "${WEIGHTS_DIR}/medsam_vit_b.pth"
else
    echo "  MedSAM weights already exist, skipping."
fi
echo "  Done."

# --------------------------------------------------
# Step 6: Verify installation
# --------------------------------------------------
echo "[6/7] Verifying installation..."

# Check CUDA
echo -n "  CUDA available: "
python -c "import torch; print(torch.cuda.is_available())"

echo -n "  GPU device: "
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

echo -n "  PyTorch version: "
python -c "import torch; print(torch.__version__)"

echo -n "  mamba_ssm: "
python -c "import mamba_ssm; print('OK')" 2>/dev/null || echo "FAILED"

echo -n "  causal_conv1d: "
python -c "import causal_conv1d; print('OK')" 2>/dev/null || echo "FAILED"

# Check weights exist
echo -n "  VMamba weights: "
[ -f "${WEIGHTS_DIR}/vmamba_tiny_e292.pth" ] && echo "OK" || echo "MISSING"

echo -n "  MedSAM weights: "
[ -f "${WEIGHTS_DIR}/medsam_vit_b.pth" ] && echo "OK" || echo "MISSING"

# --------------------------------------------------
# Step 7: Quick sanity check
# --------------------------------------------------
echo "[7/7] Running sanity check (import models)..."
cd "$WORK_DIR"
python -c "
import torch
from models.vmunet.vmunet import VMUNet
from models.vmunet.samvmnet import SAMVMNet
print('  Model imports: OK')

# Quick forward pass check with tiny input
model = VMUNet(num_classes=1, input_channels=3, depths=[2,2,2,2],
               depths_decoder=[2,2,2,1], drop_path_rate=0.2,
               load_ckpt_path='./pre_trained_weights/vmamba_tiny_e292.pth')
model.load_from()
model = model.cuda()
x = torch.randn(1, 3, 256, 256).cuda()
with torch.no_grad():
    y = model(x)
print(f'  Forward pass: OK (output shape: {y.shape})')
" || {
    echo "  WARNING: Sanity check failed. Check CUDA compatibility."
    echo "  You may need to adjust CUDA/PyTorch versions."
}

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download the Mendeley dataset to /workspace/mendeley_data/"
echo "     (https://data.mendeley.com/datasets/ydrm75xywg/1)"
echo ""
echo "  2. Prepare the dataset:"
echo "     cd ${WORK_DIR}"
echo "     python prepare_mendeley.py --data_dir /workspace/mendeley_data/ --output_dir ./data/vessel/"
echo ""
echo "  3. Run training:"
echo "     ./run_training.sh"
echo ""
