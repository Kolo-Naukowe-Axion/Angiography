#!/bin/bash
# ============================================================
# SAM-VMNet: vast.ai Instance Setup Script
# ============================================================
# Run this on a fresh vast.ai instance to set up everything.
#
# Recommended instance specs:
#   GPU: 1x H100 SXM (80 GB VRAM) or 1x RTX 4090 (24 GB VRAM)
#   RAM: 64 GB+
#   Disk: 100 GB+
#
# Usage:
#   chmod +x setup_vastai.sh
#   ./setup_vastai.sh
# ============================================================

set -e  # Exit on any error

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
# Step 2: Fix Git LFS checkout if needed
# --------------------------------------------------
echo "[2/7] Fixing repository checkout..."
cd "$WORK_DIR"

# LFS-tracked files from the original repo fail because the LFS objects
# aren't on our fork. Remove the LFS filter and just skip those files
# (we download the weights via gdown anyway).
git lfs uninstall 2>/dev/null || true
git config --local filter.lfs.smudge "git-lfs smudge --skip -- %f" 2>/dev/null || true
git config --local filter.lfs.process "git-lfs filter-process --skip" 2>/dev/null || true
git checkout -- . 2>/dev/null || true
echo "  Done."

# --------------------------------------------------
# Step 3: Install Python dependencies
# --------------------------------------------------
echo "[3/7] Installing Python dependencies..."
cd "$WORK_DIR"

# Detect installed PyTorch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "none")
echo "  Detected PyTorch: ${TORCH_VERSION}"

if [ "$TORCH_VERSION" = "none" ]; then
    echo "  No PyTorch found, installing latest..."
    pip install --quiet torch torchvision torchaudio
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    echo "  Installed PyTorch: ${TORCH_VERSION}"
fi

# Install main requirements (skip CUDA-compiled packages, installed separately)
pip install --quiet \
    monai matplotlib scikit-image "SimpleITK>=2.2.1" nibabel tqdm scipy \
    opencv-python tensorboardX scikit-learn thop h5py medpy \
    packaging "timm==0.4.12" chardet yacs termcolor submitit \
    gdown
echo "  Done."

# --------------------------------------------------
# Step 4: Install CUDA-dependent packages
# --------------------------------------------------
echo "[4/7] Installing CUDA-dependent packages (causal_conv1d, mamba_ssm)..."

# Determine compatible versions based on PyTorch version
TORCH_MAJOR=$(echo "$TORCH_VERSION" | cut -d. -f1)

if [ "$TORCH_MAJOR" -ge 2 ]; then
    echo "  PyTorch 2.x detected — installing compatible mamba_ssm and causal_conv1d..."

    # triton (latest works with PyTorch 2.x)
    pip install --quiet triton 2>/dev/null || echo "  Warning: triton install failed (non-critical)"

    # causal_conv1d >= 1.1.0 supports PyTorch 2.x
    pip install --quiet causal_conv1d 2>/dev/null || {
        echo "  Building causal_conv1d from source..."
        pip install --quiet causal_conv1d --no-build-isolation 2>/dev/null || echo "  Warning: causal_conv1d install failed"
    }

    # mamba_ssm >= 2.0 supports PyTorch 2.x
    pip install --quiet mamba_ssm 2>/dev/null || {
        echo "  Building mamba_ssm from source..."
        pip install --quiet mamba_ssm --no-build-isolation 2>/dev/null || echo "  Warning: mamba_ssm install failed"
    }
else
    echo "  PyTorch 1.x detected — installing pinned versions..."
    pip install --quiet "triton==2.0.0" 2>/dev/null || pip install --quiet triton
    pip install --quiet "causal_conv1d==1.0.0" 2>/dev/null || pip install --quiet "causal_conv1d==1.0.0" --no-build-isolation
    pip install --quiet "mamba_ssm==1.0.1" 2>/dev/null || pip install --quiet "mamba_ssm==1.0.1" --no-build-isolation
fi
echo "  Done."

# --------------------------------------------------
# Step 5: Download pre-trained weights
# --------------------------------------------------
echo "[5/7] Downloading pre-trained weights..."
mkdir -p "$WEIGHTS_DIR"

# Remove any LFS stub files (they're tiny text files, not real weights)
for f in "${WEIGHTS_DIR}/vmamba_tiny_e292.pth" "${WEIGHTS_DIR}/medsam_vit_b.pth"; do
    if [ -f "$f" ]; then
        SIZE=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "0")
        if [ "$SIZE" -lt 10000 ]; then
            echo "  Removing LFS stub: $f ($SIZE bytes)"
            rm -f "$f"
        fi
    fi
done

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

# Check weights exist and are real (not LFS stubs)
echo -n "  VMamba weights: "
if [ -f "${WEIGHTS_DIR}/vmamba_tiny_e292.pth" ]; then
    SIZE=$(stat -c%s "${WEIGHTS_DIR}/vmamba_tiny_e292.pth" 2>/dev/null || stat -f%z "${WEIGHTS_DIR}/vmamba_tiny_e292.pth")
    echo "OK (${SIZE} bytes)"
else
    echo "MISSING"
fi

echo -n "  MedSAM weights: "
if [ -f "${WEIGHTS_DIR}/medsam_vit_b.pth" ]; then
    SIZE=$(stat -c%s "${WEIGHTS_DIR}/medsam_vit_b.pth" 2>/dev/null || stat -f%z "${WEIGHTS_DIR}/medsam_vit_b.pth")
    echo "OK (${SIZE} bytes)"
else
    echo "MISSING"
fi

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
