#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

echo "=== SAM-VMNet Vast/H100 setup ==="

install_system_deps() {
  local packages=(
    git git-lfs wget curl unzip ca-certificates
    python3 python3-venv python3-pip python3-dev
    build-essential ninja-build cmake pkg-config
    libgl1 libglib2.0-0
  )

  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    else
      echo "[WARN] apt-get available but no root/sudo; skipping system deps install."
    fi
  else
    echo "[WARN] apt-get not found; install required packages manually if missing."
  fi
}

install_system_deps

git lfs install
# Pull LFS files only if this repo has lfs pointers.
if git lfs ls-files >/dev/null 2>&1; then
  git lfs pull || true
fi

VENV_PATH="${REPO_ROOT}/.venv"
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"

python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

python -m pip install \
  numpy opencv-python scikit-image scipy matplotlib tqdm \
  jupyterlab ipywidgets monai SimpleITK nibabel \
  timm==0.4.12 tensorboardX scikit-learn thop h5py medpy \
  einops yacs termcolor submitit packaging pycocotools

# VMamba custom kernels are mandatory for training/inference.
export MAX_JOBS="${MAX_JOBS:-8}"
python -m pip install --no-build-isolation causal-conv1d mamba-ssm

python - <<'PY'
import torch
import mamba_ssm
import causal_conv1d
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("torch:", torch.__version__)
print("mamba_ssm:", getattr(mamba_ssm, "__version__", "unknown"))
print("causal_conv1d:", getattr(causal_conv1d, "__version__", "unknown"))
print("selective_scan_fn import: OK")
PY

echo ""
echo "Setup complete. Activate environment with:"
echo "  source ${VENV_PATH}/bin/activate"
