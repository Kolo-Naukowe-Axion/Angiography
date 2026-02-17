# SAM-VMNet Training on vast.ai — Step-by-Step Guide

Train SAM-VMNet on the Mendeley Angiographic Dataset for stenosis detection.

**Estimated cost:** ~$2-4 total on RTX 4090 (~$0.30-0.50/hr for ~5-9 hours)

---

## Prerequisites

- vast.ai account with ~$5-10 credit
- Rented instance: 1x RTX 4090, 24 GB VRAM, 50 GB disk
- Docker image: `pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel`

---

## Step 1: Connect to the Instance

On vast.ai dashboard → your instance → **Connect** → **Open** (Jupyter) or use SSH:

```bash
ssh -p <PORT> root@<VAST_IP>
```

---

## Step 2: Set Up Environment (~5-10 min)

```bash
cd /workspace
git clone https://github.com/qimingfan10/SAM-VMNet.git
cd SAM-VMNet
chmod +x setup_vastai.sh
./setup_vastai.sh
```

This installs all dependencies and downloads pre-trained backbone weights (~700 MB).

**Verify it worked — you should see:**
```
CUDA available: True
GPU device: NVIDIA GeForce RTX 4090
mamba_ssm: OK
VMamba weights: OK
MedSAM weights: OK
Forward pass: OK (output shape: torch.Size([1, 1, 256, 256]))
```

---

## Step 3: Download the Mendeley Dataset

Go to https://data.mendeley.com/datasets/ydrm75xywg/1 and download.

```bash
mkdir -p /workspace/mendeley_data
cd /workspace/mendeley_data

# Option A: If you have a direct download link
wget -O dataset.zip "PASTE_DOWNLOAD_LINK_HERE"
unzip dataset.zip

# Option B: Upload via Jupyter file browser
# (download to your PC first, then upload through the Jupyter UI)
```

After extracting, check what's inside:
```bash
ls /workspace/mendeley_data/
# Look for image files and annotation files (JSON, CSV, XML, etc.)
```

---

## Step 4: Prepare the Dataset (~2-5 min)

```bash
cd /workspace/SAM-VMNet
python prepare_mendeley.py \
    --data_dir /workspace/mendeley_data/ \
    --output_dir ./data/vessel/
```

This converts bounding box annotations → binary mask PNGs for SAM-VMNet.

**Expected output:**
```
Train: ~6660 images
Val:   ~832 images
Test:  ~833 images
```

**If it fails** with "could not parse annotations":
1. Run `ls /workspace/mendeley_data/` to see what files exist
2. Find the annotation file and specify it manually:
   ```bash
   python prepare_mendeley.py \
       --data_dir /workspace/mendeley_data/ \
       --output_dir ./data/vessel/ \
       --annotation_file /workspace/mendeley_data/YOUR_ANNOTATION_FILE.json
   ```

---

## Step 5: Quick Sanity Test (~1 min)

```bash
cd /workspace/SAM-VMNet
chmod +x run_training.sh
./run_training.sh --quick_test
```

Runs 1 epoch with batch size 2 to verify everything works end-to-end.

---

## Step 6: Run Full Training (~5-9 hours)

**Use tmux so training survives if your browser/SSH disconnects:**

```bash
tmux new -s training
cd /workspace/SAM-VMNet
./run_training.sh
```

**Detach from tmux:** press `Ctrl+B` then `D`
**Reattach later:** `tmux attach -t training`

### What happens inside:
| Phase | Duration | What it does |
|-------|----------|--------------|
| Branch 1 | ~3-6 hrs | Trains VM-UNet (200 epochs, batch 8) |
| Test predictions | ~5 min | Generates pred_masks for test set |
| Feature extraction | ~30-60 min | Computes MedSAM features (auto) |
| Branch 2 | ~1-2 hrs | Trains SAM-VMNet (100 epochs, batch 4) |

### If you need to resume after a crash:
Branch 1 and Branch 2 both auto-resume from `latest.pth` checkpoints.
Just re-run `./run_training.sh` (or `./run_training.sh --skip_branch1` if Branch 1 finished).

---

## Step 7: Save Your Trained Model

Your trained weights are at:
```
./result_branch1/checkpoints/best-epoch*-loss*.pth   # Branch 1 (intermediate)
./result_branch2/checkpoints/best-epoch*-loss*.pth   # Branch 2 (FINAL MODEL)
```

**The Branch 2 `.pth` file is your final trained model.** It's a single file (~100-200 MB).

### Option A: Download via Jupyter (easiest)
In the Jupyter file browser, navigate to `result_branch2/checkpoints/`, click the `.pth` file, download it.

### Option B: Tar everything and download
```bash
cd /workspace/SAM-VMNet
tar czf /workspace/trained_models.tar.gz \
    result_branch1/checkpoints/ \
    result_branch2/checkpoints/ \
    result_branch1/log/ \
    result_branch2/log/
# Download /workspace/trained_models.tar.gz via Jupyter file browser
```

### Option C: SCP from your local machine
```bash
# Run this on YOUR local machine (not vast.ai)
scp -P <PORT> root@<VAST_IP>:/workspace/SAM-VMNet/result_branch2/checkpoints/best-*.pth ./
```

Find SSH details on vast.ai: your instance → **Connect** → **SSH**

### Option D: Upload to Hugging Face (permanent cloud storage)
```bash
pip install huggingface_hub
huggingface-cli login  # paste your HF token

python -c "
from huggingface_hub import HfApi
api = HfApi()
# Create a repo first at https://huggingface.co/new
api.upload_file(
    path_or_fileobj='./result_branch2/checkpoints/BEST_CHECKPOINT_NAME.pth',
    path_in_repo='best_model.pth',
    repo_id='YOUR_USERNAME/sam-vmnet-stenosis',
    repo_type='model',
)
print('Uploaded!')
"
```

---

## Step 8: DESTROY the vast.ai Instance

**You are paying by the hour! Don't forget this!**

1. vast.ai dashboard → **Instances**
2. Click **Destroy** on your instance
3. Confirm

---

## Using Your Trained Model Later

To load the model for inference on new images:

```python
import torch
from models.vmunet.samvmnet import SAMVMNet

# Load model architecture
model = SAMVMNet(
    num_classes=1, input_channels=3,
    depths=[2,2,2,2], depths_decoder=[2,2,2,1],
    drop_path_rate=0.2,
    load_ckpt_path='./pre_trained_weights/vmamba_tiny_e292.pth',
)

# Load your trained weights
model.load_state_dict(torch.load('best-epochXX-lossX.XXXX.pth', map_location='cpu'))
model.eval()

# Inference: input is [batch, 3, 256, 256] image + [batch, 256, 64, 64] MedSAM feature
# Output is [batch, 1, 256, 256] sigmoid probability mask
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `mamba_ssm` install fails | Try `pip install mamba_ssm==1.0.1 --no-build-isolation` |
| Out of VRAM | Reduce batch size: `--batch_size 4` for Branch 1 |
| Training loss not decreasing | Normal for first ~10 epochs, should start dropping |
| `prepare_mendeley.py` can't parse annotations | Check the actual file format, specify `--annotation_file` |
| SSH disconnects during training | Use `tmux` (see Step 6) |
| Disk full | Delete `./data/vessel/*/feature/` and re-extract (they regenerate) |
