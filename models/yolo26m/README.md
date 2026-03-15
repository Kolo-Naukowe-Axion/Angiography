# YOLO26m on CADICA (macOS / Apple Silicon)

This workflow prepares a clean YOLO-format CADICA dataset from `selectedVideos` keyframes only, validates it, and trains `yolo26m` locally on Apple Silicon with `uv`.

## Why this prep flow exists

CADICA supervision is defined by each video's `*_selectedFrames.txt` file. The raw `selectedVideos/*/input/` folders contain more frames than the labeled keyframes, so training on all frames would incorrectly turn unlabeled lesion-video frames into background examples. This workflow prepares only the selected keyframes.

Expected prepared keyframe counts:

- `train`: 4696 images, 3191 positive
- `val`: 634 images, 304 positive
- `test`: 796 images, 501 positive

## Environment setup

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography

uv venv --python 3.12 .venv-yolo26m
source .venv-yolo26m/bin/activate
uv pip install -r models/yolo26m/requirements-macos.txt
```

The repo's default Homebrew Python is `3.14`, which is not the target for this training workflow.

## Prepare the dataset

Symlink mode is the default and is the fastest option for local training on the same machine:

```bash
python models/yolo26m/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest datasets/cadica/CADICA/splits/patient_level_80_10_10_seed42/manifest.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42
```

If you want a fully portable copy instead of symlinks:

```bash
python models/yolo26m/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest datasets/cadica/CADICA/splits/patient_level_80_10_10_seed42/manifest.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42_copy \
  --copy-images
```

## Verify the dataset

```bash
python models/yolo26m/scripts/verify_cadica_selected.py \
  --dataset-root datasets/cadica/derived/yolo26_selected_seed42
```

This checks image/label parity, YOLO normalization, exact split counts, and split membership against the original CADICA split manifest.

## Smoke test

Run a short 1-epoch sanity check before the full training job:

```bash
python models/yolo26m/train.py \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml \
  --epochs 1 \
  --batch 2 \
  --workers 0 \
  --device mps
```

## Full training

```bash
python models/yolo26m/train.py \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml
```

Default training settings are tuned for local Apple Silicon stability:

- `model=yolo26m.pt`
- `imgsz=512`
- `epochs=300`
- `batch=16`
- `workers=4`
- `optimizer=AdamW`
- `lr0=0.001`
- `lrf=0.01`
- `patience=50`
- `device=mps` when available, otherwise `cpu`
- `amp=False` by default on MPS

Training outputs land under:

`models/yolo26m/runs/cadica_selected_seed42`

Best weights:

`models/yolo26m/runs/cadica_selected_seed42/weights/best.pt`

Last checkpoint:

`models/yolo26m/runs/cadica_selected_seed42/weights/last.pt`

## Resume training

```bash
python models/yolo26m/train.py \
  --resume models/yolo26m/runs/cadica_selected_seed42/weights/last.pt
```
