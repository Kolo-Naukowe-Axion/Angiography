# YOLO26x on CADICA (Vast.ai RTX 4090)

This workflow trains the largest Ultralytics YOLO26 detection checkpoint, `yolo26x.pt`, on the prepared CADICA selected-keyframe dataset and is tuned for training on a Vast.ai RTX 4090 instance.

## Why this workflow is separate

The existing [`models/yolo26m_cadica`](../yolo26m_cadica) workflow remains the canonical CADICA preparation path. This `yolo26x` path reuses the same CADICA dataset preparation flow, but its defaults are centered on a single RTX 4090 with CUDA.

Expected prepared keyframe counts:

- `train`: 4696 images, 3191 positive
- `val`: 634 images, 304 positive
- `test`: 796 images, 501 positive

## Dataset prep

This workflow reuses the existing CADICA selected-keyframe prep script from [`models/yolo26m_cadica`](../yolo26m_cadica).

Symlink mode is the default and is the fastest option for local training on the same machine:

```bash
python3 models/yolo26m_cadica/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42
```

If you want a fully portable copy instead of symlinks:

```bash
python3 models/yolo26m_cadica/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42_copy \
  --copy-images
```

## Verify the dataset

```bash
python3 models/yolo26m_cadica/scripts/verify_cadica_selected.py \
  --dataset-root datasets/cadica/derived/yolo26_selected_seed42
```

This checks image/label parity, YOLO normalization, exact split counts, and split membership against the original CADICA split manifest.

## Vast.ai RTX 4090 training

On CUDA the trainer now auto-selects stronger defaults:

- `device=0`
- `batch=16`
- `workers=12`
- `amp=True`
- `cache=True`

Remote helper files:

- `models/yolo26x/scripts/run_vast_4090_pipeline.sh`
- `models/yolo26x/scripts/vast_4090_sync_and_start.sh`
- `models/yolo26x/scripts/vast_4090_follow.sh`
- `models/yolo26x/scripts/periodic_iou_eval.py`
- `models/yolo26x/scripts/monitor_training.py`

The periodic IoU evaluator computes mean IoU on the validation split every 10 epochs by default, while Ultralytics continues to compute mAP each validation epoch in `results.csv`.

### Host-side launch

From your Mac, after configuring `vastai` and SSH:

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography
bash models/yolo26x/scripts/vast_4090_sync_and_start.sh <instance_id>
```

This syncs the repo, copies the prepared CADICA dataset with image symlinks dereferenced, verifies the dataset on the instance, and starts:

- the main training process
- a periodic IoU sidecar evaluator

### Live tracking from your Mac terminal

Monitor training progress, including latest mAP and latest periodic mean IoU:

```bash
bash models/yolo26x/scripts/vast_4090_follow.sh <instance_id>
```

Tail the raw training log directly:

```bash
ssh "$(/tmp/vastai-cli/bin/vastai ssh-url <instance_id> | sed 's#ssh://##')" \
  "tail -f /workspace/Angiography/models/yolo26x/runs/cadica_selected_seed42_4090/train.log"
```

Tail the periodic IoU log directly:

```bash
ssh "$(/tmp/vastai-cli/bin/vastai ssh-url <instance_id> | sed 's#ssh://##')" \
  "tail -f /workspace/Angiography/models/yolo26x/runs/cadica_selected_seed42_4090/periodic_iou.log"
```

### Remote defaults

Default training settings for the 4090 path:

- `model=yolo26x.pt`
- `imgsz=512`
- `epochs=300`
- `batch=16`
- `workers=12`
- `optimizer=AdamW`
- `lr0=0.001`
- `lrf=0.01`
- `patience=50`
- `device=0`
- `amp=True`
- `cache=True`

If you hit GPU memory limits, reduce settings in this order:

1. `batch 16 -> 12 -> 8 -> 4`
2. `imgsz 512 -> 448`

Training outputs land under:

`models/yolo26x/runs/cadica_selected_seed42_4090`

Best weights:

`models/yolo26x/runs/cadica_selected_seed42_4090/weights/best.pt`

Last checkpoint:

`models/yolo26x/runs/cadica_selected_seed42_4090/weights/last.pt`

## Manual training

```bash
python models/yolo26x/train.py \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml \
  --device 0
```

Training outputs land under:

`models/yolo26x/runs/cadica_selected_seed42_4090`

Best weights:

`models/yolo26x/runs/cadica_selected_seed42_4090/weights/best.pt`

Last checkpoint:

`models/yolo26x/runs/cadica_selected_seed42_4090/weights/last.pt`

## Resume training

```bash
python models/yolo26x/train.py \
  --resume models/yolo26x/runs/cadica_selected_seed42_4090/weights/last.pt
```
