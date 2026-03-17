# YOLO26m on CADICA

This folder contains the CADICA-specific YOLO26m training workflow and the current state of the run so far.

## Objective

Train `yolo26m` from Ultralytics on the CADICA coronary angiography dataset for single-class stenosis detection.

- Task: object detection
- Class count: `1`
- Class name: `stenosis`
- Base weights: `yolo26m.pt`
- Current training device: Apple Silicon `mps`

## Source Data

Raw CADICA data lives in:

- `datasets/cadica/CADICA`

Prepared YOLO dataset lives in:

- `datasets/cadica/derived/yolo26_selected_seed42`

The tracked split preset used for this experiment is:

- `models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json`

That preset stores the exact seed-42 patient assignment in git. The prep code expands it into the full `selected_videos` manifest at runtime from `CADICA/selectedVideos`, so anyone with the raw dataset can reproduce the split 1:1 without committing `datasets/`.

## Dataset Adjustments

The most important dataset choice in this experiment is that training uses **selected keyframes only**, not every extracted frame in `selectedVideos`.

Why that matters:

- CADICA supervision is defined by each video's `*_selectedFrames.txt`.
- Lesion videos contain more raw PNG frames than labeled keyframes.
- If all raw frames were used, many unlabeled lesion-video frames would be treated as negatives.
- That would inject false negatives and poison training.

So the prep flow does the following:

1. Reads the tracked patient-level `train` / `val` / `test` split preset.
2. For each selected video, reads only the frame IDs listed in `*_selectedFrames.txt`.
3. Converts CADICA pixel boxes from `x y w h label` into YOLO normalized boxes:
   - `0 x_center y_center width height`
4. Creates empty YOLO label files for selected keyframes from non-lesion videos.
5. Writes a clean YOLO dataset with:
   - `images/train`, `images/val`, `images/test`
   - `labels/train`, `labels/val`, `labels/test`
   - `data.yaml`
   - `summary.json`

Image materialization mode for the current prepared dataset is:

- `symlink`

That keeps local storage smaller and makes re-preparation faster on the same machine.

## Prepared Dataset Statistics

Current prepared dataset summary:

| Split | Images | Positive images | Negative images | Bounding boxes |
|---|---:|---:|---:|---:|
| Train | 4696 | 3191 | 1505 | 4940 |
| Val | 634 | 304 | 330 | 534 |
| Test | 796 | 501 | 295 | 687 |
| Total | 6126 | 3996 | 2130 | 6161 |

Split policy notes:

- Train patients: `34`
- Val patients: `4`
- Test patients: `4`
- Seed: `42`
- Strategy: patient-level split to avoid leakage across frames and videos from the same patient

## Training Configuration

The current run was started from:

- `models/yolo26m_cadica/train.py`

Resolved training arguments for the current run:

- `model=yolo26m.pt`
- `data=/Users/iwosmura/projects/angio-demo/Angiography/datasets/cadica/derived/yolo26_selected_seed42/data.yaml`
- `epochs=300`
- `batch=16`
- `imgsz=512`
- `workers=4`
- `device=mps`
- `optimizer=AdamW`
- `lr0=0.001`
- `lrf=0.01`
- `patience=50`
- `amp=false`
- `project=/Users/iwosmura/projects/angio-demo/Angiography/models/yolo26m_cadica/runs`
- `name=cadica_selected_seed42`

Augmentations in the current run:

- `hsv_h=0.015`
- `hsv_s=0.4`
- `hsv_v=0.3`
- `degrees=15`
- `translate=0.1`
- `scale=0.5`
- `flipud=0.5`
- `fliplr=0.5`
- `mosaic=1.0`
- `mixup=0.1`

## Current Run Status

Run directory:

- `models/yolo26m_cadica/runs/cadica_selected_seed42`

Current run state:

- Training was started locally on Apple Silicon.
- The run was then stopped gracefully to avoid power issues while the laptop was on a weaker charger.
- The latest checkpoint was preserved and the run is resumable.
- Training has since been resumed from `last.pt`.

Checkpoint files:

- Best weights so far: `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt`
- Latest resumable checkpoint: `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt`

Checkpoint timestamps:

- `best.pt`: March 15, 2026 at `08:00:22`
- `last.pt`: March 15, 2026 at `08:27:49`

## Current Results

The run has `17` logged epochs in `results.csv`.

Best validation metrics so far were reached at **epoch 13**:

| Metric | Value |
|---|---:|
| Precision | 0.29716 |
| Recall | 0.24307 |
| mAP@0.50 | 0.17146 |
| mAP@0.50:0.95 | 0.06098 |

Latest logged validation metrics at **epoch 17**:

| Metric | Value |
|---|---:|
| Precision | 0.14582 |
| Recall | 0.24906 |
| mAP@0.50 | 0.08231 |
| mAP@0.50:0.95 | 0.02642 |

Latest computed validation mIoU from `last.pt` at **epoch 17**:

| Metric | Value |
|---|---:|
| mean IoU | 0.66851 |
| Matched pairs | 29 |
| Ground-truth boxes | 534 |
| Predicted boxes | 121 |
| Images | 634 |

Interpretation of the current state:

- The model is still early in training.
- Validation quality improved through epoch 13, then regressed over the next few logged epochs.
- With the current `patience=50`, this run would not early-stop soon on its own.
- The saved `best.pt` is the checkpoint to use for evaluation right now.
- The saved `last.pt` is the checkpoint to use for resuming training.

## Available Artifacts

Current run artifacts already on disk:

- `models/yolo26m_cadica/runs/cadica_selected_seed42/args.yaml`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/results.csv`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/labels.jpg`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/train_batch0.jpg`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/train_batch1.jpg`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/train_batch2.jpg`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt`
- `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt`

These artifacts are enough to:

- inspect batch-level label sanity
- resume training
- compare `best.pt` and `last.pt`
- continue documenting the experiment as new epochs are completed

## Reproduce The Workflow

### 1. Create the local environment

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography
uv venv --python 3.12 .venv-yolo26m
source .venv-yolo26m/bin/activate
uv pip install -r models/yolo26m_cadica/requirements-macos.txt
```

### 2. Prepare the CADICA YOLO dataset

```bash
python3 models/yolo26m_cadica/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42
```

If you want the old dataset-local manifest file too, export it exactly from the tracked preset:

```bash
python3 models/yolo26m_cadica/scripts/export_cadica_split_manifest.py \
  --cadica-root datasets/cadica/CADICA
```

### 3. Verify the prepared dataset

```bash
python3 models/yolo26m_cadica/scripts/verify_cadica_selected.py \
  --dataset-root datasets/cadica/derived/yolo26_selected_seed42
```

### 4. Start training

```bash
python3 models/yolo26m_cadica/train.py \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml
```

### 5. Resume the interrupted run

```bash
python3 models/yolo26m_cadica/train.py \
  --resume models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt
```

### 6. Watch labeled training metrics

```bash
python3 models/yolo26m_cadica/scripts/watch_results.py --follow
```

### 7. Compute mIoU for a checkpoint

```bash
python3 models/yolo26m_cadica/scripts/compute_mean_iou.py \
  --weights models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml \
  --split val \
  --device mps \
  --output models/yolo26m_cadica/runs/cadica_selected_seed42/current_iou.json
```

### 8. Track mIoU periodically during training

```bash
python3 models/yolo26m_cadica/scripts/periodic_iou_eval.py \
  --run-dir models/yolo26m_cadica/runs/cadica_selected_seed42 \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml \
  --device mps \
  --every 5
```

## Notes For The Next Update

When training resumes, this documentation should be updated with:

- the final epoch reached
- the final best epoch
- final validation metrics
- test-set evaluation results
- qualitative prediction examples
- any hyperparameter changes such as lower patience or different batch size
