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

The split manifest used for this experiment is:

- `datasets/cadica/CADICA/splits/patient_level_80_10_10_seed42/manifest.json`

## Dataset Adjustments

The most important dataset choice in this experiment is that training uses **selected keyframes only**, not every extracted frame in `selectedVideos`.

Why that matters:

- CADICA supervision is defined by each video's `*_selectedFrames.txt`.
- Lesion videos contain more raw PNG frames than labeled keyframes.
- If all raw frames were used, many unlabeled lesion-video frames would be treated as negatives.
- That would inject false negatives and poison training.

So the prep flow does the following:

1. Reads the official patient-level `train` / `val` / `test` split manifest.
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
python models/yolo26m_cadica/scripts/prepare_cadica_selected.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest datasets/cadica/CADICA/splits/patient_level_80_10_10_seed42/manifest.json \
  --output-root datasets/cadica/derived/yolo26_selected_seed42
```

### 3. Verify the prepared dataset

```bash
python models/yolo26m_cadica/scripts/verify_cadica_selected.py \
  --dataset-root datasets/cadica/derived/yolo26_selected_seed42
```

### 4. Start training

```bash
python models/yolo26m_cadica/train.py \
  --data datasets/cadica/derived/yolo26_selected_seed42/data.yaml
```

### 5. Resume the interrupted run

```bash
python models/yolo26m_cadica/train.py \
  --resume models/yolo26m_cadica/runs/cadica_selected_seed42/weights/last.pt
```

## Notes For The Next Update

When training resumes, this documentation should be updated with:

- the final epoch reached
- the final best epoch
- final validation metrics
- test-set evaluation results
- qualitative prediction examples
- any hyperparameter changes such as lower patience or different batch size
