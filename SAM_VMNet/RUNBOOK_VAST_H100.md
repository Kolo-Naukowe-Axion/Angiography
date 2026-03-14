# SAM-VMNet on Vast.ai (H100 SXM) Runbook

## Scope
This flow trains full SAM-VMNet (`branch1 -> pred_masks -> branch2`) on ARCADE, with:
- Cloud-side ARCADE download from Zenodo (`record 10390295`)
- Leakage audit based on exact image-content overlap across splits
- Auto-rebuild fallback (`patient` grouped) if strict leakage is detected
- H100-oriented mixed precision defaults (`bf16`, TF32)

Important: current Branch2 feature fusion in `models/vmunet/vmamba.py` is wired for `256x256`
inputs (SAM feature map is pooled to `8x8` before fusion), so this pipeline keeps `input_size=256`.

## 1) Run Inside an Existing Vast Instance

```bash
cd /workspace/Angiography/SAM_VMNet
bash run_vast_h100_pipeline.sh
```

### Optional Environment Overrides

```bash
export GPU_ID=0
export BRANCH1_BATCH=128
export BRANCH2_BATCH=64
export BRANCH1_EPOCHS=160
export BRANCH2_EPOCHS=160
export NUM_WORKERS=16
export AMP_DTYPE=bf16
export AUDIT_POLICY=auto-rebuild
export GROUP_LEVEL=patient
bash run_vast_h100_pipeline.sh
```

## 2) Host-Side Vast CLI Helpers

These scripts are optional convenience wrappers.

```bash
# Launch a new H100 instance (requires vastai CLI + API key)
cd /path/to/Angiography/SAM_VMNet
bash scripts/vast_h100_launch.sh

# Run pipeline on an existing instance
bash scripts/vast_h100_execute_pipeline.sh <instance_id>
```

## 3) Key Outputs

- Dataset root: `datasets/arcade/data/vessel`
- Leakage/index reports: `datasets/arcade/data/vessel_meta`
- Branch1 outputs: `SAM_VMNet/runs/branch1_h100`
- Branch2 outputs: `SAM_VMNet/runs/branch2_h100`
- Branch1 test predictions for branch2 features: `datasets/arcade/data/vessel/test/pred_masks`

## 4) Failure Handling

1. Setup/build failure (`mamba-ssm`, `causal-conv1d`):
- Re-run `bash setup_vast.sh` in a CUDA *devel* image.
- Confirm `nvcc --version` is available.

2. OOM during training:
- Pipeline auto-retries lower batch sizes.
- You can pre-set smaller values with `BRANCH1_BATCH` / `BRANCH2_BATCH`.

3. Leakage audit failure:
- With `AUDIT_POLICY=auto-rebuild`, split is rebuilt automatically.
- Inspect `datasets/arcade/data/vessel_meta/final_split_audit.json`.

## 5) Resume Strategy

- `train_branch1.py` resumes from `<work_dir>/checkpoints/latest.pth`.
- `train_branch2.py` resumes from `<work_dir>/checkpoints/latest.pth`.
- Feature generation skips existing `.pt` files in each split.
