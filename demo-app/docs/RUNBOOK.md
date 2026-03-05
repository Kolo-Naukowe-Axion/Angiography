# Angiography Demo App Runbook (macOS)

## 1. Start the Demo

From `Angiography/`:

```bash
./demo-app/scripts/run_demo.sh start
```

That single command will:

- Create/fix the backend virtualenv.
- Install backend/frontend dependencies when missing or stale.
- Validate data and active model prerequisites.
- Report SAM-VMNet precomputed mask readiness for ARCADE.
- Start backend + frontend and print ready URLs.

Open `http://127.0.0.1:5173`.

## 2. Health / Diagnostics Commands

```bash
./demo-app/scripts/run_demo.sh doctor
./demo-app/scripts/run_demo.sh status
./demo-app/scripts/run_demo.sh stop
./demo-app/scripts/run_demo.sh restart
./demo-app/scripts/run_demo.sh bootstrap
```

## 3. Data Preparation (Only If Needed)

### Mendeley/BBox (existing flow)

```bash
python3 demo-app/scripts/prepare_patient_data.py \
  --source-root /ABSOLUTE/PATH/TO/CURATED_MENDELEY_SUBSET \
  --output-root demo-app/data/patients \
  --max-patients 10 \
  --max-frames-per-patient 300
```

### ARCADE/Mask (new flow)

```bash
python3 demo-app/scripts/prepare_arcade_data.py \
  --source-root /ABSOLUTE/PATH/TO/ARCADE_OR_ARCADE_SYNTAX \
  --output-root demo-app/data/patients
```

Default profile is medium curated size (`10` sequences, up to `240` frames/sequence). Use script flags to tune size.

## 4. SAM-VMNet Mask Precompute

Run once after ARCADE prep to enable `SAM-VMNet (ARCADE)` model card in ready mode:

```bash
python3 demo-app/scripts/precompute_sam_vmnet_masks.py \
  --data-root demo-app/data/patients \
  --checkpoint SAM_VMNet/pre_trained_weights/best-epoch142-loss0.3230.pth
```

Useful flags:

- `--dry-run` for eligibility inspection
- `--patient-id <id>` to target one patient
- `--overwrite` to regenerate existing predictions

## 5. API Smoke Checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/patients
```

## 6. Environment Overrides

- `DEMO_HOST` (default `127.0.0.1`)
- `DEMO_BACKEND_PORT` (default `8000`)
- `DEMO_FRONTEND_PORT` (default `5173`)
- `DEMO_API_HOST` (optional frontend API hostname; useful when `DEMO_HOST=0.0.0.0`)
- `DEMO_DATA_DIR` (default `demo-app/data/patients`)
- `DEMO_MODEL_PATH` (default `YOLO26s/weights/best.pt`)
- `DEMO_USE_MOCK_MODEL=1` to force mock inference for YOLO modes
- `DEMO_AUTO_SETUP=0` to disable auto-install and run checks only
- `DEMO_PYTHON_BIN=/path/to/python3.11` to pin Python interpreter

## 7. Notes

- Model selection is dataset-aware: switching model updates compatible patient list.
- BBox editing (`PUT /api/labels`) is enabled for bbox datasets only.
- ARCADE mask labels are read-only in current UI scope.
