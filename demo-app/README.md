# Angiography Demo App (macOS)

FastAPI + React demo for visualizing model outputs on curated angiography patient frame sequences.

Supported model/dataset modes:

- `YOLO26s` / `YOLO26n` on Mendeley-style bbox-labeled sequences.
- `SAM-VMNet (ARCADE)` on precomputed binary mask predictions + mask ground truth.

## Quick Start (One Command)

From the `Angiography/` directory:

```bash
./demo-app/scripts/run_demo.sh start
```

This command bootstraps everything automatically:

- Creates/fixes `backend/.venv` if needed.
- Installs backend Python dependencies from `backend/requirements.txt`.
- Installs frontend npm dependencies from `frontend/package-lock.json`.
- Validates patient data manifest and frame folders.
- Validates active model path (or falls back to mock mode if missing).
- Reports SAM-VMNet precomputed-mask readiness.
- Starts backend (`:8000`) and frontend (`:5173`).

Open `http://127.0.0.1:5173` when ready.

## Useful Commands

```bash
./demo-app/scripts/run_demo.sh doctor     # Read-only env + health checks
./demo-app/scripts/run_demo.sh bootstrap  # Install/update deps and validate data/model
./demo-app/scripts/run_demo.sh status     # Port + endpoint status
./demo-app/scripts/run_demo.sh stop       # Stop demo processes
./demo-app/scripts/run_demo.sh restart    # Restart both services
```

## Optional Environment Overrides

- `DEMO_HOST` (default `127.0.0.1`)
- `DEMO_BACKEND_PORT` (default `8000`)
- `DEMO_FRONTEND_PORT` (default `5173`)
- `DEMO_API_HOST` (optional frontend API hostname; useful when `DEMO_HOST=0.0.0.0`)
- `DEMO_DATA_DIR` (default `demo-app/data/patients`)
- `DEMO_MODEL_PATH` (default `models/yolo26s/weights/best.pt`)
- `DEMO_USE_MOCK_MODEL=1` to force mock inference
- `DEMO_SOURCE_ROOT=/ABS/PATH` to auto-run Mendeley bbox patient data preparation when data is invalid/missing

## ARCADE + SAM-VMNet Preparation

The demo serves SAM-VMNet in **precomputed mask mode**.

### 1) Prepare ARCADE frame + GT mask data into demo patient format

```bash
python3 demo-app/scripts/prepare_arcade_data.py \
  --source-root datasets/arcade/data \
  --output-root demo-app/data/patients
```

Defaults are medium curated size (`10` sequences, up to `240` frames/sequence), configurable by CLI flags.

### 2) Precompute SAM-VMNet prediction masks for prepared ARCADE patients

```bash
python3 demo-app/scripts/precompute_sam_vmnet_masks.py \
  --data-root demo-app/data/patients \
  --checkpoint models/sam_vmnet/pre_trained_weights/best-epoch142-loss0.3230.pth
```

Use `--dry-run` to inspect eligible patients and `--overwrite` to regenerate masks.

## API Endpoints

- `GET /api/health`
- `GET /api/models`
- `POST /api/models/select`
- `GET /api/patients` (filtered to active model dataset)
- `GET /api/patients/{patient_id}/frames/{frame_index}`
- `GET /api/patients/{patient_id}/frames/{frame_index}/masks/{source}` where `{source}` is `prediction` or `ground_truth`
- `POST /api/infer/frame`
- `POST /api/infer/prefetch`
- `GET /api/labels/{patient_id}/{frame_index}`
- `PUT /api/labels/{patient_id}/{frame_index}` (bbox datasets only; mask datasets are read-only)
