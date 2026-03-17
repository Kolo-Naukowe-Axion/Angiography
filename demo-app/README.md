# Angiography Demo App (CADICA)

FastAPI + React demo for visualizing CADICA coronary angiography sequences with two leakage-free YOLO checkpoints.

Supported demo models:

- `YOLO26m (CADICA)` from `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt`
- `YOLO26x (CADICA)` from `models/yolo26x/runs/cadica_selected_seed42_4090/weights/best.pt`

Legacy Mendeley / ARCADE demo support was moved to the archive branch:

- `iwosmu/data-leakage-demo-archive`

## Quick Start

From the `Angiography/` directory:

```bash
./demo-app/scripts/run_demo.sh start
```

This bootstraps dependencies, validates the CADICA demo dataset, validates the active model path, and starts backend (`:8000`) plus frontend (`:5173`).

Open `http://127.0.0.1:5173` when ready.

## Useful Commands

```bash
./demo-app/scripts/run_demo.sh doctor
./demo-app/scripts/run_demo.sh bootstrap
./demo-app/scripts/run_demo.sh status
./demo-app/scripts/run_demo.sh stop
./demo-app/scripts/run_demo.sh restart
```

## CADICA Demo Data Preparation

The supported mainline dataset is the CADICA `test` split.

```bash
python3 demo-app/scripts/prepare_cadica_demo_data.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json \
  --output-root demo-app/data/patients \
  --split test \
  --force
```

Defaults:

- one demo sequence per selected CADICA test video
- bbox labels only
- empty label files for selected negative frames
- symlinked images by default

## Optional Environment Overrides

- `DEMO_HOST` default `127.0.0.1`
- `DEMO_BACKEND_PORT` default `8000`
- `DEMO_FRONTEND_PORT` default `5173`
- `DEMO_API_HOST` optional frontend API hostname
- `DEMO_DATA_DIR` default `demo-app/data/patients`
- `DEMO_MODEL_PATH` default `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt`
- `DEMO_USE_MOCK_MODEL=1` forces mock bbox inference
- `DEMO_CADICA_ROOT` default `datasets/cadica/CADICA`
- `DEMO_CADICA_SPLIT_MANIFEST` default CADICA patient-level split manifest
- `DEMO_CADICA_SPLIT` default `test`

## API Endpoints

- `GET /api/health`
- `GET /api/models`
- `POST /api/models/select`
- `GET /api/patients`
- `GET /api/patients/{patient_id}/frames/{frame_index}`
- `POST /api/infer/frame`
- `POST /api/infer/prefetch`
- `GET /api/labels/{patient_id}/{frame_index}`
- `PUT /api/labels/{patient_id}/{frame_index}`
