# Angiography Demo App (macOS)

FastAPI + React demo for visualizing YOLO26s/YOLO26n classification on curated angiography patient frame sequences.

## Quick Start (One Command)

From the `Angiography/` directory:

```bash
./demo-app/scripts/run_demo.sh start
```

This command now bootstraps everything automatically:

- Creates/fixes `backend/.venv` if needed.
- Installs backend Python dependencies from `backend/requirements.txt`.
- Installs frontend npm dependencies from `frontend/package-lock.json`.
- Validates patient data manifest and frame folders.
- Validates model weights (or falls back to mock mode if weights are missing).
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
- `DEMO_MODEL_PATH` (default `YOLO26s/weights/best.pt`)
- `DEMO_USE_MOCK_MODEL=1` to force mock inference
- `DEMO_SOURCE_ROOT=/ABS/PATH` to auto-run patient data preparation when data is invalid/missing

## API Endpoints

- `GET /api/health`
- `GET /api/models`
- `GET /api/patients`
- `GET /api/patients/{patient_id}/frames/{frame_index}`
- `POST /api/infer/frame`
- `POST /api/infer/prefetch`
- `GET /api/labels/{patient_id}/{frame_index}`
