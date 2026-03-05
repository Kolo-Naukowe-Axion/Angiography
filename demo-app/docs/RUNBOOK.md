# Angiography Demo App Runbook (macOS)

## 1. Start the Demo

From `Angiography/`:

```bash
./demo-app/scripts/run_demo.sh start
```

That single command will:

- Create/fix the backend virtualenv.
- Install backend/frontend dependencies when missing or stale.
- Validate data and model prerequisites.
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

If your curated data is not already prepared:

```bash
python3 demo-app/scripts/prepare_patient_data.py \
  --source-root /ABSOLUTE/PATH/TO/CURATED_MENDELEY_SUBSET \
  --output-root demo-app/data/patients \
  --max-patients 10 \
  --max-frames-per-patient 300
```

Or let `start` auto-prepare by setting:

```bash
DEMO_SOURCE_ROOT=/ABSOLUTE/PATH/TO/CURATED_MENDELEY_SUBSET ./demo-app/scripts/run_demo.sh start
```

## 4. API Smoke Checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/patients
```

## 5. Environment Overrides

- `DEMO_HOST` (default `127.0.0.1`)
- `DEMO_BACKEND_PORT` (default `8000`)
- `DEMO_FRONTEND_PORT` (default `5173`)
- `DEMO_API_HOST` (optional frontend API hostname; useful when `DEMO_HOST=0.0.0.0`)
- `DEMO_DATA_DIR` (default `demo-app/data/patients`)
- `DEMO_MODEL_PATH` (default `YOLO26s/weights/best.pt`)
- `DEMO_USE_MOCK_MODEL=1` to force mock inference
- `DEMO_AUTO_SETUP=0` to disable auto-install and run checks only
- `DEMO_PYTHON_BIN=/path/to/python3.11` to pin Python interpreter

## 6. Notes

- Default model path is `YOLO26s/weights/best.pt`.
- If the model path is missing and `DEMO_AUTO_USE_MOCK_MODEL=1` (default), launcher falls back to mock model mode.
- Frontend API base is set automatically by the launcher for dev mode.
