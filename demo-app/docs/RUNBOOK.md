# Angiography Demo App Runbook (CADICA)

## 1. Supported Demo Path

`main` now supports the CADICA bbox demo only:

- dataset: CADICA `test` split
- models: `YOLO26m (CADICA)` and `YOLO26x (CADICA)`

The leakage-affected Mendeley / ARCADE demo is preserved on:

- `iwosmu/data-leakage-demo-archive`

## 2. Start the Demo

From `Angiography/`:

```bash
./demo-app/scripts/run_demo.sh start
```

That command will:

- create or repair the backend virtualenv
- install backend/frontend dependencies when missing or stale
- validate CADICA demo data and the active checkpoint
- prepare CADICA demo data automatically when possible
- start backend and frontend

Open `http://127.0.0.1:5173`.

## 3. Health / Diagnostics Commands

```bash
./demo-app/scripts/run_demo.sh doctor
./demo-app/scripts/run_demo.sh status
./demo-app/scripts/run_demo.sh stop
./demo-app/scripts/run_demo.sh restart
./demo-app/scripts/run_demo.sh bootstrap
```

## 4. Prepare CADICA Demo Data

```bash
python3 demo-app/scripts/prepare_cadica_demo_data.py \
  --cadica-root datasets/cadica/CADICA \
  --split-manifest models/yolo26m_cadica/manifests/patient_level_80_10_10_seed42.json \
  --output-root demo-app/data/patients \
  --split test \
  --force
```

What the script does:

- loads selected CADICA videos from the official split manifest
- keeps the `test` split by default
- writes one demo sequence per selected video
- writes bbox labels in YOLO format
- writes empty label files for selected negative frames

## 5. API Smoke Checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/patients
```

## 6. Environment Overrides

- `DEMO_HOST` default `127.0.0.1`
- `DEMO_BACKEND_PORT` default `8000`
- `DEMO_FRONTEND_PORT` default `5173`
- `DEMO_API_HOST` optional frontend API hostname
- `DEMO_DATA_DIR` default `demo-app/data/patients`
- `DEMO_MODEL_PATH` default `models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt`
- `DEMO_USE_MOCK_MODEL=1` forces mock inference
- `DEMO_AUTO_SETUP=0` disables auto-install and only runs checks
- `DEMO_PYTHON_BIN=/path/to/python3.11` pins the Python interpreter
- `DEMO_CADICA_ROOT` default `datasets/cadica/CADICA`
- `DEMO_CADICA_SPLIT_MANIFEST` default CADICA split manifest
- `DEMO_CADICA_SPLIT` default `test`

## 7. Notes

- Model switching no longer changes datasets because both supported checkpoints use CADICA.
- The active UI is bbox-only; mask endpoints remain unsupported in the CADICA demo path.
- Editing labels in the UI writes YOLO bbox annotations back into the prepared demo dataset.
