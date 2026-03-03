# Angiography Demo App Runbook (macOS)

## 1. Repository and Branch

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography
git checkout demo/macos-player
```

## 2. Backend Setup (Python 3.11 recommended)

```bash
cd demo-app/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Frontend Setup

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography/demo-app/frontend
npm install
```

## 4. Prepare Curated Patient Data

Place source patient folders locally, then run:

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography
python3 demo-app/scripts/prepare_patient_data.py \
  --source-root /ABSOLUTE/PATH/TO/CURATED_MENDELEY_SUBSET \
  --output-root demo-app/data/patients \
  --max-patients 10 \
  --max-frames-per-patient 300
```

This writes `demo-app/data/patients/manifest.json`.

## 5. Start Demo

```bash
cd /Users/iwosmura/projects/angio-demo/Angiography
chmod +x demo-app/scripts/run_demo.sh
demo-app/scripts/run_demo.sh
```

Open `http://127.0.0.1:5173`.

## 6. API Smoke Checks

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/patients
```

Expected with prepared data: `/api/patients` returns at least one patient (for example `dataset` with `frameCount` > 0).

If the UI shows `Failed to fetch`, verify:

```bash
lsof -iTCP:8000 -sTCP:LISTEN -n -P
curl -i http://127.0.0.1:8000/api/health
```

Also confirm frontend API base is correct in `demo-app/frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## 7. Notes

- YOLO26s weights are expected at `YOLO26s/weights/best.pt`.
- YOLO26n is also available at `YOLO26n/weights/best.pt` (set `DEMO_MODEL_PATH` to switch).
- Data under `demo-app/data/patients/` is ignored by git except `manifest.json`.
- Default runtime remains YOLO26s unless `DEMO_MODEL_PATH` is overridden.
- The model rail supports switching between YOLO26s and YOLO26n at runtime via `/api/models/select`.
