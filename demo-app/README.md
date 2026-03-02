# Angiography Demo App (macOS)

FastAPI + React demo for visualizing live YOLO26s classification on curated angiography patient frame sequences.

## Directory Layout

- `backend/`: FastAPI service, inference cache/prefetch worker, tests.
- `frontend/`: React/Vite player UI and overlays.
- `data/patients/`: curated patient timeline data + `manifest.json`.
- `scripts/`: data prep and local run helper.
- `docs/`: runbook and operational notes.

## API Endpoints

- `GET /api/health`
- `GET /api/models`
- `GET /api/patients`
- `GET /api/patients/{patient_id}/frames/{frame_index}`
- `POST /api/infer/frame`
- `POST /api/infer/prefetch`
- `GET /api/labels/{patient_id}/{frame_index}`

## Quick Start

Follow `docs/RUNBOOK.md` for full setup and launch.
