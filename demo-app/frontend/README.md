# Angiography Demo Frontend (CADICA)

React + TypeScript interface for the CADICA localhost demo.

## Features

- CADICA sequence player with play/pause, frame scrub, step controls, and speed presets
- live bbox overlays for `YOLO26m (CADICA)` and `YOLO26x (CADICA)`
- confidence threshold slider without re-running inference
- optional ground-truth overlay toggle when labels are available
- per-frame `IoU (frame)` metric from prediction vs ground truth
- inline bbox annotation editing and save flow

## Environment

Create `.env.local` if you want a custom API URL:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Run

```bash
npm install
npm run dev
```

## Quality Checks

```bash
npm run lint
npm run build
```
