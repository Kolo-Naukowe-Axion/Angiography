# Angiography Demo Frontend

React + TypeScript interface for the macOS localhost demo.

## Features

- Patient timeline player with play/pause, frame scrub, step controls, and speed presets.
- Live YOLO26s overlay rendering per frame.
- Confidence threshold slider (`0.10` to `0.90`) without forcing re-inference.
- Optional ground-truth overlay toggle (if labels are available).
- Per-frame quality metric: `IoU (frame)` computed from inference + labels.
- IoU metric value is independent of the confidence threshold slider.
- Model rail includes YOLO26s (default) and selectable YOLO26n.

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
