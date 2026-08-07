# Angiography

Research and demo tooling for coronary-angiography stenosis detection. The repository combines patient-level dataset preparation, object-detection experiments, segmentation baselines, and a FastAPI/React application for reviewing sequences, running inference, and editing annotations.

## Repository

- [`models/`](models) — YOLO and SAM-VMNet experiment families.
- [`datasets/`](datasets) — dataset adapters and split metadata.
- [`demo-app/`](demo-app) — FastAPI backend, React frontend, and orchestration scripts.
- [`dataset_notebook.ipynb`](dataset_notebook.ipynb) — dataset exploration and preparation.

The supported demo path uses CADICA test sequences and patient-level split manifests.

## Demo

From the repository root:

```bash
./demo-app/scripts/run_demo.sh start
```

See [`demo-app/README.md`](demo-app/README.md) for data preparation, model selection, and API endpoints.

## Scope

This is research software for model evaluation and interactive review. It is not a certified medical device and must not be used for clinical decisions.
