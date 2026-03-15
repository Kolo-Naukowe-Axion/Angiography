from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "datasets" / "cadica" / "derived" / "yolo26_selected_seed42" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "models" / "yolo26m_cadica" / "runs"
DEFAULT_RUN_NAME = "cadica_selected_seed42"


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO26m on the prepared CADICA dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=str, default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt for resuming training.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable mixed precision. Defaults to off on MPS and on elsewhere.",
    )
    args = parser.parse_args()

    device = args.device or detect_device()
    amp = args.amp if args.amp is not None else device != "mps"

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed in this environment. "
            "Create the uv env from models/yolo26m_cadica/README.md and reinstall dependencies."
        ) from error

    if args.resume is not None:
        model = YOLO(str(args.resume))
        model.train(resume=True)
        return

    data_path = args.data.resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml does not exist: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        device=device,
        amp=amp,
        project=str(args.project.resolve()),
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    main()
