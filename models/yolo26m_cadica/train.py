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
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_batch_for_device(device: str) -> int:
    return 64 if device not in {"cpu", "mps"} else 4


def default_workers_for_device(device: str) -> int:
    if device == "cpu":
        return 2
    if device == "mps":
        return 2
    return 8


def default_cache_for_device(device: str) -> bool:
    return device not in {"cpu", "mps"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO26m on the prepared CADICA dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=str, default="yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Training batch size. Defaults to 64 on CUDA and 4 on Apple Silicon / CPU.",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Data loader workers. Defaults to 8 on CUDA and 2 on Apple Silicon / CPU.",
    )
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
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache images in RAM. Defaults to on for CUDA and off on Apple Silicon / CPU.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = args.device or detect_device()
    amp = args.amp if args.amp is not None else device != "mps"
    batch = args.batch if args.batch is not None else default_batch_for_device(device)
    workers = args.workers if args.workers is not None else default_workers_for_device(device)
    cache = args.cache if args.cache is not None else default_cache_for_device(device)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Ultralytics is not installed in this environment. "
            "Create the uv env from models/yolo26m_cadica/README.md and reinstall dependencies."
        ) from error

    if args.resume is not None:
        model = YOLO(str(args.resume))
        model.train(
            resume=True,
            device=device,
            batch=batch,
            workers=workers,
            amp=amp,
            cache=cache,
        )
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
        cache=cache,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
