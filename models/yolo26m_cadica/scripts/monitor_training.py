from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


RESULTS_COLS = {
    "epoch": "epoch",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "box_loss": "train/box_loss",
    "cls_loss": "train/cls_loss",
}

BAR_WIDTH = 30


def read_last_row(csv_path: Path) -> tuple[dict[str, str] | None, int]:
    if not csv_path.exists():
        return None, 0
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, 0
    return {key.strip(): value.strip() for key, value in rows[-1].items()}, len(rows)


def read_latest_iou(iou_csv: Path) -> dict[str, str] | None:
    if not iou_csv.exists():
        return None
    with iou_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def bar(current: int, total: int, width: int = BAR_WIDTH) -> str:
    filled = int(width * current / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_metric(row: dict[str, str], key: str) -> str:
    raw = row.get(key, "")
    try:
        return f"{float(raw):.4f}"
    except Exception:
        return "  —  "


def main() -> None:
    parser = argparse.ArgumentParser(description="Track YOLO26m CADICA training progress.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()

    results_csv = args.run_dir / "results.csv"
    iou_csv = args.run_dir / "iou_metrics.csv"

    header = (
        f"{'Epoch':>9}  {'Progress':<{BAR_WIDTH+2}}  "
        f"{'mAP50':>7}  {'mAP50-95':>8}  {'Prec':>6}  {'Recall':>6}  "
        f"{'mIoU':>6}  {'IoU@Ep':>6}  {'box_loss':>8}  {'cls_loss':>8}"
    )
    print(header)
    print("-" * len(header))

    while True:
        row, _ = read_last_row(results_csv)
        latest_iou = read_latest_iou(iou_csv)
        if row:
            epoch = int(float(row.get("epoch", 0)))
            progress = bar(epoch, args.epochs)
            iou_value = "  —  "
            iou_epoch = "  —  "
            if latest_iou is not None:
                iou_value = format_metric(latest_iou, "mean_iou")
                iou_epoch = latest_iou.get("epoch", "  —  ")

            line = (
                f"{epoch:>4}/{args.epochs:<4}  {progress}  "
                f"{format_metric(row, RESULTS_COLS['mAP50']):>7}  "
                f"{format_metric(row, RESULTS_COLS['mAP50-95']):>8}  "
                f"{format_metric(row, RESULTS_COLS['precision']):>6}  "
                f"{format_metric(row, RESULTS_COLS['recall']):>6}  "
                f"{iou_value:>6}  {iou_epoch:>6}  "
                f"{format_metric(row, RESULTS_COLS['box_loss']):>8}  "
                f"{format_metric(row, RESULTS_COLS['cls_loss']):>8}"
            )
            print(f"\r{line}", end="", flush=True)
            if epoch >= args.epochs:
                print("\n\nTraining complete!")
                break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
