from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


LABELS = [
    ("epoch", "Epoch"),
    ("time", "Elapsed (s)"),
    ("train/box_loss", "Train box loss"),
    ("train/cls_loss", "Train cls loss"),
    ("train/dfl_loss", "Train DFL loss"),
    ("metrics/precision(B)", "Val precision"),
    ("metrics/recall(B)", "Val recall"),
    ("metrics/mAP50(B)", "Val mAP50"),
    ("metrics/mAP50-95(B)", "Val mAP50-95"),
    ("val/box_loss", "Val box loss"),
    ("val/cls_loss", "Val cls loss"),
    ("val/dfl_loss", "Val DFL loss"),
    ("mean_iou", "Val mIoU"),
    ("iou_epoch", "mIoU epoch"),
    ("lr/pg0", "LR pg0"),
    ("lr/pg1", "LR pg1"),
    ("lr/pg2", "LR pg2"),
]


def read_last_row(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {key.strip(): value.strip() for key, value in rows[-1].items()}


def read_last_iou_row(csv_path: Path) -> dict[str, str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return {key.strip(): value.strip() for key, value in rows[-1].items()}


def format_value(key: str, row: dict[str, str]) -> str:
    value = row.get(key, "")
    if key == "epoch":
        return value
    try:
        return f"{float(value):.5f}"
    except ValueError:
        return value or "—"


def render(row: dict[str, str]) -> str:
    lines = []
    for key, label in LABELS:
        lines.append(f"{label:<16} {format_value(key, row)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show labeled YOLO26m CADICA training metrics.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("models/yolo26m_cadica/runs/cadica_selected_seed42/results.csv"),
        help="Path to Ultralytics results.csv",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Polling interval in seconds when --follow is used.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep watching and print the latest labeled row whenever it changes.",
    )
    parser.add_argument(
        "--iou-csv",
        type=Path,
        default=Path("models/yolo26m_cadica/runs/cadica_selected_seed42/iou_metrics.csv"),
        help="Path to periodic IoU metrics CSV.",
    )
    args = parser.parse_args()

    previous_epoch: str | None = None

    while True:
        row = read_last_row(args.results)
        iou_row = read_last_iou_row(args.iou_csv)
        if row is None:
            print(f"No rows found yet in {args.results}")
        else:
            if iou_row is not None:
                row["mean_iou"] = iou_row.get("mean_iou", "—")
                row["iou_epoch"] = iou_row.get("epoch", "—")
            else:
                row["mean_iou"] = "—"
                row["iou_epoch"] = "—"
            epoch = row.get("epoch")
            if not args.follow or epoch != previous_epoch:
                if previous_epoch is not None:
                    print("\n" + "-" * 40)
                print(render(row))
                previous_epoch = epoch

        if not args.follow:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
