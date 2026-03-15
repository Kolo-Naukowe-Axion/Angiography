from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from iou_metrics import compute_mean_iou


def read_last_epoch(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return int(float(rows[-1]["epoch"]))


def read_completed_epochs(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return {int(row["epoch"]) for row in csv.DictReader(handle)}


def append_metrics(output_csv: Path, payload: dict[str, object]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "split",
                "mean_iou",
                "matched_pairs",
                "gt_boxes",
                "pred_boxes",
                "images",
                "weights",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean IoU every N epochs during YOLO26m CADICA training.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sleep", type=int, default=60)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    results_csv = run_dir / "results.csv"
    weights_path = run_dir / "weights" / "last.pt"
    output_csv = run_dir / "iou_metrics.csv"

    while True:
        last_epoch = read_last_epoch(results_csv)
        completed_epochs = read_completed_epochs(output_csv)

        if (
            last_epoch is not None
            and last_epoch > 0
            and last_epoch % args.every == 0
            and last_epoch not in completed_epochs
            and weights_path.exists()
        ):
            metrics = compute_mean_iou(
                weights=weights_path,
                data_yaml=args.data.resolve(),
                split=args.split,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                match_iou=args.match_iou,
                max_det=args.max_det,
            )
            payload = {
                "epoch": last_epoch,
                "split": args.split,
                "mean_iou": f"{metrics.mean_iou:.6f}",
                "matched_pairs": metrics.matched_pairs,
                "gt_boxes": metrics.gt_boxes,
                "pred_boxes": metrics.pred_boxes,
                "images": metrics.images,
                "weights": str(weights_path),
            }
            append_metrics(output_csv, payload)
            print(
                f"[periodic_iou_eval] epoch={last_epoch} split={args.split} "
                f"mean_iou={metrics.mean_iou:.4f} matched_pairs={metrics.matched_pairs}",
                flush=True,
            )

        if last_epoch is not None and last_epoch >= args.epochs:
            print("[periodic_iou_eval] Training epochs reached. Exiting.", flush=True)
            break

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
