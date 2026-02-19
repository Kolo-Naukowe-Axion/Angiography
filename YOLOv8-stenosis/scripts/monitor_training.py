"""
Monitor YOLOv8 training progress — one line per epoch with a progress bar.

Usage (in a second terminal on the GPU server):
    python scripts/monitor_training.py
    python scripts/monitor_training.py --runs-dir runs/stenosis/yolov8m_v1
"""

import argparse
import csv
import glob
import os
import time
from pathlib import Path


COLS = {
    "epoch": "epoch",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "box_loss": "train/box_loss",
    "cls_loss": "train/cls_loss",
}

BAR_WIDTH = 30


def find_latest_results(base):
    pattern = str(Path(base) / "**/results.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def bar(current, total, width=BAR_WIDTH):
    filled = int(width * current / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def read_last_row(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, 0
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}
    return last, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=".", help="Root dir to search for results.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    args = parser.parse_args()

    print(f"Searching for results.csv under {args.runs_dir} ...")
    csv_path = None

    while True:
        # (re)find the latest results.csv in case training just started
        found = find_latest_results(args.runs_dir)
        if found != csv_path:
            csv_path = found
            if csv_path:
                print(f"\nTracking: {csv_path}\n")
                header = (
                    f"{'Epoch':>6}  {'Progress':<{BAR_WIDTH+2}}  "
                    f"{'mAP50':>7}  {'mAP50-95':>8}  {'Prec':>6}  {'Recall':>6}  "
                    f"{'box_loss':>8}  {'cls_loss':>8}"
                )
                print(header)
                print("-" * len(header))

        if csv_path and os.path.exists(csv_path):
            row, n_done = read_last_row(csv_path)
            if row:
                epoch = int(float(row.get("epoch", 0)))
                total = args.epochs
                b = bar(epoch, total)

                def g(key):
                    v = row.get(key, "")
                    try:
                        return f"{float(v):.4f}"
                    except Exception:
                        return "  —  "

                line = (
                    f"{epoch:>6}/{total}  {b}  "
                    f"{g(COLS['mAP50']):>7}  {g(COLS['mAP50-95']):>8}  "
                    f"{g(COLS['precision']):>6}  {g(COLS['recall']):>6}  "
                    f"{g(COLS['box_loss']):>8}  {g(COLS['cls_loss']):>8}"
                )
                # overwrite same line
                print(f"\r{line}", end="", flush=True)

                if epoch >= total:
                    print("\n\nTraining complete!")
                    break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
