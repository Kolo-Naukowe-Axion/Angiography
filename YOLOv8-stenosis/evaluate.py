"""
Evaluate trained YOLOv8-M on the test split and save metrics to JSON.

Usage:
    python evaluate.py
    python evaluate.py --weights weights/best.pt
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate stenosis detection model")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
    )
    parser.add_argument("--data", type=str, default="dataset/data.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="results/metrics.json")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, device=args.device)

    results = {
        "model": args.weights,
        "split": args.split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Test Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
