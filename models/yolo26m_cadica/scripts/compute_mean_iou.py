from __future__ import annotations

import argparse
import json
from pathlib import Path

from iou_metrics import compute_mean_iou


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean IoU for a YOLO26m CADICA checkpoint.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = compute_mean_iou(
        weights=args.weights.resolve(),
        data_yaml=args.data.resolve(),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        match_iou=args.match_iou,
        max_det=args.max_det,
    )

    payload = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "mean_iou": metrics.mean_iou,
        "matched_pairs": metrics.matched_pairs,
        "gt_boxes": metrics.gt_boxes,
        "pred_boxes": metrics.pred_boxes,
        "images": metrics.images,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
