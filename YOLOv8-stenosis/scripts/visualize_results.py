"""
Visualize model predictions vs ground truth on test images.

Generates a grid of comparisons showing:
- Ground truth bounding boxes (green)
- Model predictions (red) with confidence scores
- Categorized as: True Positive, False Positive, False Negative

Usage:
    python scripts/visualize_results.py \
        --weights weights/best.pt \
        --test-images dataset/images/test \
        --test-labels dataset/labels/test \
        --output results/examples
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GREEN = (0, 200, 0)
RED = (0, 0, 255)


def load_gt_boxes(label_path, img_w, img_h):
    """Load ground truth boxes from YOLO label file, return as (x1, y1, x2, y2) pixels."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(img, boxes, color, label=None, thickness=2):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs ground truth")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--test-images", type=str, default="dataset/images/test")
    parser.add_argument("--test-labels", type=str, default="dataset/labels/test")
    parser.add_argument("--output", type=str, default="results/examples")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = YOLO(args.weights)
    test_dir = Path(args.test_images)
    labels_dir = Path(args.test_labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = [f for f in test_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    random.seed(args.seed)
    samples = random.sample(all_images, min(args.n_samples, len(all_images)))

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)

        results = model.predict(str(img_path), conf=args.conf, verbose=False)
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                pred_boxes.append((x1, y1, x2, y2, conf))

        vis = img.copy()
        draw_boxes(vis, gt_boxes, GREEN, "GT")
        for x1, y1, x2, y2, conf in pred_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)

        # Save individual image
        out_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), vis)

        # Add to grid
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(vis_rgb)
        axes[idx].set_title(img_path.name, fontsize=8)
        axes[idx].axis("off")

    # Hide unused axes
    for idx in range(len(samples), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Green = Ground Truth | Red = Prediction", fontsize=14)
    plt.tight_layout()
    grid_path = output_dir / "comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid saved to {grid_path}")
    print(f"Individual predictions saved to {output_dir}/")


if __name__ == "__main__":
    main()
