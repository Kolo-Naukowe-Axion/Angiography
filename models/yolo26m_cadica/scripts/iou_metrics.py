from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class IoUMetrics:
    mean_iou: float
    matched_pairs: int
    gt_boxes: int
    pred_boxes: int
    images: int


def resolve_split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    images_entry = Path(payload[split])
    images_dir = images_entry if images_entry.is_absolute() else (root / images_entry).resolve()
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/")).resolve()
    if "/images/" not in str(images_dir):
        raise ValueError(f"Could not infer labels directory from images path: {images_dir}")
    return images_dir, labels_dir


def iter_images(images_dir: Path) -> Iterable[Path]:
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def yolo_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1 = (x_center - (width / 2.0)) * image_width
    y1 = (y_center - (height / 2.0)) * image_height
    x2 = (x_center + (width / 2.0)) * image_width
    y2 = (y_center + (height / 2.0)) * image_height
    return x1, y1, x2, y2


def read_ground_truth_boxes(image_path: Path, label_path: Path) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []

    with Image.open(image_path) as image:
        width, height = image.size

    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 values, got {len(parts)}")
        class_id = int(parts[0])
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_number} expected class 0, got {class_id}")
        boxes.append(
            yolo_to_xyxy(
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                width,
                height,
            )
        )
    return boxes


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match_ious(
    predictions: list[tuple[float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    min_iou: float,
) -> list[float]:
    remaining_preds = predictions[:]
    remaining_gt = ground_truth[:]
    matches: list[float] = []

    while remaining_preds and remaining_gt:
        best_pred_index = -1
        best_gt_index = -1
        best_iou = 0.0

        for pred_index, pred_box in enumerate(remaining_preds):
            for gt_index, gt_box in enumerate(remaining_gt):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_index = pred_index
                    best_gt_index = gt_index

        if best_iou < min_iou:
            break

        matches.append(best_iou)
        remaining_preds.pop(best_pred_index)
        remaining_gt.pop(best_gt_index)

    return matches


def compute_mean_iou(
    *,
    weights: Path,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    conf: float,
    match_iou: float,
    max_det: int,
) -> IoUMetrics:
    from ultralytics import YOLO

    images_dir, labels_dir = resolve_split_dirs(data_yaml, split)
    image_paths = list(iter_images(images_dir))
    model = YOLO(str(weights))

    total_iou = 0.0
    matched_pairs = 0
    total_gt_boxes = 0
    total_pred_boxes = 0

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes = read_ground_truth_boxes(image_path, label_path)
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            stream=False,
            max_det=max_det,
        )
        result = results[0]
        pred_boxes = [tuple(box) for box in result.boxes.xyxy.cpu().tolist()]
        matched_ious = greedy_match_ious(pred_boxes, gt_boxes, min_iou=match_iou)

        total_iou += sum(matched_ious)
        matched_pairs += len(matched_ious)
        total_gt_boxes += len(gt_boxes)
        total_pred_boxes += len(pred_boxes)

    return IoUMetrics(
        mean_iou=(total_iou / matched_pairs) if matched_pairs else 0.0,
        matched_pairs=matched_pairs,
        gt_boxes=total_gt_boxes,
        pred_boxes=total_pred_boxes,
        images=len(image_paths),
    )
