from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schemas import Box


def parse_yolo_labels_to_boxes(label_path: Path, frame_path: Path) -> list[Box]:
    if not label_path.exists():
        return []

    with Image.open(frame_path) as image:
        width, height = image.size

    boxes: list[Box] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
            cx = cx_n * width
            cy = cy_n * height
            box_w = w_n * width
            box_h = h_n * height
            x1 = cx - box_w / 2
            y1 = cy - box_h / 2
            x2 = cx + box_w / 2
            y2 = cy + box_h / 2

            boxes.append(
                Box(
                    x1=float(max(0.0, x1)),
                    y1=float(max(0.0, y1)),
                    x2=float(min(width, x2)),
                    y2=float(min(height, y2)),
                    confidence=1.0,
                    classId=class_id,
                    className="stenosis",
                )
            )

    return boxes
