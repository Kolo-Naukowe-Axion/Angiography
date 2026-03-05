from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schemas import Box, GroundTruthBoxInput


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def _clamp_box_to_frame(box: GroundTruthBoxInput, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = _clamp(box.x1, 0.0, float(width))
    y1 = _clamp(box.y1, 0.0, float(height))
    x2 = _clamp(box.x2, 0.0, float(width))
    y2 = _clamp(box.y2, 0.0, float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box collapses outside frame after clamping.")
    return x1, y1, x2, y2


def _to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int, class_id: int = 0) -> str:
    cx_n = ((x1 + x2) / 2.0) / width
    cy_n = ((y1 + y2) / 2.0) / height
    w_n = (x2 - x1) / width
    h_n = (y2 - y1) / height
    return f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


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


def write_yolo_labels_from_boxes(label_path: Path, frame_path: Path, boxes: list[GroundTruthBoxInput]) -> None:
    with Image.open(frame_path) as image:
        width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Frame has invalid dimensions.")

    yolo_lines: list[str] = []
    for box in boxes:
        x1, y1, x2, y2 = _clamp_box_to_frame(box, width, height)
        yolo_lines.append(_to_yolo_line(x1, y1, x2, y2, width, height, class_id=0))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    text = ""
    if yolo_lines:
        text = "\n".join(yolo_lines) + "\n"
    label_path.write_text(text, encoding="utf-8")
