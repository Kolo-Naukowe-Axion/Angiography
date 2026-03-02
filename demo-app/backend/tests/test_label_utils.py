from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.label_utils import parse_yolo_labels_to_boxes


def test_parse_yolo_labels_to_boxes(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (100, 200), color=(0, 0, 0)).save(frame_path)

    label_path = tmp_path / "frame.txt"
    label_path.write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")

    boxes = parse_yolo_labels_to_boxes(label_path, frame_path)

    assert len(boxes) == 1
    box = boxes[0]
    assert box.x1 == 30.0
    assert box.y1 == 50.0
    assert box.x2 == 70.0
    assert box.y2 == 150.0
