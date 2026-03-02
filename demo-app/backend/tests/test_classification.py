from __future__ import annotations

from app.classification import has_stenosis
from app.schemas import Box


def test_threshold_classification_behavior() -> None:
    boxes = [
        Box(x1=0, y1=0, x2=1, y2=1, confidence=0.4, classId=0, className="stenosis"),
        Box(x1=0, y1=0, x2=1, y2=1, confidence=0.7, classId=0, className="stenosis"),
    ]

    assert has_stenosis(boxes, threshold=0.5)
    assert not has_stenosis(boxes, threshold=0.9)
