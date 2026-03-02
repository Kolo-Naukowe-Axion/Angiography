from __future__ import annotations

from .schemas import Box


def has_stenosis(boxes: list[Box], threshold: float) -> bool:
    return any(box.confidence >= threshold for box in boxes)
