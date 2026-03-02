from __future__ import annotations

from .schemas import ModelCard


def get_model_cards() -> list[ModelCard]:
    return [
        ModelCard(
            id="yolo26s",
            name="YOLO26s (Mendeley)",
            active=True,
            status="ready",
            notes="Primary live inference model for this demo.",
        ),
        ModelCard(
            id="yolo26m",
            name="YOLO26m",
            active=False,
            status="coming_soon",
            notes="Reserved slot for future side-by-side comparison.",
        ),
        ModelCard(
            id="yolov8",
            name="YOLOv8-stenosis",
            active=False,
            status="coming_soon",
            notes="Planned additional detector profile.",
        ),
        ModelCard(
            id="sam_vmnet",
            name="SAM-VMNet",
            active=False,
            status="coming_soon",
            notes="Planned segmentation-assisted analysis track.",
        ),
    ]
