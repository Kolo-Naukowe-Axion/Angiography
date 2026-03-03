from __future__ import annotations

from pathlib import Path

from .config import ROOT_DIR
from .schemas import ModelCard

MODEL_PATHS = {
    "yolo26s": (ROOT_DIR / "YOLO26s/weights/best.pt").resolve(),
    "yolo26n": (ROOT_DIR / "YOLO26n/weights/best.pt").resolve(),
}

MODEL_ORDER = ["yolo26s", "yolo26n"]

MODEL_NAMES = {
    "yolo26s": "YOLO26s (Mendeley)",
    "yolo26n": "YOLO26n (Mendeley)",
}

MODEL_NOTES = {
    "yolo26s": "Default live inference model for this demo.",
    "yolo26n": "Alternative ready-to-run demo model.",
}


def _active_model_id(model_path: Path | None) -> str:
    if model_path is None:
        return "yolo26s"

    normalized = str(model_path).lower().replace("\\", "/")
    if "/yolo26n/" in normalized:
        return "yolo26n"
    return "yolo26s"


def get_model_cards(
    model_path: Path | None = None,
    availability: dict[str, bool] | None = None,
    availability_reasons: dict[str, str] | None = None,
) -> list[ModelCard]:
    active_model_id = _active_model_id(model_path)
    cards: list[ModelCard] = []
    for model_id in MODEL_ORDER:
        is_ready = True if availability is None else availability.get(model_id, False)
        status = "ready" if is_ready else "unavailable"
        note = MODEL_NOTES[model_id]
        if not is_ready and availability_reasons and model_id in availability_reasons:
            note = f"{note} Unavailable: {availability_reasons[model_id]}"

        cards.append(
            ModelCard(
                id=model_id,
                name=MODEL_NAMES[model_id],
                active=(active_model_id == model_id and is_ready),
                status=status,
                notes=note,
            )
        )
    return cards


def get_model_path(model_id: str) -> Path | None:
    return MODEL_PATHS.get(model_id)
