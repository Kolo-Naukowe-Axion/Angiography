from __future__ import annotations

from pathlib import Path

from .config import ROOT_DIR
from .schemas import DatasetId, InferenceMode, ModelCard, ModelId, OutputType


MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_ID: ModelId = "yolo26m_cadica"

MODEL_PATHS: dict[ModelId, Path] = {
    "yolo26m_cadica": (MODELS_DIR / "yolo26m_cadica" / "runs" / "cadica_selected_seed42" / "weights" / "best.pt").resolve(),
    "yolo26x_cadica": (MODELS_DIR / "yolo26x" / "runs" / "cadica_selected_seed42_4090" / "weights" / "best.pt").resolve(),
}

MODEL_ORDER: list[ModelId] = ["yolo26m_cadica", "yolo26x_cadica"]

MODEL_NAMES: dict[ModelId, str] = {
    "yolo26m_cadica": "YOLO26m (CADICA)",
    "yolo26x_cadica": "YOLO26x (CADICA)",
}

MODEL_NOTES: dict[ModelId, str] = {
    "yolo26m_cadica": "Default CADICA checkpoint for the supported demo flow.",
    "yolo26x_cadica": "Larger CADICA checkpoint trained on the same leakage-free split.",
}

MODEL_DATASETS: dict[ModelId, DatasetId] = {
    "yolo26m_cadica": "cadica",
    "yolo26x_cadica": "cadica",
}

MODEL_OUTPUT_TYPES: dict[ModelId, OutputType] = {
    "yolo26m_cadica": "bbox",
    "yolo26x_cadica": "bbox",
}

MODEL_INFERENCE_MODES: dict[ModelId, InferenceMode] = {
    "yolo26m_cadica": "live",
    "yolo26x_cadica": "live",
}


def _active_model_id(model_path: Path | None) -> ModelId:
    if model_path is None:
        return DEFAULT_MODEL_ID

    normalized = str(model_path).lower().replace("\\", "/")
    if "/yolo26x/" in normalized:
        return "yolo26x_cadica"
    if "/yolo26m_cadica/" in normalized:
        return "yolo26m_cadica"
    return DEFAULT_MODEL_ID


def get_model_id_for_path(model_path: Path | None) -> ModelId:
    return _active_model_id(model_path)


def get_model_dataset_id(model_id: ModelId) -> DatasetId:
    return MODEL_DATASETS[model_id]


def get_model_output_type(model_id: ModelId) -> OutputType:
    return MODEL_OUTPUT_TYPES[model_id]


def get_model_inference_mode(model_id: ModelId) -> InferenceMode:
    return MODEL_INFERENCE_MODES[model_id]


def get_model_cards(
    model_path: Path | None = None,
    availability: dict[ModelId, bool] | None = None,
    availability_reasons: dict[ModelId, str] | None = None,
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
                datasetId=MODEL_DATASETS[model_id],
                outputType=MODEL_OUTPUT_TYPES[model_id],
                inferenceMode=MODEL_INFERENCE_MODES[model_id],
            )
        )
    return cards


def get_model_path(model_id: ModelId) -> Path:
    return MODEL_PATHS[model_id]


def get_model_ids() -> list[ModelId]:
    return list(MODEL_ORDER)
