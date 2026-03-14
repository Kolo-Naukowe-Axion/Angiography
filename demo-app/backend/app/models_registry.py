from __future__ import annotations

from pathlib import Path

from .config import ROOT_DIR
from .schemas import DatasetId, InferenceMode, ModelCard, OutputType

SAM_VMNET_MODEL_ID = "sam_vmnet_arcade"
MODELS_DIR = ROOT_DIR / "models"

MODEL_PATHS = {
    "yolo26s": (MODELS_DIR / "yolo26s" / "best.pt").resolve(),
    "yolo26n": (MODELS_DIR / "yolo26n" / "best.pt").resolve(),
    SAM_VMNET_MODEL_ID: (MODELS_DIR / "sam_vmnet" / "best-epoch142-loss0.3230.pth").resolve(),
}

MODEL_ORDER = ["yolo26s", "yolo26n", SAM_VMNET_MODEL_ID]

MODEL_NAMES = {
    "yolo26s": "YOLO26s (Mendeley)",
    "yolo26n": "YOLO26n (Mendeley)",
    SAM_VMNET_MODEL_ID: "SAM-VMNet (ARCADE)",
}

MODEL_NOTES = {
    "yolo26s": "Default live inference model for this demo.",
    "yolo26n": "Alternative ready-to-run demo model.",
    SAM_VMNET_MODEL_ID: "Precomputed mask inference on ARCADE sequences.",
}

MODEL_DATASETS: dict[str, DatasetId] = {
    "yolo26s": "mendeley",
    "yolo26n": "mendeley",
    SAM_VMNET_MODEL_ID: "arcade",
}

MODEL_OUTPUT_TYPES: dict[str, OutputType] = {
    "yolo26s": "bbox",
    "yolo26n": "bbox",
    SAM_VMNET_MODEL_ID: "mask",
}

MODEL_INFERENCE_MODES: dict[str, InferenceMode] = {
    "yolo26s": "live",
    "yolo26n": "live",
    SAM_VMNET_MODEL_ID: "precomputed",
}


def _active_model_id(model_path: Path | None) -> str:
    if model_path is None:
        return "yolo26s"

    normalized = str(model_path).lower().replace("\\", "/")
    if "/yolo26n/" in normalized:
        return "yolo26n"
    if "/sam_vmnet/" in normalized or "best-epoch142-loss0.3230" in normalized:
        return SAM_VMNET_MODEL_ID
    return "yolo26s"


def get_model_id_for_path(model_path: Path | None) -> str:
    return _active_model_id(model_path)


def get_model_dataset_id(model_id: str) -> DatasetId:
    return MODEL_DATASETS.get(model_id, "mendeley")


def get_model_output_type(model_id: str) -> OutputType:
    return MODEL_OUTPUT_TYPES.get(model_id, "bbox")


def get_model_inference_mode(model_id: str) -> InferenceMode:
    return MODEL_INFERENCE_MODES.get(model_id, "live")


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
                datasetId=MODEL_DATASETS[model_id],
                outputType=MODEL_OUTPUT_TYPES[model_id],
                inferenceMode=MODEL_INFERENCE_MODES[model_id],
            )
        )
    return cards


def get_model_path(model_id: str) -> Path | None:
    return MODEL_PATHS.get(model_id)


def get_model_ids() -> list[str]:
    return list(MODEL_ORDER)
