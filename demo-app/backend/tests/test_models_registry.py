from __future__ import annotations

from pathlib import Path

from app.models_registry import SAM_VMNET_MODEL_ID, get_model_cards


def test_get_model_cards_marks_yolo26n_active_when_model_path_points_to_yolo26n():
    cards = get_model_cards(Path("/tmp/models/yolo26n/weights/best.pt"))
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26n"].active is True
    assert by_id["yolo26s"].active is False


def test_get_model_cards_marks_unavailable_models():
    cards = get_model_cards(
        Path("/tmp/models/yolo26s/weights/best.pt"),
        availability={"yolo26s": True, "yolo26n": False, SAM_VMNET_MODEL_ID: False},
        availability_reasons={"yolo26n": "weights not found", SAM_VMNET_MODEL_ID: "missing predictions"},
    )
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26s"].status == "ready"
    assert by_id["yolo26s"].active is True
    assert by_id["yolo26n"].status == "unavailable"
    assert by_id["yolo26n"].active is False
    assert "Unavailable: weights not found" in by_id["yolo26n"].notes
    assert by_id[SAM_VMNET_MODEL_ID].status == "unavailable"
    assert by_id[SAM_VMNET_MODEL_ID].datasetId == "arcade"
    assert by_id[SAM_VMNET_MODEL_ID].outputType == "mask"
    assert by_id[SAM_VMNET_MODEL_ID].inferenceMode == "precomputed"


def test_get_model_cards_exposes_dataset_and_output_metadata():
    cards = get_model_cards(Path("/tmp/models/sam_vmnet/pre_trained_weights/best-epoch142-loss0.3230.pth"))
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26s"].datasetId == "mendeley"
    assert by_id["yolo26s"].outputType == "bbox"
    assert by_id[SAM_VMNET_MODEL_ID].datasetId == "arcade"
    assert by_id[SAM_VMNET_MODEL_ID].outputType == "mask"
