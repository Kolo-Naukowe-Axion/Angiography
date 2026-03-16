from __future__ import annotations

from pathlib import Path

from app.models_registry import get_model_cards


def test_get_model_cards_marks_yolo26x_active_when_model_path_points_to_yolo26x():
    cards = get_model_cards(Path("/tmp/models/yolo26x/runs/cadica_selected_seed42_4090/weights/best.pt"))
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26x_cadica"].active is True
    assert by_id["yolo26m_cadica"].active is False


def test_get_model_cards_marks_unavailable_models():
    cards = get_model_cards(
        Path("/tmp/models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt"),
        availability={"yolo26m_cadica": True, "yolo26x_cadica": False},
        availability_reasons={"yolo26x_cadica": "weights not found"},
    )
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26m_cadica"].status == "ready"
    assert by_id["yolo26m_cadica"].active is True
    assert by_id["yolo26x_cadica"].status == "unavailable"
    assert by_id["yolo26x_cadica"].active is False
    assert "Unavailable: weights not found" in by_id["yolo26x_cadica"].notes


def test_get_model_cards_exposes_cadica_bbox_metadata():
    cards = get_model_cards(Path("/tmp/models/yolo26m_cadica/runs/cadica_selected_seed42/weights/best.pt"))
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26m_cadica"].datasetId == "cadica"
    assert by_id["yolo26m_cadica"].outputType == "bbox"
    assert by_id["yolo26x_cadica"].datasetId == "cadica"
    assert by_id["yolo26x_cadica"].outputType == "bbox"
