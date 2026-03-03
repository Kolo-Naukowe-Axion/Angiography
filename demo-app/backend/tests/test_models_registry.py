from __future__ import annotations

from pathlib import Path

from app.models_registry import get_model_cards


def test_get_model_cards_marks_yolo26n_active_when_model_path_points_to_yolo26n():
    cards = get_model_cards(Path("/tmp/YOLO26n/weights/best.pt"))
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26n"].active is True
    assert by_id["yolo26s"].active is False


def test_get_model_cards_marks_unavailable_models():
    cards = get_model_cards(
        Path("/tmp/YOLO26s/weights/best.pt"),
        availability={"yolo26s": True, "yolo26n": False},
        availability_reasons={"yolo26n": "weights not found"},
    )
    by_id = {card.id: card for card in cards}
    assert by_id["yolo26s"].status == "ready"
    assert by_id["yolo26s"].active is True
    assert by_id["yolo26n"].status == "unavailable"
    assert by_id["yolo26n"].active is False
    assert "Unavailable: weights not found" in by_id["yolo26n"].notes
