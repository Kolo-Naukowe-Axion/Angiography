from __future__ import annotations


def test_get_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert {card["id"] for card in payload} == {"yolo26s", "yolo26n"}
    assert all(card["status"] == "ready" for card in payload)
    active_ids = [card["id"] for card in payload if card["active"]]
    assert active_ids == ["yolo26s"]


def test_select_model_switches_active_card(client):
    response = client.post("/api/models/select", json={"modelId": "yolo26n"})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    active_ids = [card["id"] for card in payload if card["active"]]
    assert active_ids == ["yolo26n"]

    health = client.get("/api/health")
    assert health.status_code == 200
    assert "YOLO26n/weights/best.pt" in health.json()["modelPath"].replace("\\", "/")


def test_get_patients(client):
    response = client.get("/api/patients")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == "patient_001"
    assert payload[0]["frameCount"] == 3


def test_infer_frame_cache_behavior(client):
    first = client.post("/api/infer/frame", json={"patientId": "patient_001", "frameIndex": 0})
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = client.post("/api/infer/frame", json={"patientId": "patient_001", "frameIndex": 0})
    assert second.status_code == 200
    assert second.json()["cached"] is True


def test_prefetch_updates_queue(client):
    response = client.post(
        "/api/infer/prefetch",
        json={"patientId": "patient_001", "startFrame": 1, "endFrame": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["patientId"] == "patient_001"
    assert payload["queued"] >= 0


def test_get_labels(client):
    response = client.get("/api/labels/patient_001/0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert len(payload["boxes"]) == 1


def test_get_labels_missing_label_file_returns_empty_frame_labels(client):
    response = client.get("/api/labels/patient_001/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is False
    assert payload["boxes"] == []
