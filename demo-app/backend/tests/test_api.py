from __future__ import annotations

from pathlib import Path


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
    assert len(payload) == 2
    by_id = {patient["id"]: patient for patient in payload}
    assert by_id["patient_001"]["frameCount"] == 3
    assert by_id["patient_001"]["hasLabels"] is True
    assert by_id["patient_002"]["frameCount"] == 2
    assert by_id["patient_002"]["hasLabels"] is False


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


def test_put_labels_saves_and_returns_boxes(client):
    response = client.put(
        "/api/labels/patient_001/1",
        json={
            "boxes": [
                {
                    "x1": 40,
                    "y1": 80,
                    "x2": 280,
                    "y2": 320,
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert len(payload["boxes"]) == 1
    assert payload["boxes"][0]["classId"] == 0

    follow_up = client.get("/api/labels/patient_001/1")
    assert follow_up.status_code == 200
    assert follow_up.json()["hasLabels"] is True
    assert len(follow_up.json()["boxes"]) == 1


def test_put_labels_creates_labels_dir_for_unlabeled_patient(client):
    response = client.put(
        "/api/labels/patient_002/0",
        json={
            "boxes": [
                {
                    "x1": 20,
                    "y1": 40,
                    "x2": 200,
                    "y2": 260,
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert len(payload["boxes"]) == 1

    patients = client.get("/api/patients")
    assert patients.status_code == 200
    by_id = {patient["id"]: patient for patient in patients.json()}
    assert by_id["patient_002"]["hasLabels"] is True


def test_put_labels_empty_box_list_writes_empty_file_and_marks_labeled(client, temp_data_dir: Path):
    response = client.put("/api/labels/patient_002/1", json={"boxes": []})
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert payload["boxes"] == []

    label_path = temp_data_dir / "patient_002" / "labels" / "frame_001.txt"
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8") == ""

    labels = client.get("/api/labels/patient_002/1")
    assert labels.status_code == 200
    assert labels.json()["hasLabels"] is True
    assert labels.json()["boxes"] == []


def test_put_labels_overwrites_existing_file(client, temp_data_dir: Path):
    response = client.put(
        "/api/labels/patient_001/0",
        json={"boxes": [{"x1": 0, "y1": 0, "x2": 128, "y2": 128}]},
    )
    assert response.status_code == 200

    label_path = temp_data_dir / "patient_001" / "labels" / "frame_000.txt"
    content = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1


def test_put_labels_rejects_invalid_box(client):
    response = client.put(
        "/api/labels/patient_001/0",
        json={"boxes": [{"x1": 100, "y1": 10, "x2": 40, "y2": 20}]},
    )
    assert response.status_code == 400
    assert "x2>x1" in response.text


def test_put_labels_unknown_patient(client):
    response = client.put(
        "/api/labels/unknown/0",
        json={"boxes": [{"x1": 1, "y1": 1, "x2": 20, "y2": 20}]},
    )
    assert response.status_code == 404


def test_put_labels_frame_index_out_of_range(client):
    response = client.put(
        "/api/labels/patient_001/99",
        json={"boxes": [{"x1": 1, "y1": 1, "x2": 20, "y2": 20}]},
    )
    assert response.status_code == 404
