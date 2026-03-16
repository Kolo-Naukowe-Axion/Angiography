from __future__ import annotations

from pathlib import Path


def test_get_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert {card["id"] for card in payload} == {"yolo26m_cadica", "yolo26x_cadica"}
    active_ids = [card["id"] for card in payload if card["active"]]
    assert active_ids == ["yolo26m_cadica"]

    by_id = {card["id"]: card for card in payload}
    assert by_id["yolo26m_cadica"]["datasetId"] == "cadica"
    assert by_id["yolo26m_cadica"]["outputType"] == "bbox"
    assert by_id["yolo26m_cadica"]["inferenceMode"] == "live"


def test_select_model_switches_active_card(client):
    response = client.post("/api/models/select", json={"modelId": "yolo26x_cadica"})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    active_ids = [card["id"] for card in payload if card["active"]]
    assert active_ids == ["yolo26x_cadica"]

    health = client.get("/api/health")
    assert health.status_code == 200
    assert "models/yolo26x/runs/cadica_selected_seed42_4090/weights/best.pt" in health.json()["modelPath"].replace("\\", "/")


def test_get_patients_returns_only_cadica_sequences(client):
    response = client.get("/api/patients")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    by_id = {patient["id"]: patient for patient in payload}
    assert by_id["cadica_p7_v3"]["frameCount"] == 3
    assert by_id["cadica_p7_v3"]["hasLabels"] is True
    assert by_id["cadica_p7_v3"]["datasetId"] == "cadica"
    assert by_id["cadica_p7_v4"]["frameCount"] == 2
    assert by_id["cadica_p7_v4"]["hasLabels"] is False


def test_infer_frame_cache_behavior(client):
    first = client.post("/api/infer/frame", json={"patientId": "cadica_p7_v3", "frameIndex": 0})
    assert first.status_code == 200
    assert first.json()["cached"] is False
    assert first.json()["outputType"] == "bbox"

    second = client.post("/api/infer/frame", json={"patientId": "cadica_p7_v3", "frameIndex": 0})
    assert second.status_code == 200
    assert second.json()["cached"] is True


def test_prefetch_updates_queue(client):
    response = client.post(
        "/api/infer/prefetch",
        json={"patientId": "cadica_p7_v3", "startFrame": 1, "endFrame": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["patientId"] == "cadica_p7_v3"
    assert payload["queued"] >= 0


def test_get_labels_for_positive_frame(client):
    response = client.get("/api/labels/cadica_p7_v3/0")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert payload["labelType"] == "bbox"
    assert len(payload["boxes"]) == 1


def test_get_labels_for_negative_frame_with_empty_file_returns_empty_boxes_but_labeled(client):
    response = client.get("/api/labels/cadica_p7_v3/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert payload["labelType"] == "bbox"
    assert payload["boxes"] == []


def test_get_labels_missing_label_file_returns_empty_frame_labels(client):
    response = client.get("/api/labels/cadica_p7_v4/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is False
    assert payload["labelType"] == "bbox"
    assert payload["boxes"] == []


def test_mask_assets_endpoint_is_unavailable_in_cadica_demo(client):
    response = client.get("/api/patients/cadica_p7_v3/frames/0/masks/prediction")
    assert response.status_code == 404


def test_put_labels_saves_and_returns_boxes(client):
    response = client.put(
        "/api/labels/cadica_p7_v3/2",
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
    assert payload["labelType"] == "bbox"
    assert len(payload["boxes"]) == 1
    assert payload["boxes"][0]["classId"] == 0

    follow_up = client.get("/api/labels/cadica_p7_v3/2")
    assert follow_up.status_code == 200
    assert follow_up.json()["hasLabels"] is True
    assert len(follow_up.json()["boxes"]) == 1


def test_put_labels_creates_labels_dir_for_unlabeled_patient(client):
    response = client.put(
        "/api/labels/cadica_p7_v4/0",
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
    assert by_id["cadica_p7_v4"]["hasLabels"] is True


def test_put_labels_empty_box_list_writes_empty_file_and_marks_labeled(client, temp_data_dir: Path):
    response = client.put("/api/labels/cadica_p7_v4/1", json={"boxes": []})
    assert response.status_code == 200
    payload = response.json()
    assert payload["hasLabels"] is True
    assert payload["boxes"] == []

    label_path = temp_data_dir / "cadica_p7_v4" / "labels" / "frame_001.txt"
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8") == ""

    labels = client.get("/api/labels/cadica_p7_v4/1")
    assert labels.status_code == 200
    assert labels.json()["hasLabels"] is True
    assert labels.json()["boxes"] == []


def test_put_labels_overwrites_existing_file(client, temp_data_dir: Path):
    response = client.put(
        "/api/labels/cadica_p7_v3/0",
        json={"boxes": [{"x1": 0, "y1": 0, "x2": 128, "y2": 128}]},
    )
    assert response.status_code == 200

    label_path = temp_data_dir / "cadica_p7_v3" / "labels" / "frame_000.txt"
    content = label_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1


def test_put_labels_rejects_invalid_box(client):
    response = client.put(
        "/api/labels/cadica_p7_v3/0",
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
        "/api/labels/cadica_p7_v3/99",
        json={"boxes": [{"x1": 1, "y1": 1, "x2": 20, "y2": 20}]},
    )
    assert response.status_code == 404
