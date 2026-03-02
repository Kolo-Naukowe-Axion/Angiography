from __future__ import annotations


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
