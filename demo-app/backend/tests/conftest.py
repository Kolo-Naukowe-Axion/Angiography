from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "patients"
    frames_dir_1 = data_dir / "cadica_p7_v3" / "frames"
    labels_dir_1 = data_dir / "cadica_p7_v3" / "labels"
    frames_dir_2 = data_dir / "cadica_p7_v4" / "frames"
    labels_dir_2 = data_dir / "cadica_p7_v4" / "labels"

    frames_dir_1.mkdir(parents=True)
    labels_dir_1.mkdir(parents=True)
    frames_dir_2.mkdir(parents=True)
    labels_dir_2.mkdir(parents=True)

    for index in range(3):
        img = Image.new("RGB", (512, 512), color=(20 + index * 5, 20, 20))
        img.save(frames_dir_1 / f"frame_{index:03d}.png")

    for index in range(2):
        img = Image.new("RGB", (512, 512), color=(20, 25 + index * 10, 20))
        img.save(frames_dir_2 / f"frame_{index:03d}.png")

    (labels_dir_1 / "frame_000.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (labels_dir_1 / "frame_001.txt").write_text("", encoding="utf-8")

    manifest = {
        "patients": [
            {
                "id": "cadica_p7_v3",
                "displayName": "CADICA P7 V3",
                "framesDir": "cadica_p7_v3/frames",
                "labelsDir": "cadica_p7_v3/labels",
                "defaultFps": 12,
                "datasetId": "cadica",
                "labelType": "bbox",
            },
            {
                "id": "cadica_p7_v4",
                "displayName": "CADICA P7 V4",
                "framesDir": "cadica_p7_v4/frames",
                "labelsDir": "cadica_p7_v4/labels",
                "defaultFps": 15,
                "datasetId": "cadica",
                "labelType": "bbox",
            },
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir


@pytest.fixture()
def client(temp_data_dir: Path) -> TestClient:
    settings = Settings(
        data_dir=temp_data_dir,
        model_path=Path("/tmp/mock.pt"),
        cache_size=2,
        min_infer_confidence=0.1,
        frontend_origin="http://127.0.0.1:5173",
        use_mock_model=True,
        prefetch_queue_size=16,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
