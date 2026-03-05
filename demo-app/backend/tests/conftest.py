from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.main import create_app


def _make_binary_mask(path: Path, draw_stenosis: bool = True) -> None:
    image = Image.new("L", (512, 512), color=0)
    if draw_stenosis:
        draw = ImageDraw.Draw(image)
        draw.ellipse((180, 180, 340, 340), fill=255)
    image.save(path)


@pytest.fixture()
def temp_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "patients"
    frames_dir_1 = data_dir / "patient_001" / "frames"
    labels_dir_1 = data_dir / "patient_001" / "labels"
    frames_dir_2 = data_dir / "patient_002" / "frames"

    arcade_frames_dir = data_dir / "arcade_patient_001" / "frames"
    arcade_label_masks_dir = data_dir / "arcade_patient_001" / "label_masks"
    arcade_predictions_dir = data_dir / "arcade_patient_001" / "predictions" / "sam_vmnet_arcade"

    frames_dir_1.mkdir(parents=True)
    labels_dir_1.mkdir(parents=True)
    frames_dir_2.mkdir(parents=True)
    arcade_frames_dir.mkdir(parents=True)
    arcade_label_masks_dir.mkdir(parents=True)
    arcade_predictions_dir.mkdir(parents=True)

    for index in range(3):
        img = Image.new("RGB", (512, 512), color=(20 + index * 5, 20, 20))
        img.save(frames_dir_1 / f"frame_{index:03d}.png")

    for index in range(2):
        img = Image.new("RGB", (512, 512), color=(20, 25 + index * 10, 20))
        img.save(frames_dir_2 / f"frame_{index:03d}.png")

    for index in range(2):
        img = Image.new("RGB", (512, 512), color=(25, 25, 40 + index * 20))
        img.save(arcade_frames_dir / f"frame_{index:03d}.png")

    (labels_dir_1 / "frame_000.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    _make_binary_mask(arcade_label_masks_dir / "frame_000.png", draw_stenosis=True)
    _make_binary_mask(arcade_label_masks_dir / "frame_001.png", draw_stenosis=False)
    _make_binary_mask(arcade_predictions_dir / "frame_000.png", draw_stenosis=True)
    _make_binary_mask(arcade_predictions_dir / "frame_001.png", draw_stenosis=True)

    manifest = {
        "patients": [
            {
                "id": "patient_001",
                "displayName": "Patient 001",
                "framesDir": "patient_001/frames",
                "labelsDir": "patient_001/labels",
                "defaultFps": 12,
                "datasetId": "mendeley",
                "labelType": "bbox",
            },
            {
                "id": "patient_002",
                "displayName": "Patient 002",
                "framesDir": "patient_002/frames",
                "defaultFps": 15,
                "datasetId": "mendeley",
                "labelType": "bbox",
            },
            {
                "id": "arcade_patient_001",
                "displayName": "ARCADE Patient 001",
                "framesDir": "arcade_patient_001/frames",
                "defaultFps": 12,
                "datasetId": "arcade",
                "labelType": "mask",
                "labelMasksDir": "arcade_patient_001/label_masks",
                "predictionMasks": {
                    "sam_vmnet_arcade": "arcade_patient_001/predictions/sam_vmnet_arcade"
                },
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
