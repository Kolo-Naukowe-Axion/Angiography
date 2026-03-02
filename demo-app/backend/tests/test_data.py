from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.data import ManifestValidationError, PatientStore


def _make_frame(frame_dir: Path, name: str) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=(10, 10, 10)).save(frame_dir / name)


def test_manifest_duplicate_patient_ids(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    _make_frame(data_dir / "p1" / "frames", "f1.png")
    manifest = {
        "patients": [
            {"id": "p1", "framesDir": "p1/frames"},
            {"id": "p1", "framesDir": "p1/frames"},
        ]
    }
    (data_dir / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="Duplicate patient id"):
        PatientStore(data_dir)


def test_manifest_missing_frames_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    data_dir.mkdir(parents=True)
    manifest = {"patients": [{"id": "p1", "framesDir": "p1/frames"}]}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="Frames dir missing"):
        PatientStore(data_dir)


def test_bmp_frames_supported_and_empty_labels_not_marked(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    frame_dir = data_dir / "p1" / "frames"
    label_dir = data_dir / "p1" / "labels"
    frame_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(20, 20, 20)).save(frame_dir / "f1.bmp")

    manifest = {
        "patients": [
            {"id": "p1", "framesDir": "p1/frames", "labelsDir": "p1/labels", "defaultFps": 12}
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = PatientStore(data_dir)
    summary = store.summaries()[0]
    assert summary.frameCount == 1
    assert summary.hasLabels is False
