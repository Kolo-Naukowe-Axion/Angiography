from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from app.data import ManifestValidationError, PatientStore


def _make_frame(frame_dir: Path, name: str) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), color=(10, 10, 10)).save(frame_dir / name)


def _make_mask(mask_dir: Path, name: str) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.new("L", (256, 256), color=255).save(mask_dir / name)


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


def test_ensure_labels_dir_and_mark_label_saved_for_unlabeled_patient(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    frame_dir = data_dir / "p1" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color=(20, 20, 20)).save(frame_dir / "f1.png")

    manifest = {"patients": [{"id": "p1", "framesDir": "p1/frames", "defaultFps": 12}]}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = PatientStore(data_dir)
    labels_dir = store.ensure_labels_dir("p1")
    assert labels_dir.exists()
    assert labels_dir == data_dir / "p1" / "labels"

    writable_path = store.get_writable_label_path("p1", 0)
    assert writable_path == labels_dir / "f1.txt"

    store.mark_label_saved("p1", 0)
    summary = store.summaries()[0]
    assert summary.hasLabels is True


def test_legacy_manifest_defaults_dataset_and_label_type(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    _make_frame(data_dir / "p1" / "frames", "f1.png")
    manifest = {
        "patients": [
            {"id": "p1", "framesDir": "p1/frames", "defaultFps": 12},
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = PatientStore(data_dir)
    summary = store.summaries()[0]
    assert summary.datasetId == "mendeley"
    assert summary.labelType == "bbox"


def test_mask_patient_loads_ground_truth_and_prediction_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    _make_frame(data_dir / "arcade_01" / "frames", "f1.png")
    _make_mask(data_dir / "arcade_01" / "label_masks", "f1.png")
    _make_mask(data_dir / "arcade_01" / "pred" / "sam_vmnet_arcade", "f1.png")

    manifest = {
        "patients": [
            {
                "id": "arcade_01",
                "framesDir": "arcade_01/frames",
                "labelType": "mask",
                "datasetId": "arcade",
                "labelMasksDir": "arcade_01/label_masks",
                "predictionMasks": {"sam_vmnet_arcade": "arcade_01/pred/sam_vmnet_arcade"},
            }
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = PatientStore(data_dir)
    summary = store.summaries(dataset_id="arcade")[0]
    assert summary.labelType == "mask"
    assert summary.hasLabels is True

    gt_path = store.get_mask_path("arcade_01", 0, source="ground_truth", dataset_id="arcade")
    pred_path = store.get_mask_path(
        "arcade_01",
        0,
        source="prediction",
        model_id="sam_vmnet_arcade",
        dataset_id="arcade",
    )
    assert gt_path is not None and gt_path.exists()
    assert pred_path is not None and pred_path.exists()


def test_is_model_prediction_ready_reports_missing_masks(tmp_path: Path) -> None:
    data_dir = tmp_path / "patients"
    _make_frame(data_dir / "arcade_01" / "frames", "f1.png")
    _make_frame(data_dir / "arcade_01" / "frames", "f2.png")
    _make_mask(data_dir / "arcade_01" / "label_masks", "f1.png")
    _make_mask(data_dir / "arcade_01" / "label_masks", "f2.png")
    _make_mask(data_dir / "arcade_01" / "pred" / "sam_vmnet_arcade", "f1.png")

    manifest = {
        "patients": [
            {
                "id": "arcade_01",
                "framesDir": "arcade_01/frames",
                "labelType": "mask",
                "datasetId": "arcade",
                "labelMasksDir": "arcade_01/label_masks",
                "predictionMasks": {"sam_vmnet_arcade": "arcade_01/pred/sam_vmnet_arcade"},
            }
        ]
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = PatientStore(data_dir)
    is_ready, reason = store.is_model_prediction_ready("sam_vmnet_arcade", "arcade")
    assert is_ready is False
    assert reason is not None
    assert "missing" in reason
