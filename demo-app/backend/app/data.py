from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import PatientSummary


class ManifestValidationError(Exception):
    pass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


@dataclass
class PatientRecord:
    id: str
    display_name: str
    default_fps: int
    frames_dir: Path
    labels_dir: Path | None
    frames: list[Path]
    label_stems: set[str]

    @property
    def has_labels(self) -> bool:
        return bool(self.label_stems)


class PatientStore:
    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._patients = self._load()

    def _load(self) -> dict[str, PatientRecord]:
        manifest_path = self._data_dir / "manifest.json"
        if not manifest_path.exists():
            raise ManifestValidationError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        raw_patients = manifest.get("patients")
        if raw_patients is None or not isinstance(raw_patients, list):
            raise ManifestValidationError("Manifest must contain a 'patients' array.")

        seen_ids: set[str] = set()
        patients: dict[str, PatientRecord] = {}

        for row in raw_patients:
            patient_id = row.get("id")
            if not patient_id or not isinstance(patient_id, str):
                raise ManifestValidationError("Each patient entry needs a non-empty 'id'.")
            if patient_id in seen_ids:
                raise ManifestValidationError(f"Duplicate patient id: {patient_id}")
            seen_ids.add(patient_id)

            display_name = row.get("displayName", patient_id)
            default_fps = int(row.get("defaultFps", 12))
            if default_fps < 1:
                raise ManifestValidationError(f"defaultFps must be >=1 for patient {patient_id}")

            frames_dir = (self._data_dir / row.get("framesDir", f"{patient_id}/frames")).resolve()
            if not frames_dir.exists():
                raise ManifestValidationError(f"Frames dir missing for patient {patient_id}: {frames_dir}")

            labels_dir: Path | None = None
            if "labelsDir" in row and row["labelsDir"]:
                labels_dir = (self._data_dir / row["labelsDir"]).resolve()
                if not labels_dir.exists():
                    raise ManifestValidationError(f"labelsDir does not exist for patient {patient_id}: {labels_dir}")

            frames = sorted(
                [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()],
                key=_natural_key,
            )
            if not frames:
                raise ManifestValidationError(f"No image frames found in {frames_dir}")

            label_stems: set[str] = set()
            if labels_dir and labels_dir.exists():
                label_stems = {path.stem for path in labels_dir.glob("*.txt") if path.is_file()}

            patients[patient_id] = PatientRecord(
                id=patient_id,
                display_name=display_name,
                default_fps=default_fps,
                frames_dir=frames_dir,
                labels_dir=labels_dir,
                frames=frames,
                label_stems=label_stems,
            )

        return patients

    def summaries(self) -> list[PatientSummary]:
        return [
            PatientSummary(
                id=patient.id,
                displayName=patient.display_name,
                frameCount=len(patient.frames),
                hasLabels=patient.has_labels,
                defaultFps=patient.default_fps,
            )
            for patient in self._patients.values()
        ]

    def get_patient(self, patient_id: str) -> PatientRecord:
        if patient_id not in self._patients:
            raise KeyError(patient_id)
        return self._patients[patient_id]

    def get_frame_path(self, patient_id: str, frame_index: int) -> Path:
        patient = self.get_patient(patient_id)
        if frame_index < 0 or frame_index >= len(patient.frames):
            raise IndexError(frame_index)
        return patient.frames[frame_index]

    def get_label_path(self, patient_id: str, frame_index: int) -> Path | None:
        patient = self.get_patient(patient_id)
        if not patient.labels_dir:
            return None
        frame_path = self.get_frame_path(patient_id, frame_index)
        if frame_path.stem not in patient.label_stems:
            return None
        return patient.labels_dir / f"{frame_path.stem}.txt"

    def ensure_labels_dir(self, patient_id: str) -> Path:
        patient = self.get_patient(patient_id)
        if patient.labels_dir is not None:
            patient.labels_dir.mkdir(parents=True, exist_ok=True)
            return patient.labels_dir

        labels_dir = patient.frames_dir.parent / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        patient.labels_dir = labels_dir
        return labels_dir

    def get_writable_label_path(self, patient_id: str, frame_index: int) -> Path:
        frame_path = self.get_frame_path(patient_id, frame_index)
        labels_dir = self.ensure_labels_dir(patient_id)
        return labels_dir / f"{frame_path.stem}.txt"

    def mark_label_saved(self, patient_id: str, frame_index: int) -> None:
        patient = self.get_patient(patient_id)
        frame_path = self.get_frame_path(patient_id, frame_index)
        patient.label_stems.add(frame_path.stem)
