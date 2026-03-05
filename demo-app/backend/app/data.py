from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .schemas import DatasetId, LabelType, PatientSummary


class ManifestValidationError(Exception):
    pass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def _collect_image_stems(directory: Path) -> dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir(), key=_natural_key)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


@dataclass
class PatientRecord:
    id: str
    display_name: str
    default_fps: int
    dataset_id: DatasetId
    label_type: LabelType
    frames_dir: Path
    labels_dir: Path | None
    label_masks_dir: Path | None
    prediction_masks_dirs: dict[str, Path]
    frames: list[Path]
    label_stems: set[str]
    label_mask_paths: dict[str, Path]
    prediction_mask_paths_by_model: dict[str, dict[str, Path]]

    @property
    def has_labels(self) -> bool:
        if self.label_type == "bbox":
            return bool(self.label_stems)
        return bool(self.label_mask_paths)


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

            dataset_id = row.get("datasetId", "mendeley")
            if dataset_id not in {"mendeley", "arcade"}:
                raise ManifestValidationError(f"Unsupported datasetId '{dataset_id}' for patient {patient_id}")

            label_type = row.get("labelType", "bbox")
            if label_type not in {"bbox", "mask"}:
                raise ManifestValidationError(f"Unsupported labelType '{label_type}' for patient {patient_id}")

            frames_dir = (self._data_dir / row.get("framesDir", f"{patient_id}/frames")).resolve()
            if not frames_dir.exists():
                raise ManifestValidationError(f"Frames dir missing for patient {patient_id}: {frames_dir}")

            labels_dir: Path | None = None
            label_stems: set[str] = set()
            if row.get("labelsDir"):
                labels_dir = (self._data_dir / row["labelsDir"]).resolve()
                if not labels_dir.exists():
                    raise ManifestValidationError(f"labelsDir does not exist for patient {patient_id}: {labels_dir}")
                label_stems = {path.stem for path in labels_dir.glob("*.txt") if path.is_file()}

            label_masks_dir: Path | None = None
            label_mask_paths: dict[str, Path] = {}
            if row.get("labelMasksDir"):
                label_masks_dir = (self._data_dir / row["labelMasksDir"]).resolve()
                if not label_masks_dir.exists():
                    raise ManifestValidationError(
                        f"labelMasksDir does not exist for patient {patient_id}: {label_masks_dir}"
                    )
                label_mask_paths = _collect_image_stems(label_masks_dir)

            prediction_masks_dirs: dict[str, Path] = {}
            prediction_mask_paths_by_model: dict[str, dict[str, Path]] = {}
            raw_prediction_masks = row.get("predictionMasks", {})
            if raw_prediction_masks and not isinstance(raw_prediction_masks, dict):
                raise ManifestValidationError(
                    f"predictionMasks must be an object for patient {patient_id}, got {type(raw_prediction_masks).__name__}"
                )
            for model_id, relative_dir in raw_prediction_masks.items():
                if not isinstance(model_id, str) or not isinstance(relative_dir, str):
                    raise ManifestValidationError(
                        f"predictionMasks entries must be string:string for patient {patient_id}"
                    )
                prediction_dir = (self._data_dir / relative_dir).resolve()
                prediction_masks_dirs[model_id] = prediction_dir
                prediction_mask_paths_by_model[model_id] = _collect_image_stems(prediction_dir)

            frames = sorted(
                [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()],
                key=_natural_key,
            )
            if not frames:
                raise ManifestValidationError(f"No image frames found in {frames_dir}")

            patients[patient_id] = PatientRecord(
                id=patient_id,
                display_name=display_name,
                default_fps=default_fps,
                dataset_id=dataset_id,
                label_type=label_type,
                frames_dir=frames_dir,
                labels_dir=labels_dir,
                label_masks_dir=label_masks_dir,
                prediction_masks_dirs=prediction_masks_dirs,
                frames=frames,
                label_stems=label_stems,
                label_mask_paths=label_mask_paths,
                prediction_mask_paths_by_model=prediction_mask_paths_by_model,
            )

        return patients

    def summaries(self, dataset_id: DatasetId | None = None) -> list[PatientSummary]:
        records = self._patients.values()
        if dataset_id is not None:
            records = [patient for patient in records if patient.dataset_id == dataset_id]

        return [
            PatientSummary(
                id=patient.id,
                displayName=patient.display_name,
                frameCount=len(patient.frames),
                hasLabels=patient.has_labels,
                defaultFps=patient.default_fps,
                datasetId=patient.dataset_id,
                labelType=patient.label_type,
            )
            for patient in records
        ]

    def get_patient(self, patient_id: str, dataset_id: DatasetId | None = None) -> PatientRecord:
        if patient_id not in self._patients:
            raise KeyError(patient_id)
        patient = self._patients[patient_id]
        if dataset_id is not None and patient.dataset_id != dataset_id:
            raise KeyError(patient_id)
        return patient

    def get_frame_path(self, patient_id: str, frame_index: int, dataset_id: DatasetId | None = None) -> Path:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        if frame_index < 0 or frame_index >= len(patient.frames):
            raise IndexError(frame_index)
        return patient.frames[frame_index]

    def get_label_path(self, patient_id: str, frame_index: int, dataset_id: DatasetId | None = None) -> Path | None:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        if patient.label_type != "bbox" or not patient.labels_dir:
            return None
        frame_path = self.get_frame_path(patient_id, frame_index, dataset_id=dataset_id)
        if frame_path.stem not in patient.label_stems:
            return None
        return patient.labels_dir / f"{frame_path.stem}.txt"

    def get_mask_path(
        self,
        patient_id: str,
        frame_index: int,
        source: str,
        model_id: str | None = None,
        dataset_id: DatasetId | None = None,
    ) -> Path | None:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        frame_path = self.get_frame_path(patient_id, frame_index, dataset_id=dataset_id)
        stem = frame_path.stem

        if source == "ground_truth":
            if patient.label_type != "mask":
                return None
            return patient.label_mask_paths.get(stem)

        if source == "prediction":
            if model_id is None:
                return None
            model_masks = patient.prediction_mask_paths_by_model.get(model_id, {})
            return model_masks.get(stem)

        return None

    def ensure_labels_dir(self, patient_id: str, dataset_id: DatasetId | None = None) -> Path:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        if patient.label_type != "bbox":
            raise ValueError("Mask-labeled patients are read-only in this demo.")

        if patient.labels_dir is not None:
            patient.labels_dir.mkdir(parents=True, exist_ok=True)
            return patient.labels_dir

        labels_dir = patient.frames_dir.parent / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        patient.labels_dir = labels_dir
        return labels_dir

    def get_writable_label_path(self, patient_id: str, frame_index: int, dataset_id: DatasetId | None = None) -> Path:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        if patient.label_type != "bbox":
            raise ValueError("Mask-labeled patients are read-only in this demo.")

        frame_path = self.get_frame_path(patient_id, frame_index, dataset_id=dataset_id)
        labels_dir = self.ensure_labels_dir(patient_id, dataset_id=dataset_id)
        return labels_dir / f"{frame_path.stem}.txt"

    def mark_label_saved(self, patient_id: str, frame_index: int, dataset_id: DatasetId | None = None) -> None:
        patient = self.get_patient(patient_id, dataset_id=dataset_id)
        if patient.label_type != "bbox":
            raise ValueError("Mask-labeled patients are read-only in this demo.")

        frame_path = self.get_frame_path(patient_id, frame_index, dataset_id=dataset_id)
        patient.label_stems.add(frame_path.stem)

    def is_model_prediction_ready(self, model_id: str, dataset_id: DatasetId) -> tuple[bool, str | None]:
        candidates = [patient for patient in self._patients.values() if patient.dataset_id == dataset_id]
        if not candidates:
            return False, f"no patients for dataset '{dataset_id}'"

        for patient in candidates:
            prediction_dir = patient.prediction_masks_dirs.get(model_id)
            if prediction_dir is None:
                return False, f"patient {patient.id} has no predictionMasks entry for {model_id}"
            if not prediction_dir.exists():
                return False, f"prediction dir missing for patient {patient.id}: {prediction_dir}"

            model_masks = patient.prediction_mask_paths_by_model.get(model_id, {})
            if not model_masks:
                return False, f"prediction dir has no mask files for patient {patient.id}: {prediction_dir}"

            missing_stems = [frame.stem for frame in patient.frames if frame.stem not in model_masks]
            if missing_stems:
                return False, (
                    f"missing {len(missing_stems)} prediction mask(s) for patient {patient.id}, "
                    f"example stem: {missing_stems[0]}"
                )

        return True, None
