from __future__ import annotations

import json
import re
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CADICA_ROOT = REPO_ROOT / "datasets" / "cadica" / "CADICA"
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT / "models" / "yolo26m_cadica" / "manifests" / "patient_level_80_10_10_seed42.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "cadica" / "derived" / "yolo26_selected_seed42"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")
_NUMERIC_SUFFIX_RE = re.compile(r"^[a-zA-Z_]+(\d+)$")


@dataclass(frozen=True)
class FrameSample:
    split: str
    patient_id: str
    video_id: str
    frame_stem: str
    image_path: Path
    groundtruth_path: Path | None

    @property
    def prepared_stem(self) -> str:
        return f"cadica_{self.frame_stem}"

    @property
    def video_key(self) -> str:
        return f"{self.patient_id}_{self.video_id}"


def load_split_manifest(split_manifest: Path) -> dict:
    return json.loads(split_manifest.read_text(encoding="utf-8"))


def numeric_suffix_sort_key(value: str) -> tuple[int, str]:
    match = _NUMERIC_SUFFIX_RE.match(value)
    if match is None:
        return (0, value)
    return (int(match.group(1)), value)


def derive_selected_videos(cadica_root: Path, patient_ids: Iterable[str]) -> list[str]:
    selected_root = cadica_root / "selectedVideos"
    selected_videos: list[str] = []

    for patient_id in patient_ids:
        patient_dir = selected_root / patient_id
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"Missing CADICA patient directory: {patient_dir}")

        patient_video_dirs = sorted(
            (entry for entry in patient_dir.iterdir() if entry.is_dir()),
            key=lambda path: numeric_suffix_sort_key(path.name),
        )
        patient_selected_videos: list[str] = []
        for video_dir in patient_video_dirs:
            selected_frame_files = sorted(video_dir.glob("*_selectedFrames.txt"))
            if not selected_frame_files:
                continue
            if len(selected_frame_files) != 1:
                raise FileNotFoundError(f"Expected exactly one *_selectedFrames.txt in {video_dir}")
            patient_selected_videos.append(f"{patient_id}_{video_dir.name}")

        if not patient_selected_videos:
            raise FileNotFoundError(f"No selected CADICA videos found for patient {patient_id} in {patient_dir}")

        selected_videos.extend(patient_selected_videos)

    return selected_videos


def normalize_split_manifest(payload: dict, cadica_root: Path | None = None) -> dict:
    normalized = dict(payload)
    normalized["splits"] = {}

    for split in SPLITS:
        split_payload = payload.get("splits", {}).get(split)
        if split_payload is None:
            raise ValueError(f"Missing split '{split}' in manifest payload")

        patients = split_payload.get("patients")
        if not patients:
            raise ValueError(f"Split '{split}' is missing patients in manifest payload")

        normalized_split = dict(split_payload)
        if "selected_videos" not in normalized_split:
            if cadica_root is None:
                raise ValueError(
                    f"Split '{split}' is missing selected_videos and no cadica_root was provided "
                    "to derive them from the raw dataset."
                )
            normalized_split["selected_videos"] = derive_selected_videos(cadica_root, patients)

        normalized["splits"][split] = normalized_split

    return normalized


def load_selected_frame_stems(video_dir: Path) -> list[str]:
    candidates = sorted(video_dir.glob("*_selectedFrames.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one *_selectedFrames.txt in {video_dir}")
    return [line.strip() for line in candidates[0].read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_frame_image(input_dir: Path, frame_stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{frame_stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate image for frame {frame_stem} in {input_dir}")


def read_png_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Unsupported image format for size read: {image_path}")
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        if chunk_type != b"IHDR" or length != 13:
            raise ValueError(f"Invalid PNG header in {image_path}")
        width, height = struct.unpack(">II", handle.read(8))
    return width, height


def bbox_xywh_to_yolo(
    x_min: float,
    y_min: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if box_width <= 0 or box_height <= 0:
        raise ValueError("Bounding boxes must have positive size")

    x_center = (x_min + (box_width / 2.0)) / image_width
    y_center = (y_min + (box_height / 2.0)) / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height
    return x_center, y_center, norm_width, norm_height


def parse_cadica_groundtruth(label_path: Path, image_path: Path) -> list[str]:
    image_width, image_height = read_png_size(image_path)
    annotations: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 4:
            raise ValueError(f"{label_path}:{line_number} expected at least 4 values, got {len(parts)}")
        x_min, y_min, box_width, box_height = (float(value) for value in parts[:4])
        x_center, y_center, norm_width, norm_height = bbox_xywh_to_yolo(
            x_min=x_min,
            y_min=y_min,
            box_width=box_width,
            box_height=box_height,
            image_width=image_width,
            image_height=image_height,
        )
        annotations.append(
            f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        )
    return annotations


def iter_frame_samples(cadica_root: Path, split_manifest: Path) -> Iterable[FrameSample]:
    manifest = normalize_split_manifest(load_split_manifest(split_manifest), cadica_root=cadica_root)
    selected_root = cadica_root / "selectedVideos"

    for split in SPLITS:
        for video_key in manifest["splits"][split]["selected_videos"]:
            patient_id, video_id = video_key.split("_", 1)
            video_dir = selected_root / patient_id / video_id
            input_dir = video_dir / "input"
            groundtruth_dir = video_dir / "groundtruth"
            frame_stems = load_selected_frame_stems(video_dir)
            for frame_stem in frame_stems:
                image_path = resolve_frame_image(input_dir, frame_stem)
                label_path = groundtruth_dir / f"{frame_stem}.txt"
                yield FrameSample(
                    split=split,
                    patient_id=patient_id,
                    video_id=video_id,
                    frame_stem=frame_stem,
                    image_path=image_path,
                    groundtruth_path=label_path if label_path.exists() else None,
                )


def build_expected_split_index(cadica_root: Path, split_manifest: Path) -> dict[str, dict]:
    manifest = normalize_split_manifest(load_split_manifest(split_manifest), cadica_root=cadica_root)
    result: dict[str, dict] = {}

    for split in SPLITS:
        patients = sorted(manifest["splits"][split]["patients"])
        selected_videos = sorted(manifest["splits"][split]["selected_videos"])
        frame_stems: set[str] = set()
        positive_count = 0
        bbox_count = 0

        for sample in iter_frame_samples(cadica_root, split_manifest):
            if sample.split != split:
                continue
            frame_stems.add(sample.prepared_stem)
            if sample.groundtruth_path is not None:
                positive_count += 1
                bbox_count += count_nonempty_lines(sample.groundtruth_path)

        image_count = len(frame_stems)
        result[split] = {
            "patients": patients,
            "selected_videos": selected_videos,
            "frame_stems": sorted(frame_stems),
            "image_count": image_count,
            "positive_image_count": positive_count,
            "negative_image_count": image_count - positive_count,
            "bbox_count": bbox_count,
        }

    return result


def count_nonempty_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def ensure_clean_output_root(output_root: Path, force: bool) -> None:
    if output_root.exists():
        entries = [entry for entry in output_root.iterdir()]
        if entries and not force:
            raise FileExistsError(
                f"Output root {output_root} already exists and is not empty. "
                "Remove it first or rerun with --force."
            )
        if entries and force:
            shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def materialize_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
        return
    relative_target = Path(shutil.os.path.relpath(source, start=destination.parent))
    destination.symlink_to(relative_target)


def write_data_yaml(output_root: Path) -> None:
    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: stenosis",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_dataset(
    cadica_root: Path,
    split_manifest: Path,
    output_root: Path,
    copy_images: bool = False,
    force: bool = False,
) -> dict:
    cadica_root = cadica_root.resolve()
    split_manifest = split_manifest.resolve()
    output_root = output_root.resolve()

    ensure_clean_output_root(output_root, force=force)
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_name": output_root.name,
        "dataset_root": str(output_root),
        "cadica_root": str(cadica_root),
        "split_manifest": str(split_manifest),
        "materialization": "copy" if copy_images else "symlink",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {
            split: {
                "patients": [],
                "selected_videos": [],
                "image_count": 0,
                "positive_image_count": 0,
                "negative_image_count": 0,
                "bbox_count": 0,
            }
            for split in SPLITS
        },
    }

    expected = build_expected_split_index(cadica_root, split_manifest)
    for split in SPLITS:
        summary["splits"][split]["patients"] = expected[split]["patients"]
        summary["splits"][split]["selected_videos"] = expected[split]["selected_videos"]

    for sample in iter_frame_samples(cadica_root, split_manifest):
        image_target = output_root / "images" / sample.split / f"{sample.prepared_stem}{sample.image_path.suffix.lower()}"
        label_target = output_root / "labels" / sample.split / f"{sample.prepared_stem}.txt"

        materialize_image(sample.image_path, image_target, copy_images=copy_images)

        if sample.groundtruth_path is not None:
            annotations = parse_cadica_groundtruth(sample.groundtruth_path, sample.image_path)
        else:
            annotations = []

        label_target.write_text(
            ("\n".join(annotations) + "\n") if annotations else "",
            encoding="utf-8",
        )

        split_summary = summary["splits"][sample.split]
        split_summary["image_count"] += 1
        split_summary["bbox_count"] += len(annotations)
        if annotations:
            split_summary["positive_image_count"] += 1
        else:
            split_summary["negative_image_count"] += 1

    write_data_yaml(output_root)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))
