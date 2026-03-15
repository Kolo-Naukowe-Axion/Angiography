from __future__ import annotations

import argparse
import re
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from cadica_selected_utils import (  # type: ignore
        IMAGE_EXTENSIONS,
        SPLITS,
        build_expected_split_index,
        read_summary,
    )
else:
    from .cadica_selected_utils import IMAGE_EXTENSIONS, SPLITS, build_expected_split_index, read_summary


PREPARED_STEM_RE = re.compile(r"^cadica_(p\d+)_(v\d+)_(\d+)$")


def parse_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number} expected 5 values, got {len(parts)}")
        class_id = int(parts[0])
        coords = tuple(float(value) for value in parts[1:])
        if class_id != 0:
            raise ValueError(f"{label_path}:{line_number} expected class 0, got {class_id}")
        for index, coord in enumerate(coords):
            if not 0.0 <= coord <= 1.0:
                raise ValueError(f"{label_path}:{line_number} coordinate {index} out of range: {coord}")
        boxes.append((class_id, *coords))
    return boxes


def collect_split_files(split_dir: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    return {
        path.stem: path
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the prepared CADICA YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    summary = read_summary(dataset_root / "summary.json")
    expected = build_expected_split_index(
        cadica_root=Path(summary["cadica_root"]),
        split_manifest=Path(summary["split_manifest"]),
    )

    all_errors: list[str] = []
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_files = collect_split_files(image_dir, IMAGE_EXTENSIONS)
        label_files = collect_split_files(label_dir, (".txt",))
        expected_stems = set(expected[split]["frame_stems"])

        missing_labels = sorted(set(image_files) - set(label_files))
        missing_images = sorted(set(label_files) - set(image_files))
        unexpected_stems = sorted((set(image_files) | set(label_files)) - expected_stems)
        missing_expected = sorted(expected_stems - (set(image_files) & set(label_files)))

        if missing_labels:
            all_errors.append(f"{split}: {len(missing_labels)} images are missing label files")
        if missing_images:
            all_errors.append(f"{split}: {len(missing_images)} labels are missing image files")
        if unexpected_stems:
            all_errors.append(f"{split}: found {len(unexpected_stems)} unexpected stems outside the manifest split")
        if missing_expected:
            all_errors.append(f"{split}: missing {len(missing_expected)} expected prepared stems")

        positive_count = 0
        negative_count = 0
        bbox_count = 0
        for stem in sorted(set(image_files) & set(label_files)):
            match = PREPARED_STEM_RE.match(stem)
            if not match:
                all_errors.append(f"{split}: stem does not match expected naming scheme: {stem}")
                continue
            video_key = f"{match.group(1)}_{match.group(2)}"
            if video_key not in expected[split]["selected_videos"]:
                all_errors.append(f"{split}: {stem} belongs to video {video_key}, which is not assigned to this split")

            try:
                boxes = parse_yolo_label_file(label_files[stem])
            except ValueError as error:
                all_errors.append(str(error))
                continue

            bbox_count += len(boxes)
            if boxes:
                positive_count += 1
            else:
                negative_count += 1

        actual_count = len(set(image_files) & set(label_files))
        if actual_count != expected[split]["image_count"]:
            all_errors.append(
                f"{split}: expected {expected[split]['image_count']} matched files, found {actual_count}"
            )
        if positive_count != expected[split]["positive_image_count"]:
            all_errors.append(
                f"{split}: expected {expected[split]['positive_image_count']} positive labels, found {positive_count}"
            )
        if negative_count != expected[split]["negative_image_count"]:
            all_errors.append(
                f"{split}: expected {expected[split]['negative_image_count']} negative labels, found {negative_count}"
            )
        if bbox_count != expected[split]["bbox_count"]:
            all_errors.append(
                f"{split}: expected {expected[split]['bbox_count']} boxes, found {bbox_count}"
            )

        print(
            f"{split:5s}: "
            f"{actual_count:4d} images, "
            f"{positive_count:4d} positive, "
            f"{negative_count:4d} negative, "
            f"{bbox_count:4d} boxes"
        )

    if all_errors:
        print("\nVerification FAILED:")
        for error in all_errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("\nDataset verification PASSED.")


if __name__ == "__main__":
    main()
