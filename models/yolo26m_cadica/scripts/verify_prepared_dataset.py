from __future__ import annotations

import argparse
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def read_summary(dataset_root: Path) -> dict[str, object]:
    summary_path = dataset_root / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def collect_stems(directory: Path, suffixes: set[str]) -> set[str]:
    stems: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            stems.add(path.stem)
    return stems


def verify_split(dataset_root: Path, split: str, expected_images: int) -> None:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.is_dir():
        raise SystemExit(f"Missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"Missing label directory: {label_dir}")

    image_stems = collect_stems(image_dir, IMAGE_SUFFIXES)
    label_stems = collect_stems(label_dir, {".txt"})

    if len(image_stems) != expected_images:
        raise SystemExit(
            f"Split '{split}' has {len(image_stems)} images, expected {expected_images} from summary.json"
        )
    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)[:10]
        missing_images = sorted(label_stems - image_stems)[:10]
        raise SystemExit(
            f"Image/label mismatch in split '{split}'. "
            f"Missing labels for: {missing_labels}. Missing images for: {missing_images}."
        )


def verify_prepared_dataset(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    summary = read_summary(dataset_root)

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Missing data.yaml: {data_yaml}")

    expected_root_line = f"path: {dataset_root}"
    data_yaml_text = data_yaml.read_text(encoding="utf-8")
    if expected_root_line not in data_yaml_text:
        raise SystemExit(f"data.yaml does not point to dataset root '{dataset_root}'")

    splits = summary.get("splits")
    if not isinstance(splits, dict):
        raise SystemExit("summary.json is missing the 'splits' object")

    for split in SPLITS:
        split_info = splits.get(split)
        if not isinstance(split_info, dict):
            raise SystemExit(f"summary.json is missing split info for '{split}'")
        expected_images = split_info.get("image_count")
        if not isinstance(expected_images, int):
            raise SystemExit(f"summary.json split '{split}' is missing integer image_count")
        verify_split(dataset_root, split, expected_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a prepared CADICA YOLO dataset without raw-source dependencies.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    verify_prepared_dataset(args.dataset_root)
    print(f"Prepared dataset verified: {args.dataset_root.resolve()}")


if __name__ == "__main__":
    main()
