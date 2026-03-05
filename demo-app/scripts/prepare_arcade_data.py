#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


def natural_key(value: str) -> list[int | str]:
    stem = Path(value).stem
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", stem)]


def safe_id(name: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "patient"


def sequence_key_from_filename(file_name: str, prefix_parts: int) -> str:
    stem = Path(file_name).stem
    parts = stem.split("_")
    if len(parts) >= prefix_parts + 1:
        return "_".join(parts[:prefix_parts])
    return stem


@dataclass
class SplitLayout:
    split: str
    images_dir: Path
    annotations_path: Path


@dataclass
class FrameSample:
    split: str
    file_name: str
    width: int
    height: int
    annotations: list[dict]


def discover_split_layouts(source_root: Path) -> list[SplitLayout]:
    candidates = [source_root, source_root / "syntax", source_root / "arcade" / "syntax"]
    layouts: list[SplitLayout] = []

    for base in candidates:
        if not base.exists():
            continue
        found_any = False
        for split in ("train", "val", "test"):
            images_dir = base / split / "images"
            annotations_path = base / split / "annotations" / f"{split}.json"
            if images_dir.exists() and annotations_path.exists():
                layouts.append(SplitLayout(split=split, images_dir=images_dir, annotations_path=annotations_path))
                found_any = True
        if found_any:
            break

    return layouts


def build_groups(layouts: list[SplitLayout], prefix_parts: int) -> dict[str, list[FrameSample]]:
    grouped: dict[str, list[FrameSample]] = defaultdict(list)

    for layout in layouts:
        with layout.annotations_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)

        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            image_id = int(ann.get("image_id", -1))
            if image_id >= 0:
                anns_by_image[image_id].append(ann)

        for image in coco.get("images", []):
            image_id = int(image.get("id", -1))
            file_name = str(image.get("file_name", ""))
            if image_id < 0 or not file_name:
                continue
            key = sequence_key_from_filename(file_name, prefix_parts=prefix_parts)
            grouped[key].append(
                FrameSample(
                    split=layout.split,
                    file_name=file_name,
                    width=int(image.get("width", 0)),
                    height=int(image.get("height", 0)),
                    annotations=anns_by_image.get(image_id, []),
                )
            )

    for key, samples in grouped.items():
        grouped[key] = sorted(samples, key=lambda item: natural_key(item.file_name))

    return grouped


def render_binary_mask(width: int, height: int, annotations: list[dict]) -> Image.Image:
    mask = Image.new("L", (max(1, width), max(1, height)), color=0)
    draw = ImageDraw.Draw(mask)

    for ann in annotations:
        segmentation = ann.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                continue
            points = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)

    return mask


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"patients": []}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_image_path(layouts: list[SplitLayout], split: str, file_name: str) -> Path:
    for layout in layouts:
        if layout.split == split:
            return layout.images_dir / file_name
    raise FileNotFoundError(f"No split layout registered for split '{split}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARCADE patient data (frames + GT masks) for demo-app.")
    parser.add_argument("--source-root", type=Path, required=True, help="Path to ARCADE root or ARCADE/syntax root")
    parser.add_argument("--output-root", type=Path, default=Path("demo-app/data/patients"))
    parser.add_argument("--max-patients", type=int, default=10)
    parser.add_argument("--max-frames-per-patient", type=int, default=240)
    parser.add_argument("--min-frames-per-patient", type=int, default=20)
    parser.add_argument("--prefix-parts", type=int, default=3)
    parser.add_argument("--default-fps", type=int, default=12)
    parser.add_argument("--replace-existing-arcade", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-prediction-dirs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"

    layouts = discover_split_layouts(source_root)
    if not layouts:
        raise SystemExit(
            f"Could not locate ARCADE split layout under {source_root}. "
            "Expected train/val/test with images and annotations/*.json"
        )

    grouped = build_groups(layouts, prefix_parts=args.prefix_parts)
    eligible = [(key, samples) for key, samples in grouped.items() if len(samples) >= args.min_frames_per_patient]
    eligible.sort(key=lambda item: item[0])
    selected = eligible[: args.max_patients]

    print(f"Discovered {len(grouped)} sequences, {len(eligible)} eligible, selecting {len(selected)}")

    manifest = load_manifest(manifest_path)
    existing_patients = manifest.get("patients", []) if isinstance(manifest.get("patients", []), list) else []
    preserved_patients: list[dict] = []

    if args.replace_existing_arcade:
        for row in existing_patients:
            dataset_id = row.get("datasetId", "mendeley")
            if dataset_id != "arcade":
                preserved_patients.append(row)
    else:
        preserved_patients = existing_patients

    new_patients: list[dict] = []

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for sequence_key, samples in selected:
        patient_id = safe_id(f"arcade_{sequence_key}")
        frames = samples[: args.max_frames_per_patient]

        patient_root = output_root / patient_id
        frames_dir = patient_root / "frames"
        label_masks_dir = patient_root / "label_masks"
        prediction_dir = patient_root / "predictions" / "sam_vmnet_arcade"

        if not args.dry_run:
            if patient_root.exists() and args.replace_existing_arcade:
                shutil.rmtree(patient_root)
            frames_dir.mkdir(parents=True, exist_ok=True)
            label_masks_dir.mkdir(parents=True, exist_ok=True)
            if args.create_prediction_dirs:
                prediction_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for sample in frames:
            source_image = resolve_image_path(layouts, sample.split, sample.file_name)
            if not source_image.exists():
                continue

            target_image = frames_dir / source_image.name
            target_mask = label_masks_dir / f"{source_image.stem}.png"

            if not args.dry_run:
                shutil.copy2(source_image, target_image)
                mask = render_binary_mask(sample.width, sample.height, sample.annotations)
                mask.save(target_mask)

            copied += 1

        new_patient = {
            "id": patient_id,
            "displayName": f"ARCADE {sequence_key.replace('_', ' ').upper()}",
            "datasetId": "arcade",
            "labelType": "mask",
            "framesDir": f"{patient_id}/frames",
            "labelMasksDir": f"{patient_id}/label_masks",
            "predictionMasks": {
                "sam_vmnet_arcade": f"{patient_id}/predictions/sam_vmnet_arcade",
            },
            "defaultFps": args.default_fps,
        }
        new_patients.append(new_patient)
        print(f"Prepared {patient_id}: {copied} frame(s)")

    output_manifest = {"patients": preserved_patients + new_patients}

    if args.dry_run:
        print(json.dumps(output_manifest, indent=2))
        return

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
