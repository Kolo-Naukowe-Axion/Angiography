#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def collect_patient_dirs(source_root: Path) -> list[Path]:
    candidates: list[Path] = []
    scan_dirs = [source_root] + sorted([p for p in source_root.rglob("*") if p.is_dir()])
    for directory in scan_dirs:
        has_images = any(child.suffix.lower() in IMAGE_EXTENSIONS for child in directory.iterdir() if child.is_file())
        if has_images:
            candidates.append(directory)
    return candidates


def safe_id(name: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "patient"


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def build_sequences(
    patient_dir: Path,
    image_paths: list[Path],
    split_flat_by_prefix: bool,
    prefix_parts: int,
    min_frames_per_patient: int,
) -> list[tuple[str, list[Path]]]:
    if not split_flat_by_prefix:
        return [(patient_dir.name, image_paths)]

    grouped: dict[str, list[Path]] = {}
    for image_path in image_paths:
        parts = image_path.stem.split("_")
        if len(parts) >= prefix_parts + 1:
            key = "_".join(parts[:prefix_parts])
        else:
            key = image_path.stem
        grouped.setdefault(key, []).append(image_path)

    grouped_items: list[tuple[str, list[Path]]] = []
    for key, group_paths in grouped.items():
        sorted_group = sorted(group_paths, key=natural_key)
        if len(sorted_group) >= min_frames_per_patient:
            grouped_items.append((key, sorted_group))

    if len(grouped_items) >= 2:
        return sorted(grouped_items, key=lambda item: item[0])

    return [(patient_dir.name, image_paths)]


def voc_xml_to_yolo_lines(xml_path: Path) -> list[str]:
    if not xml_path.exists():
        return []

    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return []
    width = float(size.findtext("width", default="0"))
    height = float(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        return []

    lines: list[str] = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.findtext("xmin", default="0"))
            ymin = float(bbox.findtext("ymin", default="0"))
            xmax = float(bbox.findtext("xmax", default="0"))
            ymax = float(bbox.findtext("ymax", default="0"))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare curated patient sequence data for demo-app.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("demo-app/data/patients"))
    parser.add_argument("--max-patients", type=int, default=10)
    parser.add_argument("--max-frames-per-patient", type=int, default=300)
    parser.add_argument(
        "--split-flat-by-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split flat image folders into multiple patient sequences by filename prefix.",
    )
    parser.add_argument(
        "--prefix-parts",
        type=int,
        default=3,
        help="How many underscore-separated filename parts define one sequence when splitting flat folders.",
    )
    parser.add_argument(
        "--min-frames-per-patient",
        type=int,
        default=20,
        help="Minimum frames required for a generated patient sequence.",
    )
    parser.add_argument("--default-fps", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"

    patient_dirs = collect_patient_dirs(source_root)
    patients: list[dict[str, object]] = []

    print(f"Found {len(patient_dirs)} candidate patient dirs under {source_root}")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    used_ids: set[str] = set()
    created_patients = 0
    for patient_dir in patient_dirs:
        image_paths = sorted(list_images(patient_dir), key=natural_key)
        if not image_paths:
            continue

        sequences = build_sequences(
            patient_dir=patient_dir,
            image_paths=image_paths,
            split_flat_by_prefix=args.split_flat_by_prefix,
            prefix_parts=args.prefix_parts,
            min_frames_per_patient=args.min_frames_per_patient,
        )

        for sequence_name, sequence_images in sequences:
            if created_patients >= args.max_patients:
                break

            patient_id_base = safe_id(f"{patient_dir.name}_{sequence_name}")
            patient_id = patient_id_base
            suffix_counter = 2
            while patient_id in used_ids:
                patient_id = f"{patient_id_base}_{suffix_counter:02d}"
                suffix_counter += 1
            used_ids.add(patient_id)

            selected_images = sequence_images[: args.max_frames_per_patient]
            if not selected_images:
                continue

            frames_dir = output_root / patient_id / "frames"
            labels_dir = output_root / patient_id / "labels"

            if not args.dry_run:
                frames_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            labels_written = 0
            for image_path in selected_images:
                target_image = frames_dir / image_path.name
                label_in_same_dir = image_path.with_suffix(".txt")
                xml_in_same_dir = image_path.with_suffix(".xml")
                target_label = labels_dir / f"{image_path.stem}.txt"

                if not args.dry_run:
                    shutil.copy2(image_path, target_image)
                    if label_in_same_dir.exists():
                        labels_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(label_in_same_dir, target_label)
                        labels_written += 1
                    else:
                        converted_lines = voc_xml_to_yolo_lines(xml_in_same_dir)
                        if converted_lines:
                            labels_dir.mkdir(parents=True, exist_ok=True)
                            target_label.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
                            labels_written += 1
                copied += 1

            patient_entry: dict[str, object] = {
                "id": patient_id,
                "displayName": patient_id.replace("_", " ").title(),
                "framesDir": f"{patient_id}/frames",
                "defaultFps": args.default_fps,
            }
            if labels_written > 0:
                patient_entry["labelsDir"] = f"{patient_id}/labels"
            patients.append(patient_entry)
            created_patients += 1
            print(f"Prepared {patient_id}: {copied} frames, {labels_written} labels")

        if created_patients >= args.max_patients:
            break

    manifest = {"patients": patients}
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"Manifest written: {manifest_path}")
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
