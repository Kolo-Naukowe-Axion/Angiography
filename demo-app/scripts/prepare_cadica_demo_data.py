#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo26m_cadica.scripts.cadica_selected_utils import (  # noqa: E402
    DEFAULT_CADICA_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    SPLITS,
    FrameSample,
    ensure_clean_output_root,
    iter_frame_samples,
    materialize_image,
    parse_cadica_groundtruth,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "demo-app" / "data" / "patients"
DEFAULT_FPS = 12


def sequence_id_for(sample: FrameSample) -> str:
    return f"cadica_{sample.video_key}"


def display_name_for(sample: FrameSample) -> str:
    return f"CADICA {sample.patient_id.upper()} {sample.video_id.upper()}"


def group_samples(samples: Iterable[FrameSample], split: str) -> dict[str, list[FrameSample]]:
    grouped: dict[str, list[FrameSample]] = defaultdict(list)
    for sample in samples:
        if sample.split != split:
            continue
        grouped[sample.video_key].append(sample)

    for key in grouped:
        grouped[key].sort(key=lambda item: item.frame_stem)

    return dict(sorted(grouped.items()))


def prepare_demo_data(
    cadica_root: Path,
    split_manifest: Path,
    output_root: Path,
    split: str,
    copy_images: bool = False,
    force: bool = False,
) -> dict[str, object]:
    cadica_root = cadica_root.resolve()
    split_manifest = split_manifest.resolve()
    output_root = output_root.resolve()

    ensure_clean_output_root(output_root, force=force)

    grouped = group_samples(iter_frame_samples(cadica_root, split_manifest), split=split)
    manifest_patients: list[dict[str, object]] = []

    total_frames = 0
    total_positive_frames = 0
    total_negative_frames = 0

    for video_key, samples in grouped.items():
        first = samples[0]
        patient_id = sequence_id_for(first)
        patient_root = output_root / patient_id
        frames_dir = patient_root / "frames"
        labels_dir = patient_root / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        positive_frames = 0
        for sample in samples:
            destination_image = frames_dir / sample.image_path.name
            destination_label = labels_dir / f"{sample.image_path.stem}.txt"

            materialize_image(sample.image_path, destination_image, copy_images=copy_images)

            label_lines: list[str] = []
            if sample.groundtruth_path is not None:
                label_lines = parse_cadica_groundtruth(sample.groundtruth_path, sample.image_path)
                positive_frames += 1

            label_text = "\n".join(label_lines)
            if label_text:
                label_text += "\n"
            destination_label.write_text(label_text, encoding="utf-8")

        frame_count = len(samples)
        negative_frames = frame_count - positive_frames
        total_frames += frame_count
        total_positive_frames += positive_frames
        total_negative_frames += negative_frames

        manifest_patients.append(
            {
                "id": patient_id,
                "displayName": display_name_for(first),
                "framesDir": f"{patient_id}/frames",
                "labelsDir": f"{patient_id}/labels",
                "defaultFps": DEFAULT_FPS,
                "datasetId": "cadica",
                "labelType": "bbox",
            }
        )

    manifest = {"patients": manifest_patients}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "split": split,
        "sequence_count": len(manifest_patients),
        "frame_count": total_frames,
        "positive_frame_count": total_positive_frames,
        "negative_frame_count": total_negative_frames,
        "materialization": "copy" if copy_images else "symlink",
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CADICA selected-video demo data for the angiography demo app.")
    parser.add_argument("--cadica-root", type=Path, default=DEFAULT_CADICA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing non-empty output directory before preparing the demo dataset.",
    )
    args = parser.parse_args()

    summary = prepare_demo_data(
        cadica_root=args.cadica_root,
        split_manifest=args.split_manifest,
        output_root=args.output_root,
        split=args.split,
        copy_images=args.copy_images,
        force=args.force,
    )

    print(f"Prepared CADICA demo data at {summary['output_root']}")
    print(f"Split: {summary['split']}")
    print(f"Sequences: {summary['sequence_count']}")
    print(f"Frames: {summary['frame_count']}")
    print(f"Positive frames: {summary['positive_frame_count']}")
    print(f"Negative frames: {summary['negative_frame_count']}")
    print(f"Materialization: {summary['materialization']}")


if __name__ == "__main__":
    main()
