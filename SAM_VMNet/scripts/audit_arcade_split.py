#!/usr/bin/env python3
"""Audit ARCADE split integrity and leakage risk.

The audit validates three invariants across train/val/test:
1) No duplicate filenames across splits.
2) No patient-level key overlap.
3) No sequence-level key overlap.

The patient/sequence keys are inferred from frame filenames using ARCADE naming
heuristics (e.g. 14_002_5_0048 -> patient=14_002, sequence=14_002_5).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")


@dataclass
class FrameRecord:
    source_split: str
    assigned_split: str
    image_id: int
    file_name: str
    width: int
    height: int
    image_path: Path
    patient_key: str
    sequence_key: str
    annotations: list[dict]


def _split_tokens(stem: str) -> list[str]:
    tokens = [tok for tok in re.split(r"[_-]+", stem) if tok]
    if len(tokens) >= 3 and tokens[0].lower() == "arcade" and tokens[1].lower() in SPLITS:
        tokens = tokens[2:]
    return tokens


def infer_patient_sequence_keys(file_name: str) -> tuple[str, str]:
    stem = Path(file_name).stem
    tokens = _split_tokens(stem)
    if not tokens:
        return stem, stem

    base_tokens = tokens
    if len(tokens) >= 2 and tokens[-1].isdigit():
        base_tokens = tokens[:-1]

    if not base_tokens:
        base_tokens = tokens

    patient_tokens = base_tokens[:2] if len(base_tokens) >= 2 else base_tokens[:1]
    sequence_tokens = base_tokens[:3] if len(base_tokens) >= 3 else base_tokens

    patient_key = "_".join(patient_tokens) if patient_tokens else stem
    sequence_key = "_".join(sequence_tokens) if sequence_tokens else stem
    return patient_key, sequence_key


def _validate_syntax_root(syntax_root: Path) -> None:
    for split in SPLITS:
        ann_path = syntax_root / split / "annotations" / f"{split}.json"
        img_dir = syntax_root / split / "images"
        if not ann_path.exists() or not img_dir.exists():
            raise FileNotFoundError(
                f"Missing ARCADE syntax layout for split '{split}' under {syntax_root}. "
                f"Expected {ann_path} and {img_dir}."
            )


def _load_split_records(syntax_root: Path, split: str) -> list[FrameRecord]:
    ann_path = syntax_root / split / "annotations" / f"{split}.json"
    img_dir = syntax_root / split / "images"

    with ann_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id", -1))
        if image_id < 0:
            continue
        anns_by_image.setdefault(image_id, []).append(ann)

    records: list[FrameRecord] = []
    for image in coco.get("images", []):
        image_id = int(image.get("id", -1))
        file_name = str(image.get("file_name", ""))
        if image_id < 0 or not file_name:
            continue

        patient_key, sequence_key = infer_patient_sequence_keys(file_name)
        records.append(
            FrameRecord(
                source_split=split,
                assigned_split=split,
                image_id=image_id,
                file_name=file_name,
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                image_path=(img_dir / file_name).resolve(),
                patient_key=patient_key,
                sequence_key=sequence_key,
                annotations=anns_by_image.get(image_id, []),
            )
        )

    return records


def load_records_from_syntax_root(syntax_root: Path) -> list[FrameRecord]:
    _validate_syntax_root(syntax_root)
    records: list[FrameRecord] = []
    for split in SPLITS:
        records.extend(_load_split_records(syntax_root, split))
    return records


def _pairwise_overlap(keys_by_split: dict[str, set[str]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for a, b in combinations(SPLITS, 2):
        overlap = sorted(keys_by_split[a] & keys_by_split[b])
        out[f"{a}|{b}"] = {
            "count": len(overlap),
            "sample": overlap[:25],
        }
    return out


def _filenames_overlap(records: Iterable[FrameRecord]) -> dict[str, dict[str, object]]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        by_split[record.assigned_split].add(record.file_name)
    return _pairwise_overlap(by_split)


def _key_overlap(records: Iterable[FrameRecord], attr_name: str) -> dict[str, dict[str, object]]:
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        by_split[record.assigned_split].add(getattr(record, attr_name))
    return _pairwise_overlap(by_split)


def _has_any_overlap(overlap_block: dict[str, dict[str, object]]) -> bool:
    return any(int(entry["count"]) > 0 for entry in overlap_block.values())


def audit_records(records: list[FrameRecord]) -> dict:
    split_counts = {split: 0 for split in SPLITS}
    for record in records:
        split_counts[record.assigned_split] += 1

    filename_overlap = _filenames_overlap(records)
    patient_overlap = _key_overlap(records, "patient_key")
    sequence_overlap = _key_overlap(records, "sequence_key")

    passed = not (
        _has_any_overlap(filename_overlap)
        or _has_any_overlap(patient_overlap)
        or _has_any_overlap(sequence_overlap)
    )

    return {
        "passed": passed,
        "totals": {
            "frames": len(records),
            "split_counts": split_counts,
            "unique_patients": len({record.patient_key for record in records}),
            "unique_sequences": len({record.sequence_key for record in records}),
        },
        "overlaps": {
            "filenames": filename_overlap,
            "patient_keys": patient_overlap,
            "sequence_keys": sequence_overlap,
        },
    }


def write_index_csv(records: list[FrameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_split",
                "assigned_split",
                "image_id",
                "file_name",
                "patient_key",
                "sequence_key",
                "image_path",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_split": record.source_split,
                    "assigned_split": record.assigned_split,
                    "image_id": record.image_id,
                    "file_name": record.file_name,
                    "patient_key": record.patient_key,
                    "sequence_key": record.sequence_key,
                    "image_path": str(record.image_path),
                    "width": record.width,
                    "height": record.height,
                }
            )


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ARCADE split leakage/integrity.")
    parser.add_argument("--syntax-root", type=Path, required=True, help="Path to ARCADE syntax root with train/val/test")
    parser.add_argument("--report-json", type=Path, default=Path("split_audit.json"))
    parser.add_argument("--index-csv", type=Path, default=Path("split_index.csv"))
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    syntax_root = args.syntax_root.resolve()
    records = load_records_from_syntax_root(syntax_root)

    report = audit_records(records)
    write_index_csv(records, args.index_csv.resolve())
    write_report(report, args.report_json.resolve())

    print(json.dumps(report, indent=2))

    if args.strict and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
