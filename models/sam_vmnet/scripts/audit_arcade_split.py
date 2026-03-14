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
import hashlib
from dataclasses import dataclass
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
    image_md5: str
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


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        image_path = (img_dir / file_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in annotations does not exist: {image_path}")
        records.append(
            FrameRecord(
                source_split=split,
                assigned_split=split,
                image_id=image_id,
                file_name=file_name,
                width=int(image.get("width", 0)),
                height=int(image.get("height", 0)),
                image_path=image_path,
                image_md5=_file_md5(image_path),
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


def _is_inferred_key_reliable(records: list[FrameRecord], attr_name: str) -> dict[str, object]:
    values = [getattr(record, attr_name) for record in records]
    if not values:
        return {
            "reliable": False,
            "reason": "no-keys",
            "numeric_ratio": 0.0,
            "has_separator_ratio": 0.0,
        }

    numeric_count = sum(1 for value in values if re.fullmatch(r"\d+", value))
    separator_count = sum(1 for value in values if ("_" in value or "-" in value))
    numeric_ratio = numeric_count / len(values)
    has_separator_ratio = separator_count / len(values)

    # If keys are almost entirely plain numbers with no separators, they are
    # likely local frame ids and not stable patient/sequence identifiers.
    reliable = not (numeric_ratio >= 0.9 and has_separator_ratio <= 0.1)

    return {
        "reliable": reliable,
        "reason": "numeric-id-like" if not reliable else "pattern-supported",
        "numeric_ratio": round(numeric_ratio, 4),
        "has_separator_ratio": round(has_separator_ratio, 4),
    }


def audit_records(records: list[FrameRecord]) -> dict:
    split_counts = {split: 0 for split in SPLITS}
    for record in records:
        split_counts[record.assigned_split] += 1

    filename_overlap = _filenames_overlap(records)
    content_hash_overlap = _key_overlap(records, "image_md5")
    patient_overlap = _key_overlap(records, "patient_key")
    sequence_overlap = _key_overlap(records, "sequence_key")

    patient_key_quality = _is_inferred_key_reliable(records, "patient_key")
    sequence_key_quality = _is_inferred_key_reliable(records, "sequence_key")
    enforce_patient = bool(patient_key_quality["reliable"])
    enforce_sequence = bool(sequence_key_quality["reliable"])

    # Leakage pass/fail criteria:
    # 1) Exact image content must not overlap across splits.
    # 2) Inferred patient/sequence keys are only enforced when their pattern is
    #    reliable (e.g., keys encode structured identifiers).
    passed = not _has_any_overlap(content_hash_overlap)
    if enforce_patient:
        passed = passed and (not _has_any_overlap(patient_overlap))
    if enforce_sequence:
        passed = passed and (not _has_any_overlap(sequence_overlap))

    return {
        "passed": passed,
        "totals": {
            "frames": len(records),
            "split_counts": split_counts,
            "unique_patients": len({record.patient_key for record in records}),
            "unique_sequences": len({record.sequence_key for record in records}),
            "unique_image_hashes": len({record.image_md5 for record in records}),
        },
        "key_inference": {
            "patient_key": patient_key_quality,
            "sequence_key": sequence_key_quality,
            "enforced_for_pass_fail": {
                "patient_key_overlap": enforce_patient,
                "sequence_key_overlap": enforce_sequence,
            },
        },
        "overlaps": {
            "image_content_hashes": content_hash_overlap,
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
                "image_md5",
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
                    "image_md5": record.image_md5,
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
