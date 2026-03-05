#!/usr/bin/env python3
"""Rebuild ARCADE split assignment to avoid leakage.

This script reassigns records to train/val/test by grouped keys (patient or
sequence) and writes a deterministic split map.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from audit_arcade_split import SPLITS, FrameRecord, audit_records, load_records_from_syntax_root, write_index_csv, write_report


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _group_key(record: FrameRecord, group_level: str) -> str:
    if group_level == "patient":
        return record.patient_key
    if group_level == "sequence":
        return record.sequence_key
    raise ValueError(f"Unsupported group level: {group_level}")


def reassign_records(
    records: list[FrameRecord],
    group_level: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[FrameRecord]:
    train_ratio, val_ratio, test_ratio = _normalize_ratios(train_ratio, val_ratio, test_ratio)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    groups: dict[str, list[FrameRecord]] = defaultdict(list)
    for record in records:
        groups[_group_key(record, group_level)].append(record)

    total_frames = len(records)
    targets = {split: int(total_frames * ratios[split]) for split in SPLITS}
    targets["test"] = total_frames - targets["train"] - targets["val"]

    group_items = list(groups.items())
    group_items.sort(key=lambda item: item[0])
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    current = {split: 0 for split in SPLITS}

    for _, group_records in group_items:
        best_split = max(
            SPLITS,
            key=lambda split: (targets[split] - current[split], -current[split]),
        )
        for record in group_records:
            record.assigned_split = best_split
        current[best_split] += len(group_records)

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild leakage-safe ARCADE split assignment.")
    parser.add_argument("--syntax-root", type=Path, required=True)
    parser.add_argument("--group-level", choices=("patient", "sequence"), default="patient")
    parser.add_argument("--train-ratio", type=float, default=0.667)
    parser.add_argument("--val-ratio", type=float, default=0.133)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, default=Path("split_rebuild_audit.json"))
    parser.add_argument("--index-csv", type=Path, default=Path("split_rebuild_index.csv"))
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    records = load_records_from_syntax_root(args.syntax_root.resolve())
    records = reassign_records(
        records=records,
        group_level=args.group_level,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    report = audit_records(records)
    report["rebuild"] = {
        "group_level": args.group_level,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
    }

    write_index_csv(records, args.index_csv.resolve())
    write_report(report, args.report_json.resolve())
    print(json.dumps(report, indent=2))

    if args.strict and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
