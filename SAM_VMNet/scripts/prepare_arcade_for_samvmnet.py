#!/usr/bin/env python3
"""Prepare ARCADE for SAM-VMNet training on cloud.

Pipeline:
1) Use the local `datasets/arcade/data` tree when available, otherwise download ARCADE from Zenodo (unless --syntax-root is provided).
2) Audit official split for leakage.
3) Optionally auto-rebuild split assignment if leakage is detected.
4) Convert COCO polygons into binary vessel masks and build SAM-VMNet dataset:
   datasets/arcade/data/vessel/{train,val,test}/{images,masks}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from audit_arcade_split import SPLITS, FrameRecord, audit_records, load_records_from_syntax_root, write_index_csv, write_report
from rebuild_arcade_split import reassign_records

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCADE_ROOT = REPO_ROOT / "datasets" / "arcade"
DEFAULT_LOCAL_SYNTAX_ROOT = ARCADE_ROOT / "data"
DEFAULT_DOWNLOAD_ROOT = ARCADE_ROOT / "downloads"
DEFAULT_OUTPUT_VESSEL_ROOT = ARCADE_ROOT / "data" / "vessel"
DEFAULT_REPORT_DIR = ARCADE_ROOT / "data" / "vessel_meta"


def fetch_zenodo_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _pick_zenodo_file(record: dict, preferred_key: str | None) -> dict:
    files = record.get("files", [])
    if not files:
        raise RuntimeError("Zenodo record has no files to download.")

    if preferred_key:
        for file_info in files:
            if file_info.get("key") == preferred_key:
                return file_info
        raise RuntimeError(f"Preferred file key '{preferred_key}' was not found in the record.")

    for file_info in files:
        if str(file_info.get("key", "")).lower() == "arcade.zip":
            return file_info

    zip_files = [f for f in files if str(f.get("key", "")).lower().endswith(".zip")]
    if zip_files:
        return max(zip_files, key=lambda f: int(f.get("size", 0)))

    return max(files, key=lambda f: int(f.get("size", 0)))


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_md5(checksum: str | None) -> str | None:
    if not checksum:
        return None
    checksum = str(checksum)
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1]
    return None


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def extract_archive(archive_path: Path, extract_root: Path) -> None:
    marker = extract_root / ".extracted"
    if marker.exists():
        return

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_root)

    marker.write_text("ok\n", encoding="utf-8")


def _is_syntax_root(path: Path) -> bool:
    for split in SPLITS:
        if not (path / split / "images").exists():
            return False
        if not (path / split / "annotations" / f"{split}.json").exists():
            return False
    return True


def discover_syntax_root(search_root: Path) -> Path:
    candidates: list[Path] = [search_root]
    for rel in ("syntax", "data", "data/syntax", "arcade", "arcade/syntax"):
        candidates.append(search_root / rel)

    for candidate in candidates:
        if candidate.exists() and _is_syntax_root(candidate):
            return candidate.resolve()

    for candidate in sorted(search_root.rglob("*")):
        if candidate.is_dir() and _is_syntax_root(candidate):
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find ARCADE syntax split layout under {search_root}. "
        "Expected train/val/test with images and annotations/*.json"
    )


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


def _prepare_output_root(output_root: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for split in SPLITS:
            split_root = output_root / split
            if split_root.exists():
                shutil.rmtree(split_root)

    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "masks").mkdir(parents=True, exist_ok=True)


def write_vessel_dataset(records: list[FrameRecord], output_root: Path, overwrite: bool, dry_run: bool) -> dict[str, int]:
    split_counts = {split: 0 for split in SPLITS}
    if not dry_run:
        _prepare_output_root(output_root, overwrite)

    for record in records:
        target_split = record.assigned_split
        split_counts[target_split] += 1

        if dry_run:
            continue

        image_out = output_root / target_split / "images" / record.file_name
        mask_out = output_root / target_split / "masks" / record.file_name

        image_out.parent.mkdir(parents=True, exist_ok=True)
        mask_out.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(record.image_path, image_out)
        mask = render_binary_mask(record.width, record.height, record.annotations)
        mask.save(mask_out)

    return split_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ARCADE split and SAM-VMNet vessel dataset on cloud.")
    parser.add_argument("--syntax-root", type=Path, default=None, help="Use an existing ARCADE syntax root; skips download")
    parser.add_argument("--zenodo-record", type=str, default="10390295")
    parser.add_argument("--zenodo-file-key", type=str, default=None)
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT)
    parser.add_argument("--output-vessel-root", type=Path, default=DEFAULT_OUTPUT_VESSEL_ROOT)
    parser.add_argument("--audit-policy", choices=("fail", "auto-rebuild"), default="auto-rebuild")
    parser.add_argument("--group-level", choices=("patient", "sequence"), default="patient")
    parser.add_argument("--train-ratio", type=float, default=0.667)
    parser.add_argument("--val-ratio", type=float, default=0.133)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_info: dict[str, object] = {}

    if args.syntax_root is not None:
        syntax_root = discover_syntax_root(args.syntax_root.resolve())
        source_info["mode"] = "existing-syntax-root"
        source_info["syntax_root"] = str(syntax_root)
    elif DEFAULT_LOCAL_SYNTAX_ROOT.exists():
        syntax_root = discover_syntax_root(DEFAULT_LOCAL_SYNTAX_ROOT.resolve())
        source_info["mode"] = "local-datasets-root"
        source_info["syntax_root"] = str(syntax_root)
    else:
        download_root = args.download_root.resolve()
        download_root.mkdir(parents=True, exist_ok=True)

        record = fetch_zenodo_record(args.zenodo_record)
        file_info = _pick_zenodo_file(record, args.zenodo_file_key)

        file_key = str(file_info.get("key"))
        download_url = str(file_info.get("links", {}).get("self"))
        checksum = str(file_info.get("checksum", ""))

        if not download_url:
            raise RuntimeError("Zenodo file metadata did not include a download URL.")

        archive_path = download_root / file_key
        expected_md5 = _expected_md5(checksum)
        if archive_path.exists() and expected_md5:
            if md5sum(archive_path) != expected_md5:
                archive_path.unlink()

        if not archive_path.exists():
            print(f"Downloading {download_url} -> {archive_path}")
            download_file(download_url, archive_path)

        if expected_md5:
            actual_md5 = md5sum(archive_path)
            if actual_md5 != expected_md5:
                raise RuntimeError(
                    f"Checksum mismatch for {archive_path}. expected={expected_md5} actual={actual_md5}"
                )

        extract_root = download_root / "extracted"
        extract_archive(archive_path, extract_root)
        syntax_root = discover_syntax_root(extract_root)

        source_info.update(
            {
                "mode": "zenodo-download",
                "record_id": args.zenodo_record,
                "file_key": file_key,
                "download_url": download_url,
                "archive_path": str(archive_path),
                "syntax_root": str(syntax_root),
                "checksum": checksum,
            }
        )

    records = load_records_from_syntax_root(syntax_root)

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    official_report = audit_records(records)
    write_report(official_report, report_dir / "official_split_audit.json")
    write_index_csv(records, report_dir / "official_split_index.csv")

    final_records = records
    final_report = official_report
    rebuild_applied = False

    if not official_report["passed"]:
        if args.audit_policy == "fail":
            print("Official split failed leakage audit and audit policy is 'fail'.")
            return 2

        print("Official split failed leakage audit. Rebuilding split assignment...")
        final_records = reassign_records(
            records=records,
            group_level=args.group_level,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        final_report = audit_records(final_records)
        rebuild_applied = True
        if not final_report["passed"]:
            print("Rebuilt split still failed leakage audit.")
            write_report(final_report, report_dir / "final_split_audit.json")
            write_index_csv(final_records, report_dir / "final_split_index.csv")
            return 3

    split_counts = write_vessel_dataset(
        records=final_records,
        output_root=args.output_vessel_root.resolve(),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    final_report["preparation"] = {
        "rebuild_applied": rebuild_applied,
        "audit_policy": args.audit_policy,
        "group_level": args.group_level,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "dry_run": args.dry_run,
        "output_vessel_root": str(args.output_vessel_root.resolve()),
        "split_counts_written": split_counts,
    }
    final_report["source"] = source_info

    write_report(final_report, report_dir / "final_split_audit.json")
    write_index_csv(final_records, report_dir / "final_split_index.csv")

    summary = {
        "passed": final_report["passed"],
        "rebuild_applied": rebuild_applied,
        "output_vessel_root": str(args.output_vessel_root.resolve()),
        "report_dir": str(report_dir),
        "split_counts_written": split_counts,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
