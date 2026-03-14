"""
Split dataset into train/val/test WITHOUT data leakage.

Leakage-prevention strategy:
  - CADICA: split by PATIENT ID so all frames from one patient stay in the
    same split. Stratify patients by whether they have any stenosis frames.
  - ARCADE: honour the original train/val/test split encoded in filenames
    (arcade_train_*, arcade_val_*, arcade_test_*).

Usage:
    python scripts/prepare_split.py \
        --images path/to/all/images \
        --labels path/to/all/labels \
        --output dataset
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Matches: cadica_p{patient}_v{video}_...
CADICA_PATIENT_RE = re.compile(r"^cadica_(p\d+)_")
# Matches: arcade_{split}_{rest}
ARCADE_SPLIT_RE = re.compile(r"^arcade_(train|val|test)_")


def has_stenosis(label_path):
    """Check if a label file contains at least one bounding box."""
    with open(label_path) as f:
        for line in f:
            if line.strip():
                return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-free stratified train/val/test split"
    )
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train + val ratio must be < 1.0"

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    image_files = {
        f.stem: f
        for f in images_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_files = {
        f.stem: f for f in labels_dir.iterdir() if f.suffix == ".txt"
    }
    matched = sorted(image_files.keys() & label_files.keys())

    positive_set = {s for s in matched if has_stenosis(label_files[s])}

    print(f"Total matched pairs: {len(matched)}")
    print(
        f"  With stenosis:     {len(positive_set)} "
        f"({len(positive_set)/len(matched)*100:.1f}%)"
    )
    print(
        f"  Without stenosis:  {len(matched) - len(positive_set)} "
        f"({(len(matched) - len(positive_set))/len(matched)*100:.1f}%)"
    )
    print()

    # ------------------------------------------------------------------ #
    #  1. Separate ARCADE vs CADICA stems                                 #
    # ------------------------------------------------------------------ #
    arcade_by_split = defaultdict(list)  # original_split -> [stem, ...]
    cadica_by_patient = defaultdict(list)  # patient_id -> [stem, ...]
    unknown = []

    for stem in matched:
        m_arcade = ARCADE_SPLIT_RE.match(stem)
        m_cadica = CADICA_PATIENT_RE.match(stem)

        if m_arcade:
            arcade_by_split[m_arcade.group(1)].append(stem)
        elif m_cadica:
            cadica_by_patient[m_cadica.group(1)].append(stem)
        else:
            unknown.append(stem)

    n_arcade = sum(len(v) for v in arcade_by_split.values())
    n_cadica = sum(len(v) for v in cadica_by_patient.values())

    print(f"ARCADE images: {n_arcade}  (splits: "
          f"train={len(arcade_by_split.get('train', []))}, "
          f"val={len(arcade_by_split.get('val', []))}, "
          f"test={len(arcade_by_split.get('test', []))})")
    print(f"CADICA images: {n_cadica}  ({len(cadica_by_patient)} patients)")
    if unknown:
        print(f"Unknown source: {len(unknown)} images (will be split with CADICA)")
    print()

    # ------------------------------------------------------------------ #
    #  2. ARCADE: honour original splits                                  #
    # ------------------------------------------------------------------ #
    arcade_splits = {"train": [], "val": [], "test": []}
    for orig_split, stems in arcade_by_split.items():
        arcade_splits[orig_split].extend(stems)

    # ------------------------------------------------------------------ #
    #  3. CADICA: patient-level stratified split                          #
    # ------------------------------------------------------------------ #
    # Classify each patient: "positive" if they have ANY stenosis frame.
    patient_has_stenosis = {}
    for patient_id, stems in cadica_by_patient.items():
        patient_has_stenosis[patient_id] = any(s in positive_set for s in stems)

    pos_patients = sorted(
        p for p, v in patient_has_stenosis.items() if v
    )
    neg_patients = sorted(
        p for p, v in patient_has_stenosis.items() if not v
    )

    random.seed(args.seed)
    random.shuffle(pos_patients)
    random.shuffle(neg_patients)

    def split_patients(patients):
        n = len(patients)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        return (
            patients[:n_train],
            patients[n_train : n_train + n_val],
            patients[n_train + n_val :],
        )

    pos_train_p, pos_val_p, pos_test_p = split_patients(pos_patients)
    neg_train_p, neg_val_p, neg_test_p = split_patients(neg_patients)

    cadica_patient_splits = {}
    for p in pos_train_p + neg_train_p:
        cadica_patient_splits[p] = "train"
    for p in pos_val_p + neg_val_p:
        cadica_patient_splits[p] = "val"
    for p in pos_test_p + neg_test_p:
        cadica_patient_splits[p] = "test"

    cadica_splits = {"train": [], "val": [], "test": []}
    for patient_id, stems in cadica_by_patient.items():
        target = cadica_patient_splits[patient_id]
        cadica_splits[target].extend(stems)

    # Unknown-source images: treat each as an independent "patient"
    random.shuffle(unknown)
    n_unk = len(unknown)
    n_unk_train = int(n_unk * args.train_ratio)
    n_unk_val = int(n_unk * args.val_ratio)
    unknown_splits = {
        "train": unknown[:n_unk_train],
        "val": unknown[n_unk_train : n_unk_train + n_unk_val],
        "test": unknown[n_unk_train + n_unk_val :],
    }

    # ------------------------------------------------------------------ #
    #  4. Merge and copy files                                            #
    # ------------------------------------------------------------------ #
    final_splits = {"train": [], "val": [], "test": []}
    for split_name in final_splits:
        final_splits[split_name] = (
            arcade_splits[split_name]
            + cadica_splits[split_name]
            + unknown_splits[split_name]
        )

    for split_name, stems in final_splits.items():
        random.shuffle(stems)
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            shutil.copy2(image_files[stem], img_dir / image_files[stem].name)
            shutil.copy2(label_files[stem], lbl_dir / label_files[stem].name)

        n_pos = sum(1 for s in stems if s in positive_set)
        pct = n_pos / len(stems) * 100 if stems else 0
        print(
            f"{split_name:5s}: {len(stems):6d} images  "
            f"({n_pos} with stenosis = {pct:.1f}%)"
        )

    # ------------------------------------------------------------------ #
    #  5. Integrity report                                                #
    # ------------------------------------------------------------------ #
    print("\n=== Leakage Integrity Check ===")

    # CADICA patient overlap
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        patients_a = {
            CADICA_PATIENT_RE.match(s).group(1)
            for s in cadica_splits[a]
            if CADICA_PATIENT_RE.match(s)
        }
        patients_b = {
            CADICA_PATIENT_RE.match(s).group(1)
            for s in cadica_splits[b]
            if CADICA_PATIENT_RE.match(s)
        }
        overlap = patients_a & patients_b
        status = "PASS" if not overlap else "FAIL"
        print(
            f"  CADICA patient overlap {a}/{b}: "
            f"{len(overlap)} patients [{status}]"
        )

    # ARCADE original split preservation
    for split_name, stems in arcade_splits.items():
        wrong = [
            s
            for s in stems
            if ARCADE_SPLIT_RE.match(s)
            and ARCADE_SPLIT_RE.match(s).group(1) != split_name
        ]
        status = "PASS" if not wrong else "FAIL"
        print(
            f"  ARCADE {split_name} split integrity: "
            f"{len(wrong)} misplaced [{status}]"
        )

    # Patient distribution detail
    print("\n=== CADICA Patient Distribution ===")
    for split_name in ["train", "val", "test"]:
        patients_in_split = sorted(
            cadica_patient_splits[p]
            for p in cadica_patient_splits
            if cadica_patient_splits[p] == split_name
        )
        patient_ids = sorted(
            p for p, s in cadica_patient_splits.items() if s == split_name
        )
        print(
            f"  {split_name:5s}: {len(patient_ids)} patients — "
            f"{', '.join(patient_ids)}"
        )

    # ------------------------------------------------------------------ #
    #  6. Write data.yaml                                                 #
    # ------------------------------------------------------------------ #
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: stenosis\n"
    )
    print(f"\ndata.yaml written to {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
