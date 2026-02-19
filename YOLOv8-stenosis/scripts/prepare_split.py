"""
Split dataset into train/val/test with stratification (stenosis vs no-stenosis).

Usage:
    python scripts/prepare_split.py \
        --images path/to/all/images \
        --labels path/to/all/labels \
        --output dataset
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def has_stenosis(label_path):
    """Check if a label file contains at least one bounding box."""
    with open(label_path) as f:
        for line in f:
            if line.strip():
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split")
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

    image_files = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
    label_files = {f.stem: f for f in labels_dir.iterdir() if f.suffix == ".txt"}
    matched = sorted(image_files.keys() & label_files.keys())

    positive = [s for s in matched if has_stenosis(label_files[s])]
    negative = [s for s in matched if not has_stenosis(label_files[s])]

    print(f"Total matched pairs: {len(matched)}")
    print(f"  With stenosis:     {len(positive)} ({len(positive)/len(matched)*100:.1f}%)")
    print(f"  Without stenosis:  {len(negative)} ({len(negative)/len(matched)*100:.1f}%)")
    print()

    random.seed(args.seed)
    random.shuffle(positive)
    random.shuffle(negative)

    def split_list(lst):
        n = len(lst)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    pos_train, pos_val, pos_test = split_list(positive)
    neg_train, neg_val, neg_test = split_list(negative)

    splits = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }

    for split_name, stems in splits.items():
        random.shuffle(stems)
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            shutil.copy2(image_files[stem], img_dir / image_files[stem].name)
            shutil.copy2(label_files[stem], lbl_dir / label_files[stem].name)

        n_pos = sum(1 for s in stems if s in set(positive))
        pct = n_pos / len(stems) * 100 if stems else 0
        print(f"{split_name:5s}: {len(stems):6d} images  ({n_pos} with stenosis = {pct:.1f}%)")

    # Write data.yaml
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: .\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: stenosis\n"
    )
    print(f"\ndata.yaml written to {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
