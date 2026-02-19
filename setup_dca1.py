"""
Download DCA1 coronary angiogram dataset from Kaggle, convert PGM images/masks
to PNG, split into train/val/test, and organize for SAM-VMNet training.

DCA1: 130 grayscale 300x300 coronary angiogram images with expert vessel masks.
Source: https://www.kaggle.com/datasets/bard2024/database-x-ray-coronary-angiograms-dca1

Usage:
    # Set Kaggle credentials (or have ~/.kaggle/kaggle.json)
    export KAGGLE_USERNAME=your_username
    export KAGGLE_KEY=your_key
    python setup_dca1.py
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image

KAGGLE_DATASET = "bard2024/database-x-ray-coronary-angiograms-dca1"
DOWNLOAD_DIR = "data/dca1_raw"
OUTPUT_DIR = "data/dca1"
SEED = 42
TRAIN_COUNT = 104
VAL_COUNT = 13
TEST_COUNT = 13


def download_dataset():
    """Download DCA1 from Kaggle using the kaggle CLI."""
    if os.path.exists(DOWNLOAD_DIR) and any(
        f.endswith(".pgm") for f in os.listdir(DOWNLOAD_DIR)
        if os.path.isfile(os.path.join(DOWNLOAD_DIR, f))
    ):
        print(f"Raw data already exists at {DOWNLOAD_DIR}, skipping download.")
        return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Downloading DCA1 dataset from Kaggle...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", DOWNLOAD_DIR,
                "--unzip",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: 'kaggle' CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed: {e}")
        print("Check KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        sys.exit(1)

    # Handle case where kaggle doesn't auto-unzip
    zip_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.zip"))
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DOWNLOAD_DIR)
        os.remove(zf)

    print(f"Download complete. Files in {DOWNLOAD_DIR}/")


def find_image_mask_pairs():
    """Discover image/mask PGM pairs in the extracted directory.

    DCA1 structure: images named like '1_image.pgm' and masks '1_gt.pgm'
    or images in an 'image' folder and masks in a 'gt' folder.
    We search recursively for all PGM files and match them.
    """
    # Find all PGM files recursively
    all_pgms = glob.glob(os.path.join(DOWNLOAD_DIR, "**", "*.pgm"), recursive=True)

    if not all_pgms:
        # Also check for png/bmp/tif in case format differs
        for ext in ("*.png", "*.bmp", "*.tif", "*.tiff"):
            all_pgms.extend(
                glob.glob(os.path.join(DOWNLOAD_DIR, "**", ext), recursive=True)
            )

    if not all_pgms:
        print(f"ERROR: No image files found in {DOWNLOAD_DIR}/")
        print("Contents:")
        for root, dirs, files in os.walk(DOWNLOAD_DIR):
            for f in files[:20]:
                print(f"  {os.path.join(root, f)}")
        sys.exit(1)

    # Separate images and masks
    images = {}
    masks = {}

    for path in sorted(all_pgms):
        basename = os.path.basename(path).lower()
        # Try to extract numeric ID
        # Common patterns: "1_image.pgm"/"1_gt.pgm" or "image01.pgm"/"gt01.pgm"
        if "_gt" in basename or "/gt/" in path.lower() or "\\gt\\" in path.lower():
            # This is a mask
            # Extract ID: everything before _gt or the number
            name_part = basename.replace("_gt", "").split(".")[0]
            num = "".join(c for c in name_part if c.isdigit())
            if num:
                masks[int(num)] = path
        elif (
            "_image" in basename
            or "/image/" in path.lower()
            or "\\image\\" in path.lower()
        ):
            name_part = basename.replace("_image", "").split(".")[0]
            num = "".join(c for c in name_part if c.isdigit())
            if num:
                images[int(num)] = path
        else:
            # Try to guess: if basename is just a number, check parent folder
            name_part = basename.split(".")[0]
            num = "".join(c for c in name_part if c.isdigit())
            parent = os.path.basename(os.path.dirname(path)).lower()
            if num:
                if "gt" in parent or "mask" in parent or "label" in parent:
                    masks[int(num)] = path
                elif "image" in parent or "img" in parent or "original" in parent:
                    images[int(num)] = path

    # Match pairs
    common_ids = sorted(set(images.keys()) & set(masks.keys()))
    if not common_ids:
        print("ERROR: Could not match image/mask pairs.")
        print(f"  Found {len(images)} images and {len(masks)} masks")
        print(f"  Image IDs (first 10): {sorted(images.keys())[:10]}")
        print(f"  Mask IDs (first 10): {sorted(masks.keys())[:10]}")
        # Fallback: try alphabetical pairing
        all_sorted = sorted(all_pgms)
        mid = len(all_sorted) // 2
        if len(all_sorted) % 2 == 0:
            print("  Attempting alphabetical split (first half=images, second=masks)...")
            for i, (img_path, msk_path) in enumerate(
                zip(all_sorted[:mid], all_sorted[mid:])
            ):
                images[i + 1] = img_path
                masks[i + 1] = msk_path
            common_ids = sorted(set(images.keys()) & set(masks.keys()))

    if not common_ids:
        sys.exit(1)

    pairs = [(images[i], masks[i]) for i in common_ids]
    print(f"Found {len(pairs)} image/mask pairs")
    return pairs


def convert_and_split(pairs):
    """Convert PGM to PNG, threshold masks, and split into train/val/test."""
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(pairs))

    assert len(pairs) >= TRAIN_COUNT + VAL_COUNT + TEST_COUNT, (
        f"Expected at least {TRAIN_COUNT + VAL_COUNT + TEST_COUNT} pairs, "
        f"got {len(pairs)}"
    )

    splits = {
        "train": indices[:TRAIN_COUNT],
        "val": indices[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT],
        "test": indices[TRAIN_COUNT + VAL_COUNT : TRAIN_COUNT + VAL_COUNT + TEST_COUNT],
    }

    for split_name, split_indices in splits.items():
        img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
        msk_dir = os.path.join(OUTPUT_DIR, split_name, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        for i, idx in enumerate(split_indices):
            img_path, msk_path = pairs[idx]
            out_name = f"{i + 1:03d}.png"

            # Convert image to PNG (grayscale -> RGB for 3-channel input)
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(img_dir, out_name))

            # Convert mask to binary PNG (threshold at 127)
            msk = np.array(Image.open(msk_path).convert("L"))
            msk_binary = np.where(msk > 127, 255, 0).astype(np.uint8)
            Image.fromarray(msk_binary, mode="L").save(
                os.path.join(msk_dir, out_name)
            )

        print(f"  {split_name}: {len(split_indices)} images -> {img_dir}")


def compute_normalization_stats():
    """Compute mean/std pixel values for train and test splits."""
    stats = {}
    for split in ["train", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        all_pixels = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".png"):
                continue
            img = np.array(Image.open(os.path.join(img_dir, fname)).convert("RGB"))
            all_pixels.append(img.astype(np.float64))

        all_pixels = np.concatenate([p.reshape(-1) for p in all_pixels])
        mean = float(np.mean(all_pixels))
        std = float(np.std(all_pixels))
        stats[split] = {"mean": mean, "std": std}
        print(f"  {split}: mean={mean:.3f}, std={std:.3f}")

    return stats


def verify_masks():
    """Check mask polarity: vessels should be white (255) on black (0)."""
    train_msk_dir = os.path.join(OUTPUT_DIR, "train", "masks")
    masks = sorted(os.listdir(train_msk_dir))[:5]

    print("\nMask verification (first 5 train masks):")
    for fname in masks:
        msk = np.array(Image.open(os.path.join(train_msk_dir, fname)))
        white_pct = np.mean(msk == 255) * 100
        black_pct = np.mean(msk == 0) * 100
        print(f"  {fname}: {white_pct:.1f}% white, {black_pct:.1f}% black")

    # If more than 50% white on average, masks are likely inverted
    avg_white = np.mean(
        [
            np.mean(np.array(Image.open(os.path.join(train_msk_dir, f))) == 255)
            for f in masks
        ]
    )
    if avg_white > 0.5:
        print(
            "  WARNING: Masks appear inverted (>50% white). "
            "Vessels should be white on black background."
        )
        print("  Inverting all masks...")
        for split in ["train", "val", "test"]:
            msk_dir = os.path.join(OUTPUT_DIR, split, "masks")
            for fname in os.listdir(msk_dir):
                if not fname.endswith(".png"):
                    continue
                path = os.path.join(msk_dir, fname)
                msk = np.array(Image.open(path))
                msk_inv = 255 - msk
                Image.fromarray(msk_inv).save(path)
        print("  Masks inverted successfully.")


def main():
    print("=== DCA1 Dataset Setup ===\n")

    print("Step 1: Downloading dataset...")
    download_dataset()

    print("\nStep 2: Finding image/mask pairs...")
    pairs = find_image_mask_pairs()

    print(f"\nStep 3: Converting and splitting ({TRAIN_COUNT}/{VAL_COUNT}/{TEST_COUNT})...")
    convert_and_split(pairs)

    print("\nStep 4: Verifying mask polarity...")
    verify_masks()

    print("\nStep 5: Computing normalization statistics...")
    stats = compute_normalization_stats()

    print(f"\nDone! Dataset ready at ./{OUTPUT_DIR}/")
    print(f"\nUpdate SAM_VMNet/utils.py myNormalize.STATS['dca1'] with:")
    print(f"  'dca1': {{")
    print(f"      'train': {{'mean': {stats['train']['mean']:.3f}, 'std': {stats['train']['std']:.3f}}},")
    print(f"      'test':  {{'mean': {stats['test']['mean']:.3f}, 'std': {stats['test']['std']:.3f}}},")
    print(f"  }}")


if __name__ == "__main__":
    main()
