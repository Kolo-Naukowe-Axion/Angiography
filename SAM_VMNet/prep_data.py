"""
One-shot script to prepare ./data/vessel/ from the Mendeley dataset CSVs.

Reads train_labels.csv and test_labels.csv, creates images + masks for
train/val/test splits (val is carved out of train at 90/10 by patient).
"""

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---- paths (edit if needed) ----
SRC_DIR = '/Users/iwosmura/projects/angiography home/saved samvnet shit/Stenosis detection'
DATASET_DIR = os.path.join(SRC_DIR, 'dataset')
TRAIN_CSV = os.path.join(SRC_DIR, 'train_labels.csv')
TEST_CSV = os.path.join(SRC_DIR, 'test_labels.csv')
OUTPUT_DIR = './data/vessel/'
SEED = 42
VAL_RATIO = 0.1  # fraction of *train* patients to hold out as val


def read_csv(path):
    """Read a labels CSV and group bboxes by filename."""
    grouped = defaultdict(list)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['filename']
            bbox = (
                int(row['xmin']),
                int(row['ymin']),
                int(row['xmax']),
                int(row['ymax']),
            )
            grouped[fname].append(bbox)
    return grouped  # {filename: [(xmin, ymin, xmax, ymax), ...]}


def patient_id(filename):
    """Extract patient ID from Mendeley filename (e.g. '14_024' from '14_024_2_0042.bmp')."""
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) >= 2:
        return f'{parts[0]}_{parts[1]}'
    return stem


def split_by_patient(grouped, val_ratio, seed):
    """Split a grouped dict into train and val by patient ID."""
    random.seed(seed)
    patients = defaultdict(list)
    for fname in grouped:
        pid = patient_id(fname)
        patients[pid].append(fname)

    pids = sorted(patients.keys())
    random.shuffle(pids)

    n_val = max(1, int(len(pids) * val_ratio))
    val_pids = set(pids[:n_val])
    train_pids = set(pids[n_val:])

    train_grouped = {f: grouped[f] for pid in train_pids for f in patients[pid]}
    val_grouped = {f: grouped[f] for pid in val_pids for f in patients[pid]}
    return train_grouped, val_grouped


def process_split(grouped, split_name, output_dir, dataset_dir):
    """Create images/ and masks/ for a split."""
    img_out = os.path.join(output_dir, split_name, 'images')
    mask_out = os.path.join(output_dir, split_name, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    # Also create feature/ dir so feature_processor doesn't need to
    os.makedirs(os.path.join(output_dir, split_name, 'feature'), exist_ok=True)

    saved = 0
    skipped = 0

    for fname in tqdm(sorted(grouped.keys()), desc=f'{split_name}'):
        src_path = os.path.join(dataset_dir, fname)
        if not os.path.isfile(src_path):
            skipped += 1
            continue

        # Load image
        img = Image.open(src_path).convert('RGB')
        img_arr = np.array(img)

        # Create mask from bboxes (xmin, ymin, xmax, ymax)
        mask = np.zeros(img_arr.shape[:2], dtype=np.uint8)
        for (xmin, ymin, xmax, ymax) in grouped[fname]:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_arr.shape[1], xmax)
            ymax = min(img_arr.shape[0], ymax)
            mask[ymin:ymax, xmin:xmax] = 255

        out_name = f'{saved:05d}.png'
        img.save(os.path.join(img_out, out_name))
        Image.fromarray(mask).save(os.path.join(mask_out, out_name))
        saved += 1

    return saved, skipped


def main():
    print(f'Source:  {SRC_DIR}')
    print(f'Output:  {OUTPUT_DIR}')

    # 1. Read CSVs
    print('\n--- Reading CSVs ---')
    train_all = read_csv(TRAIN_CSV)
    test_grouped = read_csv(TEST_CSV)
    print(f'  train_labels.csv: {len(train_all)} unique images')
    print(f'  test_labels.csv:  {len(test_grouped)} unique images')

    # 2. Split train -> train + val
    print(f'\n--- Splitting train into train/val (val_ratio={VAL_RATIO}) ---')
    train_grouped, val_grouped = split_by_patient(train_all, VAL_RATIO, SEED)
    print(f'  train: {len(train_grouped)} images')
    print(f'  val:   {len(val_grouped)} images')
    print(f'  test:  {len(test_grouped)} images')

    # 3. Process each split
    print('\n--- Processing splits ---')
    for name, grouped in [('train', train_grouped), ('val', val_grouped), ('test', test_grouped)]:
        saved, skipped = process_split(grouped, name, OUTPUT_DIR, DATASET_DIR)
        print(f'  {name}: saved={saved}, skipped={skipped}')

    print(f'\nDone! Dataset at {OUTPUT_DIR}')
    print('Next: run feature_processor.py to generate MedSAM features.')


if __name__ == '__main__':
    main()
