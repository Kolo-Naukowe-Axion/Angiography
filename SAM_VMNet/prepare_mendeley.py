"""
Prepare the Mendeley Angiographic Dataset for SAM-VMNet training.

Dataset: https://data.mendeley.com/datasets/ydrm75xywg/1

This script:
1. Reads LabelBox bounding box annotations (JSON or NDJSON)
2. Creates binary mask PNGs (white filled rectangles on black background)
3. Splits data into train/val/test (80/10/10) with patient-level splitting
4. Outputs to ./data/vessel/{train,val,test}/{images,masks}/

Usage:
    python prepare_mendeley.py --data_dir /path/to/mendeley/download --output_dir ./data/vessel/

The Mendeley dataset contains X-ray coronary angiography images with stenosis
annotations as bounding boxes. SAM-VMNet expects binary mask PNGs, so we
convert bounding boxes into filled white rectangles on a black background.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_annotations(annotation_path):
    """
    Load annotations from LabelBox export (JSON or NDJSON).

    Returns a list of dicts with keys:
        - 'image_filename': str
        - 'bboxes': list of (x, y, w, h) tuples

    Supports multiple annotation formats:
        1. LabelBox JSON export (list of annotation objects)
        2. NDJSON (one JSON object per line)
        3. Simple JSON with image_name -> bbox mapping
    """
    annotations = []

    with open(annotation_path, 'r') as f:
        content = f.read().strip()

    # Try NDJSON (one JSON per line)
    if content.startswith('{') and '\n' in content:
        lines = content.split('\n')
        try:
            records = [json.loads(line) for line in lines if line.strip()]
            annotations = _parse_labelbox_records(records)
            if annotations:
                return annotations
        except json.JSONDecodeError:
            pass

    # Try standard JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Cannot parse annotation file: {annotation_path}")

    # If it's a list, try LabelBox format
    if isinstance(data, list):
        annotations = _parse_labelbox_records(data)
        if annotations:
            return annotations

    # Try simple dict format: {"image_name": [{"x": ..., "y": ..., "w": ..., "h": ...}, ...]}
    if isinstance(data, dict):
        for img_name, bboxes_data in data.items():
            if isinstance(bboxes_data, list):
                bboxes = []
                for bb in bboxes_data:
                    if isinstance(bb, dict):
                        bboxes.append(_extract_bbox(bb))
                    elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                        bboxes.append(tuple(bb))
                if bboxes:
                    annotations.append({
                        'image_filename': img_name,
                        'bboxes': bboxes,
                    })
        if annotations:
            return annotations

    raise ValueError(
        f"Unrecognized annotation format in {annotation_path}. "
        "Please check the Mendeley dataset structure and update this script."
    )


def _parse_labelbox_records(records):
    """Parse LabelBox export format records."""
    annotations = []
    for record in records:
        image_filename = None
        bboxes = []

        # LabelBox v2 NDJSON format
        if 'data_row' in record:
            image_filename = record.get('data_row', {}).get('external_id', '')
            if not image_filename:
                row_data = record.get('data_row', {}).get('row_data', '')
                if row_data:
                    image_filename = os.path.basename(row_data)

            # Extract annotations from projects or labels
            projects = record.get('projects', {})
            for proj_id, proj_data in projects.items():
                for label in proj_data.get('labels', []):
                    for ann in label.get('annotations', {}).get('objects', []):
                        bbox = ann.get('bounding_box', {})
                        if bbox:
                            bboxes.append((
                                int(bbox.get('left', bbox.get('x', 0))),
                                int(bbox.get('top', bbox.get('y', 0))),
                                int(bbox.get('width', bbox.get('w', 0))),
                                int(bbox.get('height', bbox.get('h', 0))),
                            ))

        # LabelBox v1 JSON export format
        elif 'External ID' in record or 'external_id' in record:
            image_filename = record.get('External ID', record.get('external_id', ''))
            label_data = record.get('Label', record.get('label', {}))
            if isinstance(label_data, str):
                try:
                    label_data = json.loads(label_data)
                except json.JSONDecodeError:
                    label_data = {}

            if isinstance(label_data, dict):
                objects = label_data.get('objects', [])
                for obj in objects:
                    bbox = obj.get('bbox', obj.get('bounding_box', {}))
                    if bbox:
                        bboxes.append(_extract_bbox(bbox))

        # Generic format with filename and bbox fields
        elif 'filename' in record or 'image' in record or 'file_name' in record:
            image_filename = record.get('filename',
                            record.get('image',
                            record.get('file_name', '')))
            bbox_data = record.get('bbox', record.get('bounding_box',
                        record.get('annotations', [])))
            if isinstance(bbox_data, dict):
                bboxes.append(_extract_bbox(bbox_data))
            elif isinstance(bbox_data, list):
                for bb in bbox_data:
                    if isinstance(bb, dict):
                        bboxes.append(_extract_bbox(bb))
                    elif isinstance(bb, (list, tuple)) and len(bb) == 4:
                        bboxes.append(tuple(int(v) for v in bb))

        if image_filename and bboxes:
            annotations.append({
                'image_filename': image_filename,
                'bboxes': bboxes,
            })

    return annotations


def _extract_bbox(bbox_dict):
    """Extract (x, y, w, h) from various bbox dict formats."""
    x = int(bbox_dict.get('left', bbox_dict.get('x', bbox_dict.get('Left', 0))))
    y = int(bbox_dict.get('top', bbox_dict.get('y', bbox_dict.get('Top', 0))))
    w = int(bbox_dict.get('width', bbox_dict.get('w', bbox_dict.get('Width', 0))))
    h = int(bbox_dict.get('height', bbox_dict.get('h', bbox_dict.get('Height', 0))))
    return (x, y, w, h)


def create_binary_mask(image_shape, bboxes):
    """
    Create a binary mask from bounding box annotations.

    Args:
        image_shape: (height, width) of the source image
        bboxes: list of (x, y, w, h) bounding boxes

    Returns:
        Binary mask as numpy array (H, W), uint8, values 0 or 255
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for (x, y, w, h) in bboxes:
        # Clamp to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_shape[1], x + w)
        y2 = min(image_shape[0], y + h)
        mask[y1:y2, x1:x2] = 255
    return mask


def find_annotation_file(data_dir):
    """Find the annotation file in the dataset directory."""
    data_dir = Path(data_dir)

    # Common annotation file patterns
    patterns = [
        '*.json', '*.ndjson', '**/*.json', '**/*.ndjson',
        'annotations/*', 'labels/*',
    ]

    candidates = []
    for pattern in patterns:
        for f in data_dir.glob(pattern):
            if f.is_file() and f.suffix in ('.json', '.ndjson'):
                candidates.append(f)

    if not candidates:
        return None

    # Prefer files with common annotation names
    priority_names = ['annotations', 'labels', 'export', 'labelbox', 'bbox', 'stenosis']
    for name in priority_names:
        for c in candidates:
            if name in c.stem.lower():
                return str(c)

    # Return the largest JSON file (most likely the annotation file)
    candidates.sort(key=lambda f: f.stat().st_size, reverse=True)
    return str(candidates[0])


def find_images(data_dir):
    """Find all image files in the dataset directory."""
    data_dir = Path(data_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = {}

    for f in data_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in image_extensions:
            images[f.name] = str(f)

    return images


def extract_patient_id(filename):
    """
    Extract a patient ID from the filename for patient-level splitting.

    Mendeley angiography filenames: "14_024_2_0042.bmp"
        prefix=14, patient=024, sequence=2, frame=0042
        -> patient ID = "14_024"

    Generic fallback: first segment before separator.
    """
    stem = Path(filename).stem
    parts = stem.replace('-', '_').split('_')

    # Mendeley format: prefix_patientID_sequence_frame (4+ segments)
    if len(parts) >= 4:
        return f"{parts[0]}_{parts[1]}"

    if len(parts) >= 2:
        return parts[0]

    return stem


def patient_level_split(annotations, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Split annotations by patient ID to avoid data leakage.

    Returns (train_annotations, val_annotations, test_annotations)
    """
    random.seed(seed)

    # Group by patient
    patient_groups = defaultdict(list)
    for ann in annotations:
        pid = extract_patient_id(ann['image_filename'])
        patient_groups[pid].append(ann)

    # Shuffle patients
    patient_ids = list(patient_groups.keys())
    random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_pids = set(patient_ids[:n_train])
    val_pids = set(patient_ids[n_train:n_train + n_val])
    test_pids = set(patient_ids[n_train + n_val:])

    train_anns = [a for pid in train_pids for a in patient_groups[pid]]
    val_anns = [a for pid in val_pids for a in patient_groups[pid]]
    test_anns = [a for pid in test_pids for a in patient_groups[pid]]

    return train_anns, val_anns, test_anns


def process_and_save(annotations, image_lookup, output_dir, split_name):
    """
    Process annotations and save images + masks for a given split.

    Args:
        annotations: list of annotation dicts
        image_lookup: dict mapping filename -> full path
        output_dir: base output directory (e.g., ./data/vessel/)
        split_name: 'train', 'val', or 'test'
    """
    img_out = os.path.join(output_dir, split_name, 'images')
    mask_out = os.path.join(output_dir, split_name, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    skipped = 0
    saved = 0

    for ann in tqdm(annotations, desc=f'Processing {split_name}'):
        filename = ann['image_filename']

        # Find the image file
        img_path = image_lookup.get(filename)
        if img_path is None:
            # Try without extension or with different extension
            stem = Path(filename).stem
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                candidate = stem + ext
                if candidate in image_lookup:
                    img_path = image_lookup[candidate]
                    break

        if img_path is None:
            skipped += 1
            continue

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
            skipped += 1
            continue

        img_array = np.array(img)

        # Create binary mask from bounding boxes
        mask = create_binary_mask(img_array.shape, ann['bboxes'])

        # Save with consistent naming (zero-padded index)
        out_name = f'{saved:05d}.png'
        img.save(os.path.join(img_out, out_name))
        Image.fromarray(mask).save(os.path.join(mask_out, out_name))
        saved += 1

    return saved, skipped


def try_labelbox_csv(data_dir):
    """
    Try to load annotations from a CSV file (alternative Mendeley format).

    Some Mendeley datasets provide CSV annotations with columns like:
    filename, xmin, ymin, xmax, ymax, class
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.rglob('*.csv'))

    if not csv_files:
        return None

    try:
        import csv
    except ImportError:
        return None

    annotations = []
    for csv_path in csv_files:
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                if not headers:
                    continue

                # Check for common column names
                has_bbox_cols = any(h.lower() in ('xmin', 'x_min', 'x1', 'left', 'bbox_x')
                                   for h in headers)
                has_file_col = any(h.lower() in ('filename', 'file_name', 'image', 'image_name')
                                   for h in headers)

                if not (has_bbox_cols and has_file_col):
                    continue

                # Find column names
                file_col = next(h for h in headers
                               if h.lower() in ('filename', 'file_name', 'image', 'image_name'))
                x_col = next((h for h in headers
                             if h.lower() in ('xmin', 'x_min', 'x1', 'left', 'bbox_x')), None)
                y_col = next((h for h in headers
                             if h.lower() in ('ymin', 'y_min', 'y1', 'top', 'bbox_y')), None)

                # Check if we have x2/y2 or w/h format
                x2_col = next((h for h in headers
                              if h.lower() in ('xmax', 'x_max', 'x2', 'right')), None)
                y2_col = next((h for h in headers
                              if h.lower() in ('ymax', 'y_max', 'y2', 'bottom')), None)
                w_col = next((h for h in headers
                             if h.lower() in ('width', 'w', 'bbox_w', 'bbox_width')), None)
                h_col = next((h for h in headers
                             if h.lower() in ('height', 'h', 'bbox_h', 'bbox_height')), None)

                if not (x_col and y_col):
                    continue

                # Group bboxes by image
                img_bboxes = defaultdict(list)
                for row in reader:
                    fname = row[file_col]
                    x = int(float(row[x_col]))
                    y = int(float(row[y_col]))

                    if x2_col and y2_col:
                        x2 = int(float(row[x2_col]))
                        y2 = int(float(row[y2_col]))
                        w = x2 - x
                        h = y2 - y
                    elif w_col and h_col:
                        w = int(float(row[w_col]))
                        h = int(float(row[h_col]))
                    else:
                        continue

                    img_bboxes[fname].append((x, y, w, h))

                for fname, bboxes in img_bboxes.items():
                    annotations.append({
                        'image_filename': fname,
                        'bboxes': bboxes,
                    })

        except Exception:
            continue

    return annotations if annotations else None


def try_xml_annotations(data_dir):
    """
    Try to load annotations from Pascal VOC XML format.
    """
    data_dir = Path(data_dir)
    xml_files = list(data_dir.rglob('*.xml'))

    if not xml_files:
        return None

    try:
        import xml.etree.ElementTree as ET
    except ImportError:
        return None

    annotations = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_elem = root.find('filename')
            if filename_elem is None:
                continue

            filename = filename_elem.text
            bboxes = []

            for obj in root.findall('object'):
                bbox_elem = obj.find('bndbox')
                if bbox_elem is None:
                    continue

                xmin = int(float(bbox_elem.find('xmin').text))
                ymin = int(float(bbox_elem.find('ymin').text))
                xmax = int(float(bbox_elem.find('xmax').text))
                ymax = int(float(bbox_elem.find('ymax').text))
                bboxes.append((xmin, ymin, xmax - xmin, ymax - ymin))

            if filename and bboxes:
                annotations.append({
                    'image_filename': filename,
                    'bboxes': bboxes,
                })
        except Exception:
            continue

    return annotations if annotations else None


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Mendeley Angiographic Dataset for SAM-VMNet training'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the downloaded Mendeley dataset directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./data/vessel/',
        help='Output directory for prepared dataset (default: ./data/vessel/)'
    )
    parser.add_argument(
        '--annotation_file', type=str, default=None,
        help='Path to annotation file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='Train split ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Ensure output_dir ends with / (required by SAM-VMNet dataloader)
    if not output_dir.endswith('/'):
        output_dir += '/'

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Find and load annotations
    print("\n=== Step 1: Loading annotations ===")

    annotations = None

    if args.annotation_file:
        print(f"Using provided annotation file: {args.annotation_file}")
        annotations = load_annotations(args.annotation_file)
    else:
        # Try auto-detection in order: JSON/NDJSON, CSV, XML
        ann_file = find_annotation_file(data_dir)
        if ann_file:
            print(f"Auto-detected annotation file: {ann_file}")
            try:
                annotations = load_annotations(ann_file)
            except ValueError as e:
                print(f"Warning: {e}")

        if not annotations:
            print("Trying CSV format...")
            annotations = try_labelbox_csv(data_dir)

        if not annotations:
            print("Trying Pascal VOC XML format...")
            annotations = try_xml_annotations(data_dir)

    if not annotations:
        print(
            "\nERROR: Could not find or parse annotations.\n"
            "The Mendeley dataset annotation format was not automatically recognized.\n\n"
            "Please inspect the dataset files and either:\n"
            "  1. Specify the annotation file with --annotation_file\n"
            "  2. Update this script to handle the specific format\n\n"
            "Common locations to check:\n"
            f"  ls {data_dir}/*.json\n"
            f"  ls {data_dir}/*.csv\n"
            f"  ls {data_dir}/*.xml\n"
            f"  ls {data_dir}/annotations/\n"
        )
        return

    print(f"Loaded {len(annotations)} annotated images")

    # Step 2: Find all images
    print("\n=== Step 2: Finding images ===")
    image_lookup = find_images(data_dir)
    print(f"Found {len(image_lookup)} image files")

    # Match check
    matched = sum(1 for a in annotations if a['image_filename'] in image_lookup)
    print(f"Matched {matched}/{len(annotations)} annotations to image files")

    if matched == 0:
        print("\nWARNING: No annotations matched to image files!")
        print("Sample annotation filenames:", [a['image_filename'] for a in annotations[:5]])
        print("Sample image filenames:", list(image_lookup.keys())[:5])
        print("\nYou may need to adjust the annotation parsing or file matching logic.")
        return

    # Step 3: Patient-level split
    print(f"\n=== Step 3: Splitting data ({args.train_ratio}/{args.val_ratio}/"
          f"{1 - args.train_ratio - args.val_ratio:.1f}) ===")

    train_anns, val_anns, test_anns = patient_level_split(
        annotations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Train: {len(train_anns)} images")
    print(f"Val:   {len(val_anns)} images")
    print(f"Test:  {len(test_anns)} images")

    # Step 4: Process and save
    print("\n=== Step 4: Processing and saving ===")

    for split_name, split_anns in [('train', train_anns), ('val', val_anns), ('test', test_anns)]:
        saved, skipped = process_and_save(split_anns, image_lookup, output_dir, split_name)
        print(f"  {split_name}: saved {saved}, skipped {skipped}")

    # Summary
    print("\n=== Done! ===")
    print(f"Dataset prepared at: {output_dir}")
    print(f"  {output_dir}train/images/  and  {output_dir}train/masks/")
    print(f"  {output_dir}val/images/    and  {output_dir}val/masks/")
    print(f"  {output_dir}test/images/   and  {output_dir}test/masks/")
    print(f"\nNext step: Run training with:")
    print(f"  python train_branch1.py --batch_size 8 --gpu_id '0' --epochs 200 "
          f"--work_dir './result_branch1/' --data_path '{output_dir}'")


if __name__ == '__main__':
    main()
