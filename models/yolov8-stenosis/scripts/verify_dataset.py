"""
Verify YOLO-format dataset integrity.

Checks:
- Every image has a matching .txt label file (and vice versa)
- Label format: class x_center y_center width height (normalized 0-1)
- Image sizes
- Class distribution (stenosis vs empty)

Usage:
    python scripts/verify_dataset.py --images path/to/images --labels path/to/labels
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_label_file(label_path):
    """Parse a YOLO label file. Returns list of (class_id, x, y, w, h) tuples."""
    boxes = []
    with open(label_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path}:{line_num} — expected 5 values, got {len(parts)}: {line}")
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            for i, v in enumerate(coords):
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"{label_path}:{line_num} — coordinate {i} out of range [0,1]: {v}")
            boxes.append((cls, *coords))
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Verify YOLO dataset")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels directory")
    parser.add_argument("--check-sizes", action="store_true", help="Read image dimensions (slower)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    image_files = {f.stem: f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
    label_files = {f.stem: f for f in labels_dir.iterdir() if f.suffix == ".txt"}

    image_stems = set(image_files.keys())
    label_stems = set(label_files.keys())

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    matched = image_stems & label_stems

    print(f"Images directory:  {images_dir}")
    print(f"Labels directory:  {labels_dir}")
    print(f"Total images:      {len(image_files)}")
    print(f"Total label files: {len(label_files)}")
    print(f"Matched pairs:     {len(matched)}")
    print()

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images without labels:")
        for s in sorted(missing_labels)[:10]:
            print(f"  {image_files[s].name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
        print()

    if missing_images:
        print(f"WARNING: {len(missing_images)} labels without images:")
        for s in sorted(missing_images)[:10]:
            print(f"  {label_files[s].name}")
        if len(missing_images) > 10:
            print(f"  ... and {len(missing_images) - 10} more")
        print()

    total_boxes = 0
    class_counts = Counter()
    empty_labels = 0
    errors = []

    for stem in sorted(matched):
        try:
            boxes = parse_label_file(label_files[stem])
            if not boxes:
                empty_labels += 1
            total_boxes += len(boxes)
            for cls, *_ in boxes:
                class_counts[cls] += 1
        except ValueError as e:
            errors.append(str(e))

    with_stenosis = len(matched) - empty_labels
    pct = with_stenosis / len(matched) * 100 if matched else 0

    print(f"=== Label Statistics ===")
    print(f"Images with stenosis bbox: {with_stenosis} ({pct:.1f}%)")
    print(f"Images without bbox:       {empty_labels} ({100 - pct:.1f}%)")
    print(f"Total bounding boxes:      {total_boxes}")
    print(f"Class distribution:        {dict(class_counts)}")
    print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        print()

    if args.check_sizes:
        sizes = Counter()
        for stem in sorted(matched)[:500]:
            img = cv2.imread(str(image_files[stem]))
            if img is not None:
                h, w = img.shape[:2]
                sizes[(w, h)] += 1
        print(f"=== Image Sizes (sampled up to 500) ===")
        for (w, h), count in sizes.most_common(10):
            print(f"  {w}x{h}: {count} images")
        print()

    if not errors and not missing_labels and not missing_images:
        print("Dataset verification PASSED.")
    else:
        print("Dataset verification completed with WARNINGS — review issues above.")


if __name__ == "__main__":
    main()
