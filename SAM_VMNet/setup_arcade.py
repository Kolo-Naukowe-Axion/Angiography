"""
Convert ARCADE syntax dataset (COCO polygon annotations) to binary mask PNGs
for SAM-VMNet's expected ../datasets/arcade/data/vessel/ structure.
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCADE_DIR = REPO_ROOT / "datasets" / "arcade" / "data" / "syntax"
OUTPUT_DIR = REPO_ROOT / "datasets" / "arcade" / "data" / "vessel"
SPLITS = ["train", "val", "test"]


def coco_to_binary_masks(annotation_path, image_dir, out_images_dir, out_masks_dir):
    with annotation_path.open() as f:
        coco = json.load(f)

    # Group annotations by image_id
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Build image_id -> image info mapping
    images = {img["id"]: img for img in coco["images"]}

    for img_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        filename = img_info["file_name"]

        # Create binary mask (all vessel categories merged)
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for ann in anns_by_image.get(img_id, []):
            for seg in ann["segmentation"]:
                # COCO segmentation: flat list [x1, y1, x2, y2, ...]
                if len(seg) < 6:
                    continue
                polygon = list(zip(seg[0::2], seg[1::2]))
                draw.polygon(polygon, fill=255)

        # Save mask
        mask.save(out_masks_dir / filename)

        # Copy image
        src = image_dir / filename
        dst = out_images_dir / filename
        shutil.copy2(src, dst)

    return len(images)


def main():
    for split in SPLITS:
        out_images = OUTPUT_DIR / split / "images"
        out_masks = OUTPUT_DIR / split / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        ann_path = ARCADE_DIR / split / "annotations" / f"{split}.json"
        img_dir = ARCADE_DIR / split / "images"

        n = coco_to_binary_masks(ann_path, img_dir, out_images, out_masks)
        print(f"{split}: converted {n} images")

    print(f"\nDone. Dataset ready at {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
