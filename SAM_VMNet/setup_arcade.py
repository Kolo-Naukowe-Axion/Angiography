"""
Convert ARCADE syntax dataset (COCO polygon annotations) to binary mask PNGs
for SAM-VMNet's expected ./data/vessel/ structure.
"""

import json
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

ARCADE_DIR = "arcade/syntax"
OUTPUT_DIR = "data/vessel"
SPLITS = ["train", "val", "test"]


def coco_to_binary_masks(annotation_path, image_dir, out_images_dir, out_masks_dir):
    with open(annotation_path) as f:
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
        mask.save(os.path.join(out_masks_dir, filename))

        # Copy image
        src = os.path.join(image_dir, filename)
        dst = os.path.join(out_images_dir, filename)
        shutil.copy2(src, dst)

    return len(images)


def main():
    for split in SPLITS:
        out_images = os.path.join(OUTPUT_DIR, split, "images")
        out_masks = os.path.join(OUTPUT_DIR, split, "masks")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_masks, exist_ok=True)

        ann_path = os.path.join(ARCADE_DIR, split, "annotations", f"{split}.json")
        img_dir = os.path.join(ARCADE_DIR, split, "images")

        n = coco_to_binary_masks(ann_path, img_dir, out_images, out_masks)
        print(f"{split}: converted {n} images")

    print(f"\nDone. Dataset ready at ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
