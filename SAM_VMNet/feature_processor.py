#!/usr/bin/env python3
"""Feature precompute for SAM-VMNet branch 2.

Unlike the original version, this implementation loads MedSAM once and reuses
it for all frames, which is required for practical throughput on cloud GPUs.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from med_sam.segment_anything.build_sam import sam_model_registry
from med_sam.segment_anything.predictor import SamPredictor


def _sample_prompt_points(mask_path: Path, max_points: int = 10) -> np.ndarray:
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise ValueError(f"Failed to load mask: {mask_path}")

    _, binary = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    y_coords, x_coords = np.where(binary == 255)

    if len(x_coords) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points = np.column_stack((x_coords, y_coords)).astype(np.float32)
    step = max(1, len(points) // max_points)
    sampled = points[::step][:max_points]
    return sampled


def _extract_feature(
    predictor: SamPredictor,
    image_path: Path,
    mask_path: Path,
) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    point_coords = _sample_prompt_points(mask_path)

    if point_coords.size == 0:
        # Fallback prompt in image center keeps feature extraction robust for
        # empty masks/predictions.
        h, w = image.shape[:2]
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)

    point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)
    predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    return predictor.Returnfeatures().detach().cpu()


def _iter_pairs(image_dir: Path, mask_dir: Path):
    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()])
    mask_by_stem = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

    for image_path in image_files:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        yield image_path, mask_path


def process_images(
    data_path: str,
    model_path: str,
    device: torch.device | str = "cuda:0",
    test_mask_subdir: str = "pred_masks",
) -> None:
    base = Path(data_path)
    if isinstance(device, torch.device):
        runtime_device = device
    else:
        runtime_device = torch.device(device if torch.cuda.is_available() else "cpu")

    medsam_model = sam_model_registry["vit_b"](checkpoint=model_path)
    medsam_model = medsam_model.to(runtime_device)
    medsam_model.eval()
    predictor = SamPredictor(medsam_model)

    splits = ["train", "val", "test"]

    for split in splits:
        image_dir = base / split / "images"
        mask_subdir = "masks" if split != "test" else test_mask_subdir
        mask_dir = base / split / mask_subdir
        output_dir = base / split / "feature"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(
                f"Missing required directories for split '{split}': image_dir={image_dir}, mask_dir={mask_dir}"
            )

        pairs = list(_iter_pairs(image_dir, mask_dir))
        if not pairs:
            raise RuntimeError(f"No image/mask pairs found for split '{split}' (images={image_dir}, masks={mask_dir})")

        print(f"Processing {split}: {len(pairs)} frames")
        for image_path, mask_path in tqdm(pairs, desc=f"feature:{split}"):
            output_file = output_dir / f"{image_path.stem}.pt"
            if output_file.exists():
                continue

            feature = _extract_feature(predictor, image_path, mask_path)
            torch.save(feature, output_file)
