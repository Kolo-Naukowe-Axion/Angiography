#!/usr/bin/env python3
"""Generate Branch1 test predictions for Branch2 feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from configs.config_setting import setting_config
from dataset import Branch1_datasets
from models.vmunet.vmunet import VMUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export Branch1 predicted masks for test split')
    parser.add_argument('--data_path', type=str, required=True, help='Path to vessel dataset root')
    parser.add_argument('--pretrained_weight', type=str, required=True, help='Path to branch1 checkpoint (.pth)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g. cuda:0 or cpu)')
    parser.add_argument('--pred_masks_dir', type=str, default=None, help='Output folder for predicted masks')
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


def _load_model(pretrained_weight: Path, device: torch.device) -> VMUNet:
    model_cfg = setting_config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()

    checkpoint = torch.load(pretrained_weight, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _preprocess_image(image_path: Path):
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    dummy_mask = np.zeros((h, w, 1), dtype=np.uint8)
    image_tensor, _ = setting_config.test_transformer((image, dummy_mask))
    return image_tensor.unsqueeze(0), (h, w)


def main() -> int:
    args = parse_args()

    data_root = Path(args.data_path).resolve()
    test_images_dir = data_root / 'test' / 'images'
    pred_masks_dir = Path(args.pred_masks_dir).resolve() if args.pred_masks_dir else (data_root / 'test' / 'pred_masks')
    pred_masks_dir.mkdir(parents=True, exist_ok=True)

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory does not exist: {test_images_dir}")

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    if device.type == 'cuda' and device.index is not None:
        setting_config.gpu_id = str(device.index)
    model = _load_model(Path(args.pretrained_weight).resolve(), device)

    image_paths = sorted([p for p in test_images_dir.iterdir() if p.is_file()])
    if not image_paths:
        raise RuntimeError(f"No test images found in {test_images_dir}")

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc='branch1->pred_masks'):
            input_tensor, original_size = _preprocess_image(image_path)
            input_tensor = input_tensor.to(device, non_blocking=True).float()

            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]

            pred = output.squeeze().detach().cpu().numpy()
            pred = (pred >= args.threshold).astype(np.uint8) * 255

            mask = Image.fromarray(pred, mode='L')
            mask = mask.resize((original_size[1], original_size[0]), resample=Image.NEAREST)
            mask.save(pred_masks_dir / f"{image_path.stem}.png")

    print(f"Saved {len(image_paths)} predicted masks to {pred_masks_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
