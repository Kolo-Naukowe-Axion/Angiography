#!/usr/bin/env python3
"""
SAM-VMNet Local Inference
Runs on CPU or MPS (Apple Silicon) — no CUDA required.

Usage:
    # Branch 1 only (simpler, faster):
    python inference/predict.py --image path/to/angiogram.png

    # Branch 2 / SAM-VMNet (better quality, slower):
    python inference/predict.py --image path/to/angiogram.png --branch 2

    # Batch process a directory:
    python inference/predict.py --image_dir path/to/images/ --output_dir results/
"""

import sys
import os

# Insert this directory FIRST so our CPU-compatible mamba_ssm is found
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
from PIL import Image
import argparse
import time

# Now safe to import model code (will find our fake mamba_ssm)
from models.vmunet.vmunet import VMUNet
from models.vmunet.samvmnet import SAMVMNet
from configs.config_setting import setting_config


def get_device():
    """Pick best available device: MPS (Apple Silicon) > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_branch1_model(weights_path, device):
    """Load Branch 1 (VM-UNet) model."""
    cfg = setting_config.model_config
    model = VMUNet(
        num_classes=cfg['num_classes'],
        input_channels=cfg['input_channels'],
        depths=cfg['depths'],
        depths_decoder=cfg['depths_decoder'],
        drop_path_rate=cfg['drop_path_rate'],
        load_ckpt_path=None,  # not needed — we load trained weights directly
    )

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    # Filter out thop profiling keys if present
    state_dict = {k: v for k, v in state_dict.items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.device = device  # override hardcoded CUDA device for MPS/CPU
    model.eval()
    return model


def load_branch2_model(weights_path, device):
    """Load Branch 2 (SAM-VMNet) model."""
    cfg = setting_config.model_config
    model = SAMVMNet(
        num_classes=cfg['num_classes'],
        input_channels=cfg['input_channels'],
        depths=cfg['depths'],
        depths_decoder=cfg['depths_decoder'],
        drop_path_rate=cfg['drop_path_rate'],
        load_ckpt_path=None,  # not needed — we load trained weights directly
    )

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = {k: v for k, v in state_dict.items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.device = device  # override hardcoded CUDA device for MPS/CPU
    model.eval()
    return model


def load_medsam(medsam_path, device):
    """Load MedSAM model for feature extraction."""
    med_sam_dir = os.path.join(PROJECT_DIR, 'med_sam')
    sys.path.insert(0, med_sam_dir)
    from segment_anything.build_sam import sam_model_registry
    from segment_anything.predictor import SamPredictor

    medsam_model = sam_model_registry["vit_b"](checkpoint=medsam_path)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()
    predictor = SamPredictor(medsam_model)
    return predictor


def preprocess_image(image_path):
    """Load and preprocess image to model input tensor."""
    img = np.array(Image.open(image_path).convert('RGB'))

    # Normalize (same stats as training)
    mean, std = 157.561, 26.706
    img_norm = (img.astype(np.float32) - mean) / std
    img_norm = ((img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())) * 255.0

    # To tensor: (H, W, 3) -> (1, 3, H, W)
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    # Resize to 256x256
    tensor = torch.nn.functional.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return tensor


def extract_medsam_features(image_path, mask_np, predictor):
    """Extract MedSAM features using point prompts from mask."""
    import cv2

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # Sample points from mask (white regions)
    y_coords, x_coords = np.where(mask_np > 127)
    if len(x_coords) == 0:
        # No detections — use center point as fallback
        h, w = mask_np.shape
        points = np.array([[w // 2, h // 2]])
    else:
        all_points = np.column_stack((x_coords, y_coords))
        n_points = min(10, len(all_points))
        skip = max(1, len(all_points) // n_points)
        points = all_points[::skip][:n_points]

    labels = np.ones(len(points))
    predictor.predict(point_coords=points, point_labels=labels, multimask_output=False)
    features = predictor.Returnfeatures()
    return features


def predict_single(image_path, branch1_model, branch2_model, medsam_predictor, device, threshold=0.5):
    """Run full inference pipeline on a single image."""
    # Step 1: Preprocess
    img_tensor = preprocess_image(image_path).to(device)

    # Step 2: Branch 1 prediction
    with torch.no_grad():
        branch1_out = branch1_model(img_tensor)
    branch1_mask = (branch1_out.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

    if branch2_model is None:
        return branch1_mask, None

    # Step 3: Extract MedSAM features using Branch 1's predicted mask
    features = extract_medsam_features(image_path, branch1_mask, medsam_predictor)
    features = features.to(device)

    # Step 4: Branch 2 prediction
    with torch.no_grad():
        branch2_out = branch2_model(img_tensor, features)
    branch2_mask = (branch2_out.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

    return branch1_mask, branch2_mask


def main():
    parser = argparse.ArgumentParser(description='SAM-VMNet Local Inference')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Output directory for masks')
    parser.add_argument('--branch', type=int, default=1, choices=[1, 2],
                        help='Which branch to use (1=VM-UNet only, 2=SAM-VMNet)')
    parser.add_argument('--branch1_weights', type=str,
                        default=os.path.join(PROJECT_DIR, 'saved/branch1/checkpoints/best-epoch90-loss0.7782.pth'))
    parser.add_argument('--branch2_weights', type=str,
                        default=os.path.join(PROJECT_DIR, 'saved/branch2/checkpoints/best-epoch77-loss0.7787.pth'))
    parser.add_argument('--medsam_weights', type=str,
                        default=os.path.join(PROJECT_DIR, 'saved/medsam_vit_b.pth'))
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image_dir")

    device = get_device()
    print(f"Device: {device}")

    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_paths = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
    print(f"Images to process: {len(image_paths)}")

    # Load models
    print("Loading Branch 1 model...")
    t0 = time.time()
    branch1_model = load_branch1_model(args.branch1_weights, device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    branch2_model = None
    medsam_predictor = None
    if args.branch == 2:
        print("Loading Branch 2 model...")
        t0 = time.time()
        branch2_model = load_branch2_model(args.branch2_weights, device)
        print(f"  Loaded in {time.time() - t0:.1f}s")

        print("Loading MedSAM...")
        t0 = time.time()
        medsam_predictor = load_medsam(args.medsam_weights, device)
        print(f"  Loaded in {time.time() - t0:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process
    for i, img_path in enumerate(image_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{i+1}/{len(image_paths)}] {name}...", end=" ", flush=True)
        t0 = time.time()

        branch1_mask, branch2_mask = predict_single(
            img_path, branch1_model, branch2_model, medsam_predictor,
            device, args.threshold
        )

        # Save Branch 1 output
        out_path = os.path.join(args.output_dir, f"{name}_branch1.png")
        Image.fromarray(branch1_mask).save(out_path)

        # Save Branch 2 output if available
        if branch2_mask is not None:
            out_path = os.path.join(args.output_dir, f"{name}_branch2.png")
            Image.fromarray(branch2_mask).save(out_path)

        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

    print(f"\nDone! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
