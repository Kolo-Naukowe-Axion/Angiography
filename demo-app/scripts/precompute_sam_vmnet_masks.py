#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute SAM-VMNet prediction masks for demo-app ARCADE patients.")
    parser.add_argument("--data-root", type=Path, default=Path("demo-app/data/patients"))
    parser.add_argument("--model-id", type=str, default="sam_vmnet_arcade")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("SAM_VMNet/pre_trained_weights/best-epoch142-loss0.3230.pth"),
    )
    parser.add_argument("--dataset-id", type=str, default="arcade")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--patient-id", action="append", dest="patient_ids")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(torch_module, requested: str):
    if requested == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda:0")
        return torch_module.device("cpu")
    if requested.startswith("cuda") and torch_module.cuda.is_available():
        return torch_module.device(requested)
    return torch_module.device("cpu")


def load_model(repo_root: Path, checkpoint_path: Path, requested_device: str):
    sam_repo = (repo_root / "SAM_VMNet").resolve()
    if not sam_repo.exists():
        raise FileNotFoundError(f"SAM_VMNet directory not found at {sam_repo}")

    sys.path.insert(0, str(sam_repo))

    try:
        import torch
        from configs.config_setting import setting_config
        from models.vmunet.vmunet import VMUNet
    except Exception as error:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "Failed to import SAM_VMNet runtime dependencies. "
            "Install SAM_VMNet requirements in the environment used to run this script."
        ) from error

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(torch, requested_device)
    if device.type == "cuda":
        setting_config.gpu_id = str(device.index or 0)

    model_cfg = setting_config.model_config
    model = VMUNet(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        depths=model_cfg["depths"],
        depths_decoder=model_cfg["depths_decoder"],
        drop_path_rate=model_cfg["drop_path_rate"],
        load_ckpt_path=model_cfg["load_ckpt_path"],
    )
    model.load_from()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    filtered_state_dict = {
        key: value
        for key, value in checkpoint.items()
        if "total_ops" not in key and "total_params" not in key
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    return model, setting_config


def infer_binary_mask(model, setting_config, frame_path: Path, threshold: float) -> Image.Image:
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)
    height, width = image_np.shape[:2]

    dummy_mask = np.zeros((height, width, 1), dtype=np.float32)
    image_tensor, _ = setting_config.test_transformer((image_np, dummy_mask))

    with np.errstate(all="ignore"):
        import torch

        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0).float())
            if isinstance(output, tuple):
                output = output[0]
            prediction = output.squeeze().detach().cpu().numpy()

    prediction_binary = (prediction >= threshold).astype(np.uint8) * 255
    prediction_image = Image.fromarray(prediction_binary, mode="L").resize((width, height), Image.Resampling.NEAREST)
    return prediction_image


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "manifest.json"
    checkpoint_path = args.checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[2]

    manifest = load_manifest(manifest_path)
    patients = manifest.get("patients", []) if isinstance(manifest.get("patients", []), list) else []

    selected_patients = []
    for patient in patients:
        if patient.get("datasetId", "mendeley") != args.dataset_id:
            continue
        if patient.get("labelType", "bbox") != "mask":
            continue
        if args.patient_ids and patient.get("id") not in set(args.patient_ids):
            continue
        prediction_masks = patient.get("predictionMasks", {})
        if args.model_id not in prediction_masks:
            continue
        selected_patients.append(patient)

    if not selected_patients:
        print("No eligible patients found for precompute.")
        return

    print(f"Selected {len(selected_patients)} patient(s) for model '{args.model_id}'")

    if args.dry_run:
        for patient in selected_patients:
            print(f"[dry-run] {patient['id']}")
        return

    model, setting_config = load_model(repo_root=repo_root, checkpoint_path=checkpoint_path, requested_device=args.device)

    written = 0
    skipped = 0

    for patient in selected_patients:
        patient_id = patient["id"]
        frames_dir = (data_root / patient["framesDir"]).resolve()
        prediction_dir = (data_root / patient["predictionMasks"][args.model_id]).resolve()
        prediction_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(
            [path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}],
            key=lambda path: path.name,
        )

        print(f"Processing {patient_id}: {len(frame_paths)} frame(s)")

        for frame_path in frame_paths:
            target_mask = prediction_dir / f"{frame_path.stem}.png"
            if target_mask.exists() and not args.overwrite:
                skipped += 1
                continue

            mask_image = infer_binary_mask(model=model, setting_config=setting_config, frame_path=frame_path, threshold=args.threshold)
            mask_image.save(target_mask)
            written += 1

    print(f"Done. Masks written: {written}, skipped: {skipped}")


if __name__ == "__main__":
    main()
