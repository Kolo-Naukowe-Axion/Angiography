"""
Train YOLOv8-M for coronary artery stenosis detection.

Usage:
    python train.py
    python train.py --batch 64 --imgsz 512 --device 0
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-M on stenosis dataset")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None, help="Path to last.pt to resume training")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        model = YOLO("yolov8m.pt")
        model.train(
            data="dataset/data.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            degrees=15,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            device=args.device,
            project="runs/stenosis",
            name="yolov8m_v1",
        )


if __name__ == "__main__":
    main()
