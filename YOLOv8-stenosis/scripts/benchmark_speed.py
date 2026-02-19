"""
Benchmark inference speed for the stenosis detection model.

Tests PyTorch (.pt) and ONNX (.onnx) on GPU and CPU.

Usage:
    python scripts/benchmark_speed.py
    python scripts/benchmark_speed.py --weights weights/best.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def benchmark(model, imgsz, n_warmup=10, n_runs=100):
    """Run inference benchmark and return average ms/frame."""
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(n_warmup):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000 / np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--weights", type=str, default="weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--export-onnx", action="store_true", help="Also export and benchmark ONNX")
    parser.add_argument("--output", type=str, default="results/benchmark.json")
    args = parser.parse_args()

    results = {}

    # PyTorch GPU
    print("=== PyTorch GPU ===")
    try:
        model_gpu = YOLO(args.weights)
        model_gpu.to("cuda")
        r = benchmark(model_gpu, args.imgsz, n_runs=args.runs)
        results["pytorch_gpu"] = r
        print(f"  {r['mean_ms']:.1f} ms/frame ({r['fps']:.1f} FPS)")
    except Exception as e:
        print(f"  Skipped (no GPU): {e}")

    # PyTorch CPU
    print("=== PyTorch CPU ===")
    model_cpu = YOLO(args.weights)
    model_cpu.to("cpu")
    r = benchmark(model_cpu, args.imgsz, n_runs=args.runs)
    results["pytorch_cpu"] = r
    print(f"  {r['mean_ms']:.1f} ms/frame ({r['fps']:.1f} FPS)")

    # ONNX
    if args.export_onnx:
        print("=== Exporting ONNX ===")
        model_export = YOLO(args.weights)
        onnx_path = model_export.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"  Exported to {onnx_path}")

        print("=== ONNX CPU ===")
        model_onnx = YOLO(onnx_path)
        r = benchmark(model_onnx, args.imgsz, n_runs=args.runs)
        results["onnx_cpu"] = r
        print(f"  {r['mean_ms']:.1f} ms/frame ({r['fps']:.1f} FPS)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark saved to {output_path}")


if __name__ == "__main__":
    main()
