#!/usr/bin/env python3
"""Benchmark inference: latency and memory (PyTorch). Usage: python scripts/benchmark_inference.py --checkpoint checkpoints/best.pt [--sizes 256 512] [--output results.json]"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.predict import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default="checkpoints/best.pt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512], help="Spatial sizes to benchmark (H=W)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", "-o", default=None, help="Write summary to JSON (performance under different conditions)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    results = {"device": str(device), "checkpoint": args.checkpoint, "runs": args.runs, "sizes": {}}
    for size in args.sizes:
        x = torch.rand(1, 1, size, size, device=device)
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            with torch.no_grad():
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        ms = (elapsed / args.runs) * 1000
        mem = "N/A"
        if device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated(device) / 2**20
            mem = f"{mem_mb:.1f} MB"
            results["sizes"][f"{size}x{size}"] = {"ms_per_image": round(ms, 2), "peak_gpu_mb": round(mem_mb, 1)}
        else:
            results["sizes"][f"{size}x{size}"] = {"ms_per_image": round(ms, 2)}
        print(f"  {size}x{size}: {ms:.1f} ms/image  peak GPU mem: {mem}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary written to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
