#!/usr/bin/env python3
"""Export trained PyTorch checkpoint to ONNX. Usage: python scripts/export_onnx.py --checkpoint <path> --output model.onnx"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.inference.predict import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--output", "-o", default="model.onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    model = load_model(args.checkpoint, torch.device("cpu"))
    dummy = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "H", 3: "W"}, "output": {0: "batch", 2: "H", 3: "W"}},
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
