#!/usr/bin/env python3
"""Evaluate checkpoint on test set: PSNR/SSIM. Usage: python scripts/run_evaluate.py --checkpoint <path> --data_root <path> [--config config/train_config.yaml]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import build_datasets
from src.inference.predict import load_model
from src.evaluation.metrics import compute_psnr, compute_ssim
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--data_root", default=None, help="Fallback if config has no root/train_dir/test_dir")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--output_dir", default="./predictions")
    args = parser.parse_args()
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = config.get("seed", 42)
    data_cfg = config["data"]
    root = data_cfg.get("root") or args.data_root or "."
    size = tuple(data_cfg["size"]) if data_cfg.get("size") else None
    _, _, test_ds = build_datasets(
        root=root,
        image_ext=data_cfg.get("image_ext", "jpg"),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        noise_type=config["noise"]["type"],
        sigma=config["noise"]["sigma"],
        seed=seed,
        train_dir=data_cfg.get("train_dir"),
        test_dir=data_cfg.get("test_dir"),
        size=size,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    device = get_device()
    model = load_model(args.checkpoint, device)
    psnr_list, ssim_list = [], []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        with torch.no_grad():
            pred = model(noisy)
        psnr_list.append(compute_psnr(pred, clean))
        ssim_list.append(compute_ssim(pred, clean))
    mean_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    mean_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0
    print(f"Test PSNR: {mean_psnr:.2f} ± {float(torch.tensor(psnr_list).std()):.2f}")
    print(f"Test SSIM: {mean_ssim:.4f} ± {float(torch.tensor(ssim_list).std()):.4f}")


if __name__ == "__main__":
    main()
