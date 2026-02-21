#!/usr/bin/env python3
"""Create a noisy dataset from a folder (e.g. Testing): add synthetic noise and save. Preserves subfolder structure."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.noise import add_gaussian_noise, add_rician_noise


def main():
    parser = argparse.ArgumentParser(description="Create noisy images from a clean image folder")
    parser.add_argument("--input_dir", "-i", default="data/Testing", help="Root folder of clean images")
    parser.add_argument("--output_dir", "-o", default="data/Testing_noisy", help="Root folder for noisy images")
    parser.add_argument("--noise", choices=("gaussian", "rician"), default="gaussian")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("H", "W"), help="Resize to H W (optional)")
    parser.add_argument("--ext", default="jpg", help="Image extension to look for")
    args = parser.parse_args()

    root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    if not root.is_dir():
        print(f"Error: input dir not found: {root}")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = [args.ext.lower(), args.ext.upper()]
    paths = []
    for ext in exts:
        paths.extend(root.rglob(f"*.{ext}"))
    paths = sorted(set(paths))
    if not paths:
        print(f"No *.{args.ext} images under {root}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    for i, p in enumerate(paths):
        rel = p.relative_to(root)
        out_path = out_root / rel.parent
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / (rel.stem + "_noisy." + rel.suffix)

        img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        if args.size:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((args.size[1], args.size[0]), Image.BILINEAR)
            img = np.array(img_pil, dtype=np.float32) / 255.0
        if args.noise == "rician":
            noisy = add_rician_noise(img, args.sigma, rng)
        else:
            noisy = add_gaussian_noise(img, args.sigma, rng)
        out_arr = (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out_arr).save(out_file)
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  {i + 1}/{len(paths)}")
    print(f"Done. Wrote {len(paths)} noisy images to {out_root}")


if __name__ == "__main__":
    main()
