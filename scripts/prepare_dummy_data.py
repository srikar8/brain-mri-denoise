#!/usr/bin/env python3
"""Create minimal dummy MRI-like images under data/brain_mri for testing training without external dataset."""
from pathlib import Path

import numpy as np
from PIL import Image

def main():
    root = Path(__file__).resolve().parent.parent / "data" / "brain_mri"
    root.mkdir(parents=True, exist_ok=True)
    (root / "Training" / "dummy").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(20):
        img = rng.uniform(0.2, 0.8, (128, 128))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(root / "Training" / "dummy" / f"slice_{i:02d}.jpg")
    print(f"Created 20 dummy images under {root}")


if __name__ == "__main__":
    main()
