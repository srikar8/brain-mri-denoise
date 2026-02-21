"""Brain MRI dataset with synthetic noise for denoising (noisy -> clean pairs)."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .noise import add_gaussian_noise, add_rician_noise


def _collect_image_paths(root: str, ext: str) -> list:
    root = Path(root)
    paths = []
    for e in (ext.lower(), ext.upper(), ext):
        paths.extend(root.rglob(f"*.{e}"))
    return [str(p) for p in sorted(set(paths))]


class BrainMRIDenoisingDataset(Dataset):
    """Clean MRI as target; synthetic noise applied on-the-fly for input."""

    def __init__(
        self,
        image_paths: list,
        noise_type: str = "gaussian",
        sigma: float = 0.1,
        seed: Optional[int] = None,
        transform=None,
        size: Optional[Tuple[int, int]] = None,
    ):
        self.image_paths = image_paths
        self.noise_type = noise_type.lower()
        self.sigma = sigma
        self.transform = transform
        self.size = size  # (H, W) to resize to; None = keep original (batch size 1 if variable)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_normalize(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        if self.size is not None:
            img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        if img.ndim == 2:
            img = img[np.newaxis, ...]  # (1, H, W)
        else:
            img = np.transpose(img, (2, 0, 1))
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        clean = self._load_and_normalize(path)
        if self.noise_type == "rician":
            noisy = add_rician_noise(clean, self.sigma, self.rng)
        else:
            noisy = add_gaussian_noise(clean, self.sigma, self.rng)
        clean_t = torch.from_numpy(clean).float()
        noisy_t = torch.from_numpy(noisy).float()
        if self.transform:
            # Apply same transform to both for paired aug (e.g. flips/rotation)
            seed = self.rng.integers(0, 2 ** 31)
            torch.manual_seed(seed)
            noisy_t = self.transform(noisy_t)
            torch.manual_seed(seed)
            clean_t = self.transform(clean_t)
        return noisy_t, clean_t


def build_datasets(
    root: str,
    image_ext: str = "jpg",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    noise_type: str = "gaussian",
    sigma: float = 0.1,
    seed: int = 42,
    train_transform=None,
    train_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    size: Optional[Tuple[int, int]] = None,
):
    """Build train/val/test datasets. If train_dir and test_dir are set, use them
    (train_dir for train+val split by val_ratio, test_dir for test). Otherwise
    collect all from root and split by val_ratio and test_ratio."""
    if train_dir and test_dir:
        train_val_paths = _collect_image_paths(train_dir, image_ext)
        test_paths = _collect_image_paths(test_dir, image_ext)
        if not train_val_paths:
            raise FileNotFoundError(f"No images with ext '{image_ext}' under {train_dir}")
        if not test_paths:
            raise FileNotFoundError(f"No images with ext '{image_ext}' under {test_dir}")
        rng = np.random.default_rng(seed)
        rng.shuffle(train_val_paths)
        n_val = max(1, int(len(train_val_paths) * val_ratio))
        train_paths = train_val_paths[n_val:]
        val_paths = train_val_paths[:n_val]
    else:
        paths = _collect_image_paths(root, image_ext)
        if not paths:
            raise FileNotFoundError(f"No images with ext '{image_ext}' under {root}")
        rng = np.random.default_rng(seed)
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        train_paths = paths[:n_train]
        val_paths = paths[n_train : n_train + n_val]
        test_paths = paths[n_train + n_val :]
    train_ds = BrainMRIDenoisingDataset(
        train_paths, noise_type=noise_type, sigma=sigma, seed=seed, transform=train_transform, size=size
    )
    val_ds = BrainMRIDenoisingDataset(
        val_paths, noise_type=noise_type, sigma=sigma, seed=seed + 1, size=size
    )
    test_ds = BrainMRIDenoisingDataset(
        test_paths, noise_type=noise_type, sigma=sigma, seed=seed + 2, size=size
    )
    return train_ds, val_ds, test_ds
