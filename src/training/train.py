"""Training loop: config-driven, fixed seed, validation with PSNR/SSIM."""
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from ..data.dataset import build_datasets
from ..models.unet import UNet
from ..evaluation.metrics import compute_psnr, compute_ssim
from ..utils import get_device


def _get_transform(config: Dict[str, Any]):
    from torchvision import transforms
    aug = config.get("augmentation", {})
    t = []
    if aug.get("horizontal_flip"):
        t.append(transforms.RandomHorizontalFlip())
    if aug.get("vertical_flip"):
        t.append(transforms.RandomVerticalFlip())
    if aug.get("rotation_degrees", 0) > 0:
        t.append(transforms.RandomRotation(aug["rotation_degrees"]))
    return transforms.Compose(t) if t else None


def train(config_path: str = "config/train_config.yaml", config_overrides: Dict[str, Any] = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if config_overrides:
        for k, v in config_overrides.items():
            if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                config[k].update(v)
            else:
                config[k] = v
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = get_device()
    print(f"Using device: {device}")
    data_cfg = config["data"]
    size = tuple(data_cfg["size"]) if data_cfg.get("size") else None
    train_ds, val_ds, test_ds = build_datasets(
        root=data_cfg.get("root", "."),
        image_ext=data_cfg.get("image_ext", "jpg"),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        noise_type=config["noise"]["type"],
        sigma=config["noise"]["sigma"],
        seed=seed,
        train_transform=_get_transform(config),
        train_dir=data_cfg.get("train_dir"),
        test_dir=data_cfg.get("test_dir"),
        size=size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model_cfg = config["model"]
    model = UNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 3),
    )
    model = model.to(device)
    loss_name = config["training"].get("loss", "l1")
    if device.type == "mps" and loss_name == "l1":
        criterion = nn.MSELoss()
        print("Using MSE loss on MPS (L1 backward uses unsupported sgn op).")
    else:
        criterion = nn.L1Loss() if loss_name == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_psnr = -1.0
    patience = config["training"].get("early_stopping_patience", 10)
    patience_counter = 0
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for noisy, clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        model.eval()
        val_psnr_sum, val_ssim_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_psnr_sum += compute_psnr(pred, clean)
                val_ssim_sum += compute_ssim(pred, clean)
                n_val += 1
        val_psnr = val_psnr_sum / max(n_val, 1)
        val_ssim = val_ssim_sum / max(n_val, 1)
        print(f"Epoch {epoch+1} loss={train_loss:.4f} val_PSNR={val_psnr:.2f} val_SSIM={val_ssim:.4f}")
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
                ckpt_dir / "best.pt",
            )
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    return str(ckpt_dir / "best.pt")
