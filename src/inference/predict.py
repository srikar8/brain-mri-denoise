"""CLI inference: input image path(s) -> output denoised path(s)."""
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image

from ..models.unet import UNet
from ..utils import get_device


def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})
    model = UNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    if device is None:
        device = get_device()
    model = model.to(device).eval()
    return model


def denoise_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run model on a single image (H, W) or (1, H, W), float [0,1] or uint8. Returns denoised (H, W) uint8."""
    if device is None:
        device = next(model.parameters()).device
    if image.dtype != np.float32:
        image = np.array(image, dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = image[np.newaxis, np.newaxis, ...]
    elif image.ndim == 3:
        image = image[np.newaxis, ...]
    x = torch.from_numpy(image).float().to(device)
    with torch.no_grad():
        pred = model(x)
    out = pred[0, 0].cpu().numpy()
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out


def run_inference(
    model: torch.nn.Module,
    image_paths: List[str],
    output_dir: str,
    device: Optional[torch.device] = None,
) -> List[str]:
    if device is None:
        device = next(model.parameters()).device
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for path in image_paths:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        pred_np = denoise_image(model, img, device)
        name = Path(path).stem + "_denoised.png"
        out_file = out_path / name
        Image.fromarray(pred_np).save(out_file)
        out_paths.append(str(out_file))
    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Denoise brain MRI image(s)")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input image path(s)")
    parser.add_argument("--output", "-o", default="./predictions", help="Output directory")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"), help="Device (default: auto, uses MPS on Apple Silicon)")
    args = parser.parse_args()
    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    out_paths = run_inference(model, args.input, args.output, device)
    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
