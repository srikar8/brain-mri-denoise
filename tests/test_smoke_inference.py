"""Smoke test: run inference on one sample; check output shape and value range."""
import numpy as np
from pathlib import Path
from PIL import Image

from src.inference.predict import load_model, run_inference


def test_smoke_inference(tmp_path):
    # Create minimal checkpoint
    import torch
    from src.models.unet import UNet
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    model = UNet(1, 1, 32, 3)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model": {"in_channels": 1, "out_channels": 1, "base_channels": 32, "depth": 3}},
    }, ckpt_dir / "smoke.pt")
    # Dummy input
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(img).save(in_dir / "sample.png")
    # Run inference
    loaded = load_model(str(ckpt_dir / "smoke.pt"))
    out_paths = run_inference(loaded, [str(in_dir / "sample.png")], str(out_dir))
    assert len(out_paths) == 1
    out = np.array(Image.open(out_paths[0]))
    assert out.shape == (64, 64)
    assert out.min() >= 0 and out.max() <= 255
