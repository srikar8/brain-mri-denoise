import torch
import pytest
from src.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(1, 1, 32, 3)
    x = torch.rand(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_unet_output_range():
    model = UNet(1, 1, 16, 2)
    x = torch.rand(1, 1, 32, 32)
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.min() >= -10 and y.max() <= 10  # raw logits, not necessarily [0,1]
