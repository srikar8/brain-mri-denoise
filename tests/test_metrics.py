import torch
import pytest
from src.evaluation.metrics import compute_psnr, compute_ssim


def test_psnr_identical():
    x = torch.rand(1, 1, 32, 32)
    assert compute_psnr(x, x) > 40.0


def test_psnr_different():
    a = torch.zeros(1, 1, 32, 32)
    b = torch.ones(1, 1, 32, 32)
    assert compute_psnr(a, b) < 20.0


def test_ssim_identical():
    x = torch.rand(1, 1, 32, 32)
    assert abs(compute_ssim(x, x) - 1.0) < 1e-5


def test_ssim_range():
    a = torch.rand(1, 1, 32, 32)
    b = torch.rand(1, 1, 32, 32)
    s = compute_ssim(a, b)
    assert -1 <= s <= 1  # SSIM can be slightly negative for very different images
