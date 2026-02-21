import numpy as np
import pytest
from src.data.noise import add_gaussian_noise, add_rician_noise


def test_gaussian_noise_shape():
    rng = np.random.default_rng(42)
    x = np.ones((32, 32), dtype=np.float32) * 0.5
    y = add_gaussian_noise(x, 0.1, rng)
    assert y.shape == x.shape
    assert 0 <= y.min() <= 1 and 0 <= y.max() <= 1


def test_rician_noise_shape():
    rng = np.random.default_rng(42)
    x = np.ones((32, 32), dtype=np.float32) * 0.5
    y = add_rician_noise(x, 0.1, rng)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1


def test_gaussian_different_with_seed():
    x = np.ones((16, 16), dtype=np.float32) * 0.5
    y1 = add_gaussian_noise(x, 0.2, np.random.default_rng(1))
    y2 = add_gaussian_noise(x, 0.2, np.random.default_rng(2))
    assert not np.allclose(y1, y2)
