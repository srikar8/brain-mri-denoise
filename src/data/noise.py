"""Synthetic noise for MRI: Gaussian and Rician (MRI-realistic magnitude noise)."""
import numpy as np


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise. Image assumed in [0, 1] or normalized."""
    noise = rng.normal(0, sigma, image.shape).astype(image.dtype)
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0)


def add_rician_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Rician noise for magnitude MRI: sqrt((A+n1)^2 + n2^2) with n1,n2 ~ N(0, sigma)."""
    n1 = rng.normal(0, sigma, image.shape).astype(image.dtype)
    n2 = rng.normal(0, sigma, image.shape).astype(image.dtype)
    magnitude = np.sqrt((image + n1) ** 2 + n2 ** 2)
    return np.clip(magnitude, 0.0, 1.0)
