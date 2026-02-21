"""Image quality metrics: PSNR and SSIM (predictions vs clean reference)."""
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _to_np(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        t = t[0]
    return t.detach().cpu().numpy()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB. pred/target: (B,C,H,W) or (C,H,W); assumed in [0, data_range]."""
    p = _to_np(pred)
    t = _to_np(target)
    return float(sk_psnr(t, p, data_range=data_range))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """SSIM. pred/target: (B,C,H,W) or (C,H,W)."""
    p = _to_np(pred)
    t = _to_np(target)
    if p.ndim == 3:
        p, t = p[np.newaxis], t[np.newaxis]
    return float(sk_ssim(t, p, data_range=data_range, channel_axis=0))
