"""Shared utilities."""
from typing import Optional
import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return best available device: MPS (Apple) > CUDA > CPU. prefer can override (e.g. 'cpu')."""
    if prefer:
        p = prefer.lower()
        if p == "cpu":
            return torch.device("cpu")
        if p == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if p == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
