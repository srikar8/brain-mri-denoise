"""ONNX inference: no PyTorch at runtime. Use for low-memory deployment (e.g. Render free tier)."""
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    _onnx_import_error = e
else:
    _onnx_import_error = None

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_model(checkpoint_path: str) -> Any:
    if ort is None:
        msg = "onnxruntime is required for ONNX inference. Install with: pip install onnxruntime"
        if _onnx_import_error is not None:
            raise RuntimeError(f"{msg} (import failed: {_onnx_import_error})") from _onnx_import_error
        raise RuntimeError(msg)
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {checkpoint_path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def denoise_image(session: Any, image: np.ndarray) -> np.ndarray:
    """Run ONNX model on a single image (H, W) or (1, H, W), float [0,1] or uint8. Returns denoised (H, W) uint8."""
    if image.dtype != np.float32:
        image = np.array(image, dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = image[np.newaxis, np.newaxis, ...]
    elif image.ndim == 3:
        image = image[np.newaxis, ...]
    out = session.run([OUTPUT_NAME], {INPUT_NAME: image})[0]
    out = out[0, 0]
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out
