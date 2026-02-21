# Lazy import so ONNX-only deployments (no torch) can load api without importing predict.py
__all__ = ["run_inference"]


def __getattr__(name: str):
    if name == "run_inference":
        from .predict import run_inference
        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
