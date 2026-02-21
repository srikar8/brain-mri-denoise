#!/usr/bin/env python3
"""Run FastAPI inference server. Usage: DENOISE_CHECKPOINT=checkpoints/best.pt python scripts/run_api.py [--port 8000]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(
        "src.inference.api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
