#!/usr/bin/env python3
"""Run training from config. Usage: python scripts/run_training.py [config_path]"""
import sys
from pathlib import Path

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_config.yaml"
    train(config_path)
