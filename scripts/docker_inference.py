#!/usr/bin/env python3
"""Entrypoint for Docker: run inference with argv from docker run."""
from src.inference.predict import main

if __name__ == "__main__":
    main()
