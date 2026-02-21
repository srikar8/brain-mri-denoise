# Changelog

All notable changes to this project are documented here. Modification log for code is maintained in version control (git history).

## [1.0.0] – 2026-02-21

Initial release: end-to-end brain MRI denoising with U-Net.

### Features

- U-Net model (configurable depth and channels) for grayscale MRI denoising.
- Synthetic noise (Gaussian/Rician) applied on-the-fly during training.
- Config-driven training with reproducible seed, validation PSNR/SSIM, and early stopping.
- CLI and FastAPI inference (POST /denoise with upload UI).
- ONNX export for lightweight deployment.
- Docker container with uvicorn API server.
- CI: lint (ruff), unit tests (pytest), smoke test, Docker build.
- QMS-oriented docs: design (IEC 62304), risk analysis (ISO 14971), traceability, test plan.
- Benchmark script with optional JSON output (performance under different conditions).
- Deployed demo on Render.
