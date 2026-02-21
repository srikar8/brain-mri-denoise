# Changelog

All notable changes to this project are documented here. Modification log for code is maintained in version control (git history).

## [0.2.0] – 2025-02-20

### Added

- FastAPI inference API with POST /denoise and upload UI (noisy vs denoised side by side).
- Docker default: containerized inference server (uvicorn) with DENOISE_CHECKPOINT and volume mount.
- Script to create noisy dataset from a folder: `scripts/create_noisy_dataset.py`.
- MPS (Apple Silicon) support in training and inference; automatic MSE on MPS when L1 is selected.
- Design doc (`docs/DESIGN.md`), test plan (`docs/TEST_PLAN.md`), traceability matrix (`docs/TRACEABILITY.md`), risk analysis (`docs/RISK_ANALYSIS.md`).
- Inference benchmark script: `scripts/benchmark_inference.py`.
- CI: Docker image build job.
- README: data/privacy, coding standards, deployment, benchmark, reproducibility.

### Changed

- U-Net decoder fix for correct channel counts; skip connection resized with interpolate for variable input sizes.
- Dataset: optional train_dir/test_dir and fixed image size for batching.
- ONNX export: always interpolate skip (removes tracer warning).

### Fixed

- python-multipart added for FastAPI file upload in Docker.
- SSIM test: allow [-1, 1] range.

## [0.1.0] – Initial

- U-Net denoising, config-driven training, PSNR/SSIM evaluation.
- CLI and Docker CLI inference; ONNX export.
- GitHub Actions: lint and pytest smoke test.
