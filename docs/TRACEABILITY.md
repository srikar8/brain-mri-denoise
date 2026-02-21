# Requirements Traceability Matrix

| Req ID | Requirement | Design Element | Test / Verification |
|--------|-------------|----------------|---------------------|
| REQ-1 | System shall accept brain MRI images and produce denoised images. | U-Net model; inference in `predict.py` and `api.py`. | test_smoke_inference; manual CLI/API run. |
| REQ-2 | Training shall use configurable synthetic noise (Gaussian or Rician). | `src/data/noise.py`; `config/train_config.yaml` (noise.type, sigma). | test_noise.py (Gaussian, Rician). |
| REQ-3 | Training and evaluation shall be reproducible (fixed seed, single config). | `train_config.yaml` (seed, paths, hyperparams); torch.manual_seed in train.py. | Reproducibility statement in README; same config + seed yields same split. |
| REQ-4 | System shall expose inference via CLI (input path → output path). | `src/inference/predict.py` (argparse, run_inference). | test_smoke_inference; manual CLI. |
| REQ-5 | System shall expose inference via HTTP API (upload → denoised image). | `src/inference/api.py` (POST /denoise). | Manual or automated POST to /denoise. |
| REQ-6 | Denoised output shall be valid image (correct shape, value range). | denoise_image / run_inference clip to [0,255], save as PNG. | test_smoke_inference (shape, 0–255). |
| REQ-7 | Quality shall be measurable (PSNR, SSIM vs clean reference). | `src/evaluation/metrics.py`; used in train and run_evaluate. | test_metrics.py; run_evaluate on test set. |
