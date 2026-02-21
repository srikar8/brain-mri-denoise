# Test Plan

## 1. Scope

Unit tests, integration-style tests, and one system-level smoke test for the Brain MRI denoising pipeline. Not a full IEC 62304 qualification; supports quality and regression.

## 2. Test Levels

### 2.1 Unit

| Test | File | Description |
|------|------|-------------|
| PSNR identical | test_metrics.py | PSNR(x,x) is high. |
| PSNR different | test_metrics.py | PSNR(zeros, ones) is low. |
| SSIM identical | test_metrics.py | SSIM(x,x) ≈ 1. |
| SSIM range | test_metrics.py | SSIM in [-1, 1]. |
| Gaussian noise | test_noise.py | Shape and value range after add_gaussian_noise. |
| Rician noise | test_noise.py | Shape and value range after add_rician_noise. |
| Noise randomness | test_noise.py | Different seeds give different noisy images. |
| U-Net forward | test_unet.py | Output shape matches input spatial dimensions. |
| U-Net output range | test_unet.py | Output values in reasonable range. |

**Run:** `pytest tests/ -v`

### 2.2 Integration

| Test | File | Description |
|------|------|-------------|
| Smoke inference | test_smoke_inference.py | Load minimal checkpoint, run inference on one image, check output shape and value range (0–255). |

**Run:** `pytest tests/test_smoke_inference.py -v`

### 2.3 System / Manual

- **CLI:** Run `python -m src.inference.predict -c checkpoints/best.pt -i <image> -o predictions`; verify output file exists and is valid image.
- **API:** Start server; POST image to /denoise; verify 200 and response is PNG.
- **Docker:** Build image, run with mounted checkpoint; call /denoise or CLI; verify output.

## 3. CI

- GitHub Actions: lint (ruff), full unit + smoke test suite on push/PR. See `.github/workflows/ci.yml`.

## 4. Acceptance

- All unit and smoke tests pass.
- No ruff violations in `src` and `scripts`.
- Smoke test confirms inference output shape and value range.
