# Brain MRI Denoising with Deep Learning

Deep learning denoising for brain MRI: U-Net trained on synthetic noise (Gaussian/Rician), with reproducible config, Docker inference, and CI.

**Disclaimer:** This project is for research, education, and portfolio use. It is not a medical device and is not intended for clinical use. No FDA clearance or regulatory compliance is claimed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data

Use a brain tumor MRI dataset (e.g. [Brain Tumor MRI from Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset-29-categories) or similar). Place images under `data/brain_mri/` or use `train_dir`/`test_dir` in config (e.g. `data/Training`, `data/Testing`). The code discovers images recursively.

- Images are treated as **clean** references.
- **Synthetic noise** is added on-the-fly during training (no separate noisy/clean pairs).

### Data and privacy

- This repo uses **no real patient data or PHI**. Training uses only synthetic noise applied to public/demo images.
- If you use your own data, ensure you have appropriate rights and handle data in line with your policies (retention, de-identification, access).
- For audits: keep PHI out of logs, checkpoints, and CI artifacts; use synthetic or de-identified data only in the repo.

## Config

Single config: `config/train_config.yaml`. Set `data.root`, `noise.type` (gaussian | rician), `model.*`, `training.*`, and `seed` for full reproducibility.

## Training

```bash
python scripts/run_training.py [config/train_config.yaml]
```

Best model is saved to `checkpoints/best.pt`. Training uses L1 or L2 loss, Adam, and validation PSNR/SSIM with optional early stopping.

## Evaluation

```bash
python scripts/run_evaluate.py --checkpoint checkpoints/best.pt --data_root data/brain_mri
```

Reports test PSNR and SSIM (mean ± std).

## Inference (CLI)

```bash
python -m src.inference.predict --checkpoint checkpoints/best.pt --input path/to/img1.png path/to/img2.png --output ./predictions
```

## Inference (FastAPI)

```bash
export DENOISE_CHECKPOINT=checkpoints/best.pt
python scripts/run_api.py --port 8000
```

Then POST an image to `http://localhost:8000/denoise` (e.g. with curl or the `/docs` UI). Response is the denoised PNG.

## Docker (containerized inference)

Build and run the **FastAPI inference server** (upload UI at http://localhost:8000):

```bash
docker build -t brain-mri-denoise .
docker run --rm -p 8000:8000 \
  -v $(pwd)/checkpoints:/checkpoints \
  -e DENOISE_CHECKPOINT=/checkpoints/best.pt \
  brain-mri-denoise
```

Then open http://localhost:8000 and use the upload button.

**CLI inference in Docker** (optional):

```bash
docker run --rm -v $(pwd)/checkpoints:/ckpt -v $(pwd)/data:/data -v $(pwd)/predictions:/out \
  --entrypoint python brain-mri-denoise /app/scripts/docker_inference.py \
  -c /ckpt/best.pt -i /data/Testing/glioma_tumor/image\(1\).jpg -o /out
```

## ONNX Export

```bash
python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output model.onnx
```

## Benchmark (inference latency)

```bash
python scripts/benchmark_inference.py --checkpoint checkpoints/best.pt --sizes 256 512 [--output benchmark_results.json]
```

Reports mean latency (ms/image) and, on CUDA, peak GPU memory. Use `--output` to write a JSON summary (performance under different input sizes). Use ONNX/ONNX Runtime for additional deployment benchmarks.

## Model comparison (ablations)

To compare configurations (e.g. L1 vs L2, Gaussian vs Rician, depth), change `config/train_config.yaml` and run training and evaluation for each. Record test PSNR/SSIM in a table (e.g. in README or `docs/`). Example:

| Config        | Test PSNR | Test SSIM |
|---------------|-----------|-----------|
| L2, Gaussian  | 28.3      | 0.77      |
| L1, Gaussian  | (run and fill) | |
| L2, Rician    | (run and fill) | |

## Deployment

- **Local:** Run uvicorn or CLI as above.
- **Docker:** Build and run the API container; mount checkpoints and expose port 8000.
- **Cloud:** Deploy the Docker image to a container service (e.g. Render, AWS ECS, GCP Cloud Run, Azure Container Apps) with DENOISE_CHECKPOINT set and checkpoints in a mounted volume or object store. For ONNX-only (no PyTorch), use `requirements-onnx.txt` and build with an exported `model.onnx` in the image.

## Documentation and compliance

- **Design and testing:** `docs/DESIGN.md`, `docs/TEST_PLAN.md`, `docs/TRACEABILITY.md`.
- **Risk analysis:** `docs/RISK_ANALYSIS.md` (portfolio-level; not a certified SaMD).
- **Coding standards:** Ruff for linting; type hints and docstrings used in key modules. Run `ruff check src scripts`.

## Reproducibility

All results are reproducible with `config/train_config.yaml` and `seed: 42`. Same seed is used for data split and training. Use the same `requirements.txt` (pinned versions).

## CI

GitHub Actions: lint (ruff), unit tests (pytest), smoke test (inference), and Docker image build. See `.github/workflows/ci.yml`.
