# Software Design (IEC 62304–oriented)

## 1. Purpose

Brain MRI denoising system: ingest noisy (or clean) brain MRI images, run a trained U-Net, output denoised images. Supports training with synthetic noise, CLI and API inference, and containerized deployment.

## 2. System Context

- **Users:** Researchers / operators running training or inference.
- **Inputs:** Configuration (YAML), image files (training set or single image for inference), optional checkpoint path.
- **Outputs:** Trained checkpoint, denoised image(s), metrics (PSNR/SSIM).

## 3. Architecture

### 3.1 Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| Config | `config/train_config.yaml` | Single source for paths, hyperparameters, seed. |
| Data pipeline | `src/data/` | Load images, add synthetic noise (Gaussian/Rician), resize, augment; build train/val/test splits. |
| Model | `src/models/unet.py` | U-Net: encoder–decoder with skip connections, configurable depth and channels. |
| Training | `src/training/train.py` | Training loop: optimizer, loss, validation, checkpointing, early stopping. |
| Metrics | `src/evaluation/metrics.py` | PSNR and SSIM (predictions vs clean reference). |
| Inference (CLI) | `src/inference/predict.py` | Load checkpoint, run on image path(s), write denoised file(s). |
| Inference (API) | `src/inference/api.py` | FastAPI app: POST /denoise (file upload → denoised PNG), GET / health and upload UI. |
| Utils | `src/utils.py` | Device selection (MPS/CUDA/CPU). |

### 3.2 Data Flow

- **Training:** Config → build_datasets (paths, noise, split) → DataLoader → model(noisy) → loss(pred, clean) → backward → save best.pt.
- **Inference:** Checkpoint + image(s) → load_model → denoise_image (or run_inference) → output file(s) or HTTP response.

### 3.3 Interfaces

- **CLI:** `python -m src.inference.predict --checkpoint <path> --input <paths> --output <dir>`.
- **API:** POST /denoise with multipart file; response: image/png (denoised). GET / returns upload UI; GET /health returns status and model_loaded.
- **Docker:** Image runs uvicorn by default; DENOISE_CHECKPOINT env; volumes for checkpoints (and optionally data/output).

## 4. Dependencies

- Python 3.9+, PyTorch, FastAPI, uvicorn, Pillow, numpy, scikit-image, PyYAML. See `requirements.txt` (pinned).

## 5. Versioning and Modification Log

- Code changes tracked in version control. High-level changes summarized in `CHANGELOG.md`.
