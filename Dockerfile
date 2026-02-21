# Brain MRI Denoising - ONNX API (low memory, e.g. Render free tier)
# Stage 1: export PyTorch checkpoint to ONNX
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY checkpoints/ checkpoints/
ENV PYTHONPATH=/app
RUN python scripts/export_onnx.py --checkpoint checkpoints/best.pt --output /app/model.onnx

# Stage 2: run API with ONNX only (no PyTorch)
FROM python:3.11-slim
WORKDIR /app
COPY requirements-onnx.txt .
RUN pip install --no-cache-dir -r requirements-onnx.txt
COPY config/ config/
COPY src/ src/
COPY --from=builder /app/model.onnx /app/model.onnx
ENV PYTHONPATH=/app
ENV DENOISE_CHECKPOINT=/app/model.onnx
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
