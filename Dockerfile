# Brain MRI Denoising - ONNX API only (no PyTorch). Export model first: python scripts/export_onnx.py -c checkpoints/best.pt -o model.onnx
FROM python:3.11-slim
WORKDIR /app
COPY requirements-onnx.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 120 -r requirements-onnx.txt
COPY config/ config/
COPY src/ src/
COPY model.onnx /app/model.onnx
ENV PYTHONPATH=/app
ENV DENOISE_CHECKPOINT=/app/model.onnx
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
