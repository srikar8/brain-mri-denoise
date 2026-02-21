# Brain MRI Denoising - containerized inference (API server by default)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY checkpoints/ checkpoints/

ENV PYTHONPATH=/app
ENV DENOISE_CHECKPOINT=/app/checkpoints/best.pt
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
