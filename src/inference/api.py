"""FastAPI inference: upload image, return denoised image. Uses ONNX when checkpoint is .onnx (no PyTorch)."""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

CHECKPOINT = os.environ.get("DENOISE_CHECKPOINT", "checkpoints/best.pt")
_model = None
_use_onnx = None


def _denoise(model, img):
    """Dispatch to ONNX or PyTorch denoise."""
    if _use_onnx:
        from .predict_onnx import denoise_image as denoise_onnx
        return denoise_onnx(model, img)
    from .predict import denoise_image as denoise_pt
    return denoise_pt(model, img)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _use_onnx
    _model = None
    _use_onnx = str(CHECKPOINT).endswith(".onnx")
    if os.path.isfile(CHECKPOINT):
        if _use_onnx:
            from .predict_onnx import load_model as load_onnx
            _model = load_onnx(CHECKPOINT)
        else:
            from .predict import load_model as load_pt
            from ..utils import get_device
            _model = load_pt(CHECKPOINT, get_device())
    yield
    _model = None
    _use_onnx = None


app = FastAPI(title="Brain MRI Denoising", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/sample/{filename}")
def get_sample(filename: str):
    """Serve sample images from the sample/ directory."""
    sample_dir = Path(__file__).resolve().parent.parent.parent / "sample"
    file_path = sample_dir / filename
    if not file_path.is_file() or not filename.endswith((".jpg", ".png")):
        raise HTTPException(404, "Sample not found")
    return FileResponse(file_path)


@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    """Upload a grayscale or RGB image; returns denoised PNG."""
    if _model is None:
        raise HTTPException(503, "Model not loaded. Set DENOISE_CHECKPOINT to a valid .pt path.")
    try:
        raw = await file.read()
        img = np.array(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32) / 255.0
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")
    out = _denoise(_model, img)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def root():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Brain MRI Denoising</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; }
    input[type="file"] { margin: 0.5rem 0; display: block; }
    button { padding: 0.5rem 1rem; cursor: pointer; background: #2563eb; color: white; border: none; border-radius: 6px; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .samples { display: flex; gap: 0.5rem; margin-top: 0.5rem; flex-wrap: wrap; }
    .sample-thumb { width: 80px; height: 80px; object-fit: cover; border: 2px solid #e5e7eb; border-radius: 6px; cursor: pointer; }
    .sample-thumb:hover { border-color: #2563eb; }
    .error { color: #dc2626; margin-top: 0.5rem; }
    .side-by-side { display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap; }
    .side-by-side > div { flex: 1; min-width: 280px; }
    .side-by-side img { width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; display: block; }
    .side-by-side p { margin: 0.25rem 0 0.5rem; font-size: 0.875rem; color: #6b7280; }
  </style>
</head>
<body>
  <h1>Brain MRI Denoising</h1>
  <p>Upload a brain MRI image (grayscale or RGB). Noisy and denoised images are shown side by side.</p>
  <form id="form">
    <input type="file" id="file" name="file" accept="image/*" required />
    <button type="submit" id="btn">Upload and denoise</button>
  </form>
  <p style="margin-top: 0.75rem; font-size: 0.875rem; color: #6b7280;">Or click a sample below:</p>
  <div class="samples">
    <img src="/sample/01_glioma_noisy.jpg" alt="Sample 1" class="sample-thumb" data-file="01_glioma_noisy.jpg" />
    <img src="/sample/02_meningioma_noisy.jpg" alt="Sample 2" class="sample-thumb" data-file="02_meningioma_noisy.jpg" />
    <img src="/sample/03_no_tumor_noisy.jpg" alt="Sample 3" class="sample-thumb" data-file="03_no_tumor_noisy.jpg" />
    <img src="/sample/04_pituitary_noisy.jpg" alt="Sample 4" class="sample-thumb" data-file="04_pituitary_noisy.jpg" />
  </div>
  <p class="error" id="err"></p>
  <div id="result"></div>
  <script>
    const form = document.getElementById("form");
    const fileInput = document.getElementById("file");
    const btn = document.getElementById("btn");
    const err = document.getElementById("err");
    const result = document.getElementById("result");

    async function processImage(file, noisyUrl) {
      err.textContent = "";
      result.innerHTML = "";
      btn.disabled = true;
      const fd = new FormData();
      fd.append("file", file);
      try {
        const r = await fetch("/denoise", { method: "POST", body: fd });
        if (!r.ok) {
          const t = await r.text();
          throw new Error(t || r.statusText);
        }
        const blob = await r.blob();
        const denoisedUrl = URL.createObjectURL(blob);
        result.innerHTML = '<div class="side-by-side">'
          + '<div><p>Noisy (uploaded)</p><img src="' + noisyUrl + '" alt="noisy" /></div>'
          + '<div><p>Denoised</p><img src="' + denoisedUrl + '" alt="denoised" /></div>'
          + '</div>';
      } catch (e) {
        err.textContent = e.message || "Upload failed";
        URL.revokeObjectURL(noisyUrl);
      }
      btn.disabled = false;
    }

    form.onsubmit = async (e) => {
      e.preventDefault();
      if (!fileInput.files.length) return;
      const noisyUrl = URL.createObjectURL(fileInput.files[0]);
      await processImage(fileInput.files[0], noisyUrl);
    };

    document.querySelectorAll(".sample-thumb").forEach(thumb => {
      thumb.onclick = async () => {
        const filename = thumb.dataset.file;
        const sampleUrl = "/sample/" + filename;
        const resp = await fetch(sampleUrl);
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: "image/jpeg" });
        await processImage(file, sampleUrl);
      };
    });
  </script>
</body>
</html>
"""
