# Google Cloud Run. Build remotely with `gcloud run deploy --source .` —
# no local Docker needed. See deploy/CLOUD_RUN.md.
# Expects backend/data/ to contain products_siglip.index, id_map_siglip.npy and
# a slimmed dorm_deals.db — see deploy/slim_db.py.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image a few GB smaller than the default CUDA wheels
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# bake the SigLIP weights into the image so cold starts don't wait on a
# download. Only the vision tower is served; embeddings.py (CLIP) is used by the
# offline scripts, which do not run here.
RUN python -c "from transformers import AutoImageProcessor, SiglipVisionModel; \
    SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224'); \
    AutoImageProcessor.from_pretrained('google/siglip-base-patch16-224')"

COPY backend/ backend/
COPY frontend/ frontend/

ENV PYTHONUNBUFFERED=1
WORKDIR /app/backend/src

# Cloud Run injects PORT and ignores EXPOSE. Shell form so the variable expands;
# exec so uvicorn is PID 1 and receives SIGTERM on scale-down.
EXPOSE 8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
