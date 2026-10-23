# Chest X-ray triage API. CPU-only image; runs on Hugging Face Spaces (port 7860).
FROM python:3.11-slim

# OpenCV (pulled in by grad-cam) needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-service.txt

USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1
COPY --chown=user cxr ./cxr
COPY --chown=user service ./service

# Bake the model weights into the image so the first request is not a download
RUN python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-chex')"

EXPOSE 7860
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
