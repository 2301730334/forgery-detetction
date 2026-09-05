# Document Forgery Detector -- serving image
#
# Two-stage-ish build kept simple: install CPU-only torch explicitly (the
# default PyPI wheel pulls in CUDA libraries nobody needs for this model,
# which would roughly triple the image size for zero benefit on a
# CPU-only free-tier deploy target).

FROM python:3.11-slim

WORKDIR /srv

# libgl1/libglib2.0-0: required by opencv-python-headless's native bindings
# even in "headless" mode, for a handful of image codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^torch(vision)?$' requirements.txt > requirements.nocuda.txt \
    && pip install --no-cache-dir -r requirements.nocuda.txt

COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

ENV MLFLOW_DISABLE_TELEMETRY=true
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
