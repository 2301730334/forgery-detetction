# Document Forgery Detector

Detects tampering (splicing, copy-move) in scanned documents by combining
classical forensic CV techniques with a CNN classifier, served through an
MLOps pipeline (tracking, API, CI/CD, monitoring).

See the full project brief and architecture diagram: (link to your Claude artifact here).

## Structure

- `src/ela.py` — Error Level Analysis: recompress + diff to expose re-compression artifacts.
- `src/copy_move.py` — ORB keypoint matching to find duplicated regions within an image.
- `src/generate_synthetic_data.py` — builds a small authentic/tampered document test set.
- `src/demo.py` — runs both detectors end-to-end on a sample image and saves visualizations.
- `data/authentic/`, `data/tampered/` — sample images.
- `data/outputs/` — generated visualizations.

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

## Run the Week 1 demo

```bash
python src/generate_synthetic_data.py
python src/demo.py
```

## Week 2-3: CNN on ELA maps

```bash
python src/train.py          # trains, logs to sqlite:///mlflow.db, saves models/best_model.pt
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse runs at localhost:5000
python src/demo.py           # now fuses copy-move + trained CNN scores
```

Findings from this run (360 synthetic docs, 120/class):
- Held-out val AUC: ~0.997, val accuracy ~94-96% (see train.py output).
- The CNN picks up splice cases the raw ELA statistic couldn't (Week 1's finding) --
  it learns the texture-normalized pattern a single global threshold can't.
- Remaining errors are false positives on *authentic* pages (score just over 0.5) --
  consistent with training class imbalance (120 authentic vs 240 tampered). Next
  tuning step: balance the classes or apply class weighting in the loss.

## Week 4: FastAPI serving + Docker

The trained model is now served behind a real API instead of being run through
one-off scripts -- this is the "FastAPI" box in the architecture diagram.

```bash
uvicorn app.main:app --reload --port 8000    # from the project root
curl -X POST http://localhost:8000/analyze -F "file=@data/tampered/sample_000_splice.png"
```

Tested against real samples in this repo:
- authentic sample -> `{"verdict": "authentic", "fused_score": 0.33, ...}`
- copy-move sample -> `{"verdict": "tampered", "fused_score": 1.0, ...}`
- splice sample -> `{"verdict": "tampered", "fused_score": 0.9999, ...}`

`GET /health` reports whether the model checkpoint loaded. First request after
startup is slower (~2.5s, PyTorch's one-time internal warmup); every request
after that is 100-250ms.

### Docker

```bash
docker build -t forgery-detector .
docker run -p 8000:8000 forgery-detector
```

Note: the Docker image builds and runs a plain Python + FastAPI + PyTorch CPU
service with no unusual dependencies, but this dev sandbox's network policy
blocks pulls from Docker Hub, so the build itself couldn't be verified *in
this session* -- verify it once with `docker build .` in an environment with
normal registry access (your own machine, or your CI) before relying on it.
