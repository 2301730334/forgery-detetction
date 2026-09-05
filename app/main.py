"""
FastAPI serving layer -- the "FastAPI" box in the architecture diagram.

This is the online path: a client uploads a document, the request fans out
to the three detectors (copy-move, raw ELA, CNN-on-ELA), the scores fuse
into one verdict, and the response (plus a log line) goes back to the
client. Nothing here trains anything -- it only *loads* the model artifact
that src/train.py produced.

Run directly:
  uvicorn app.main:app --reload --port 8000        (from the project root)

Or via Docker (see Dockerfile).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ela import compute_ela, ela_heatmap        # noqa: E402
from copy_move import detect_copy_move          # noqa: E402
from model import ELACNN                        # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_model.pt")
IMG_SIZE = 128  # must match dataset.py's training-time resolution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forgery-api")

app = FastAPI(
    title="Document Forgery Detector",
    description="Fuses ELA, copy-move keypoint matching, and a CNN trained on ELA maps "
                 "into a single tamper verdict for an uploaded document image.",
    version="0.1.0",
)

_model: ELACNN | None = None


def get_model() -> ELACNN:
    """Lazy-loaded, process-wide singleton -- the weights are read from disk
    once per container/process, not once per request."""
    global _model
    if _model is None:
        m = ELACNN()
        if os.path.exists(MODEL_PATH):
            m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            log.info(f"loaded model weights from {MODEL_PATH}")
        else:
            log.warning(f"no model checkpoint found at {MODEL_PATH} -- serving an untrained "
                        "model. Run `python src/train.py` first.")
        m.eval()
        _model = m
    return _model


class DetectorScores(BaseModel):
    ela_score: float
    copy_move_score: float
    copy_move_inliers: int
    cnn_score: float


class AnalyzeResponse(BaseModel):
    request_id: str
    verdict: str
    fused_score: float
    scores: DetectorScores
    latency_ms: float


@app.on_event("startup")
def _warm_start():
    # Load the model at container start, not on the first request, so the
    # first real user doesn't pay the disk-read/deserialize cost.
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


def _cnn_score(model: ELACNN, pil_img: Image.Image) -> float:
    ela_result = compute_ela(pil_img, quality=90)
    heatmap = ela_heatmap(ela_result.residual)
    resized = Image.fromarray(heatmap.astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        logit = model(tensor)
        return torch.sigmoid(logit).item()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    t0 = time.time()
    request_id = str(uuid.uuid4())[:8]

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"unsupported content type: {file.content_type}")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        pil_img = Image.open(io.BytesIO(raw))
        pil_img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}")

    np_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    ela_result = compute_ela(pil_img)
    cm_result = detect_copy_move(np_bgr)
    model = get_model()
    cnn_p = _cnn_score(model, pil_img)

    fused = max(cm_result.score, cnn_p)
    verdict = "tampered" if fused > 0.5 else "authentic"
    latency_ms = (time.time() - t0) * 1000

    log.info(f"[{request_id}] verdict={verdict} fused={fused:.3f} "
             f"ela={ela_result.score:.3f} cm={cm_result.score:.3f} cnn={cnn_p:.3f} "
             f"latency_ms={latency_ms:.1f}")

    return AnalyzeResponse(
        request_id=request_id,
        verdict=verdict,
        fused_score=round(fused, 4),
        scores=DetectorScores(
            ela_score=round(ela_result.score, 4),
            copy_move_score=round(cm_result.score, 4),
            copy_move_inliers=cm_result.inlier_count,
            cnn_score=round(cnn_p, 4),
        ),
        latency_ms=round(latency_ms, 1),
    )
