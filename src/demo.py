"""
End-to-end demo: runs all three detectors on a document and fuses their
scores into a single tamper probability -- the full "parallel detectors ->
fusion" step from the architecture diagram.

  - copy_move.py  -> geometric self-similarity (catches copy-move)
  - ela.py         -> raw recompression statistic (weak alone on documents,
                       see the note printed below -- but its *map* is what
                       feeds the CNN)
  - model.py + models/best_model.pt -> CNN trained on ELA maps (catches
                       splice, the case the raw ELA statistic couldn't)

Fusion is max(copy_move_score, cnn_score): each targets a different,
non-overlapping tamper type, so "either one is confident" is correct.
The raw ELA score is reported for reference but isn't part of the fused
verdict, per the Week 1 finding.

Caveat: this demo scores the *same* synthetic images the CNN was trained
on, so its near-perfect numbers here are a pipeline sanity check, not a
generalization claim -- that's what the held-out val_auc from train.py is
for. Swap in unseen documents (or real CASIA v2 test images) to see a fair
number.
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
from PIL import Image

from ela import compute_ela
from copy_move import detect_copy_move
from dataset import _load_ela_tensor
from model import ELACNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pt")


def load_cnn():
    model = ELACNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    else:
        print(f"warning: no trained model found at {MODEL_PATH} -- run src/train.py first. "
              "CNN scores below will be untrained/meaningless.")
    model.eval()
    return model


def cnn_score(model, path: str) -> float:
    x = _load_ela_tensor(path).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return prob


def analyze(path: str, cnn_model) -> dict:
    pil_img = Image.open(path)
    ela_result = compute_ela(pil_img)

    cv_img = cv2.imread(path)
    cm_result = detect_copy_move(cv_img)

    cnn_p = cnn_score(cnn_model, path)

    fused_score = max(cm_result.score, cnn_p)
    verdict = "TAMPERED" if fused_score > 0.5 else "authentic"

    return {
        "path": path,
        "ela_score": round(ela_result.score, 3),
        "copy_move_score": round(cm_result.score, 3),
        "cnn_score": round(cnn_p, 3),
        "fused_score": round(fused_score, 3),
        "verdict": verdict,
    }


def main():
    os.makedirs(os.path.join(ROOT, "data/outputs"), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(ROOT, "data/authentic/*.png"))) + \
        sorted(glob.glob(os.path.join(ROOT, "data/tampered/*.png")))

    cnn_model = load_cnn()
    results = [analyze(p, cnn_model) for p in paths]

    header = f"{'file':30} {'ela':>6} {'copymove':>9} {'cnn':>6} {'fused':>7}  verdict"
    print(header)
    print("-" * len(header))
    correct = 0
    for r in results:
        label = "TAMPERED" if "/tampered/" in r["path"] else "authentic"
        is_correct = (r["verdict"] == "TAMPERED") == (label == "TAMPERED")
        correct += is_correct
        flag = "" if is_correct else "  <-- MISS"
        print(f"{os.path.basename(r['path']):30} {r['ela_score']:6.3f} {r['copy_move_score']:9.3f} "
              f"{r['cnn_score']:6.3f} {r['fused_score']:7.3f}  {r['verdict']:9}{flag}")

    print(f"\n{correct}/{len(results)} correct (copy-move + CNN fusion; ELA raw score shown for reference only).")
    print(
        "\nNote: this set was also used to train the CNN, so these numbers are a pipeline "
        "sanity check, not a generalization claim -- see val_auc printed by src/train.py for "
        "the honest held-out number. Next real step: evaluate on documents the CNN has never "
        "seen (a fresh synthetic batch, or real CASIA v2 test images)."
    )


if __name__ == "__main__":
    main()
