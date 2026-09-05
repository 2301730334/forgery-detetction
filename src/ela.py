"""
Error Level Analysis (ELA)

The idea: a JPEG is a lossy format, so every save re-quantizes the image.
If a document has been edited and re-saved, the pasted/edited region was
compressed a *second* time (or came from a different source image), so it
settles into a different local error level than the untouched parts of the
page, which were compressed only once.

Recipe:
  1. Re-save the (possibly already-JPEG) image at a known quality (e.g. 90).
  2. Diff the re-saved version against the input, pixel by pixel.
  3. Amplify the diff so it's visible / usable as a feature map.

Untampered regions -> low, uniform residual.
Tampered regions    -> localized bright spots in the residual map.

This module is deliberately dependency-light (Pillow + numpy) and is the
Week 1 baseline referenced in the architecture diagram's "Preprocess + ELA
maps" step, and its output later becomes the CNN's input.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops


@dataclass
class ELAResult:
    ela_image: Image.Image      # amplified visual residual (for display)
    residual: np.ndarray        # raw per-pixel diff, shape (H, W, 3), 0-255
    max_diff: int                # max pixel-level difference found
    score: float                 # single scalar "how suspicious" 0-1


def compute_ela(image: Image.Image, quality: int = 90, scale: float = 15.0) -> ELAResult:
    """
    Run ELA on a PIL image.

    quality: JPEG re-save quality. 90 is the standard choice in the ELA
             literature -- high enough not to introduce its own artifacts,
             low enough to expose recompression differences.
    scale:   multiplier used only for the *visual* ela_image; the raw
             residual returned is unscaled so downstream models see the
             true signal, not a clipped/amplified one.
    """
    rgb = image.convert("RGB")

    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(rgb, resaved)
    residual = np.asarray(diff, dtype=np.uint8)

    extrema = diff.getextrema()
    max_diff = max(ch[1] for ch in extrema) or 1

    # Visual version: stretch contrast so faint differences become visible.
    visual_scale = min(255.0 / max_diff, scale)
    ela_visual = diff.point(lambda p: min(255, int(p * visual_scale)))

    # A simple scalar score: high-percentile residual energy, normalized.
    # Real tampered regions show up as a small patch of high values against
    # an otherwise near-zero background, so a high percentile (not the
    # mean) is what actually separates "clean scan" from "edited region".
    score = float(np.percentile(residual, 99.5) / 255.0)

    return ELAResult(ela_image=ela_visual, residual=residual, max_diff=max_diff, score=score)


def ela_heatmap(residual: np.ndarray) -> np.ndarray:
    """Collapse the 3-channel residual into a single-channel intensity map,
    the form the CNN and the fusion step actually consume."""
    return residual.astype(np.float32).mean(axis=2)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/tampered/sample_01.png"
    img = Image.open(path)
    result = compute_ela(img)
    out_path = "data/outputs/" + path.split("/")[-1].rsplit(".", 1)[0] + "_ela.png"
    result.ela_image.save(out_path)
    print(f"ELA score: {result.score:.4f}  max_diff: {result.max_diff}  saved: {out_path}")
