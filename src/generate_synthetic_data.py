"""
Generates an authentic-vs-tampered document test set so the detection
pipeline can be exercised end-to-end without downloading the full CASIA v2
dataset in this environment. Swap this out for real CASIA v2 / your own
scans once the pipeline is validated -- the point right now is to prove the
mechanism works, and give the CNN enough *varied* examples to actually
generalize from, before scaling up to real data.

Two tamper types are simulated, matching the two detectors we're building:
  - "copy_move": a region (e.g. an amount or a signature) is duplicated
    elsewhere on the same page.
  - "splice": a region is pasted in after being re-compressed at a
    different JPEG quality, simulating content lifted from another source
    image -- this is what ELA is specifically designed to catch.

Everything that could make the CNN memorize a template instead of learning
the tamper signal is randomized per-sample: company name, dates, box
position/size, paste location, and (for splice) the JPEG quality gap.
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT_AUTH = "data/authentic"
OUT_TAMP = "data/tampered"
W, H = 850, 1100  # roughly A4 at 100dpi

COMPANIES = [
    "Acme Logistics Pvt Ltd", "Northwind Traders", "Meridian Supplies",
    "Solaris Freight Co", "Vantage Retail Group", "Harborline Imports",
    "Crestpoint Manufacturing", "Bluepeak Distributors", "Ironvale Textiles",
    "Sundial Wholesale",
]
TERMS = ["Net 15", "Net 30", "Net 45", "Due on Receipt"]


def _blank_document(rng: np.random.Generator) -> tuple[Image.Image, tuple, tuple]:
    """Fake a simple scanned invoice/form: white page, header rule, body
    text lines, and a boxed 'amount' field -- enough structure for copy-move
    and splice edits to have something plausible to target. Box position,
    size and page tone all vary per sample."""
    bg = tuple(int(v) for v in (248 + rng.integers(-4, 4, size=3)))
    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    draw.text((60, 50), "INVOICE", fill=(20, 20, 20), font=font)
    draw.line([(60, 95), (W - 60, 95)], fill=(60, 60, 60), width=2)

    company = COMPANIES[rng.integers(0, len(COMPANIES))]
    lines = [
        f"Invoice #: INV-{int(rng.integers(1000, 9999))}",
        f"Bill To: {company}",
        f"Date: 2026-{int(rng.integers(1,13)):02d}-{int(rng.integers(1,28)):02d}",
        f"Terms: {TERMS[rng.integers(0, len(TERMS))]}",
    ]
    y = 140
    for line in lines:
        draw.text((60, y), line, fill=(40, 40, 40), font=font_small)
        y += 40

    # Amount box: position/size jitter so the CNN can't just memorize
    # "the tamper is always at pixel (60, 340)".
    bx = 60 + int(rng.integers(-10, 30))
    by = y + 40 + int(rng.integers(-10, 20))
    bw = 240 + int(rng.integers(-20, 40))
    bh = 55 + int(rng.integers(-8, 12))
    box = (bx, by, bx + bw, by + bh)
    amount = rng.integers(80, 9999)
    draw.rectangle(box, outline=(30, 30, 30), width=2)
    draw.text((box[0] + 15, box[1] + 18), f"AMOUNT DUE: ${amount:,}.00", fill=(10, 10, 10), font=font_small)

    # Signature scribble, position jitter too.
    sx = 60 + int(rng.integers(-10, 10))
    sy = H - 220 + int(rng.integers(-15, 15))
    sig_box = (sx, sy, sx + 260, sy + 70)
    draw.rectangle(sig_box, outline=(30, 30, 30), width=2)
    pts = [(sig_box[0] + 20 + rng.integers(0, 200), sig_box[1] + 10 + rng.integers(0, 50)) for _ in range(8)]
    draw.line(pts, fill=(20, 20, 80), width=3)
    draw.text((sig_box[0], sig_box[1] - 25), "Signature", fill=(80, 80, 80), font=font_small)

    return img, box, sig_box


def make_authentic(idx: int, rng: np.random.Generator) -> str:
    img, _, _ = _blank_document(rng)
    path = os.path.join(OUT_AUTH, f"sample_{idx:03d}.png")
    img.save(path)
    return path


def make_tampered_copy_move(idx: int, rng: np.random.Generator) -> str:
    """Duplicate the amount box onto a random blank area of the page --
    simulates copy-pasting a value from elsewhere in the same document."""
    img, amount_box, sig_box = _blank_document(rng)

    patch = img.crop(amount_box)
    # Paste somewhere in the empty middle of the page, away from both boxes.
    paste_x = int(rng.integers(400, 700))
    paste_y = int(rng.integers(550, 850))
    img.paste(patch, (paste_x, paste_y))

    path = os.path.join(OUT_TAMP, f"sample_{idx:03d}_copymove.png")
    img.save(path)
    return path


def make_tampered_splice(idx: int, rng: np.random.Generator) -> str:
    """Paste a region that's been through a *second* JPEG compression pass
    at a different quality -- simulates content spliced in from another
    (already-compressed) source image. This is the case ELA targets."""
    img, amount_box, _ = _blank_document(rng)

    patch = img.crop(amount_box)
    quality = int(rng.integers(20, 55))  # vary the compression gap
    buf = io.BytesIO()
    patch.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    degraded_patch = Image.open(buf).convert("RGB")

    img.paste(degraded_patch, amount_box)

    path = os.path.join(OUT_TAMP, f"sample_{idx:03d}_splice.png")
    img.save(path)
    return path


def main(n: int = 120):
    os.makedirs(OUT_AUTH, exist_ok=True)
    os.makedirs(OUT_TAMP, exist_ok=True)
    os.makedirs("data/outputs", exist_ok=True)

    made = []
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        made.append(make_authentic(i, rng))
        rng = np.random.default_rng(2000 + i)
        made.append(make_tampered_copy_move(i, rng))
        rng = np.random.default_rng(3000 + i)
        made.append(make_tampered_splice(i, rng))

    print(f"Generated {len(made)} images ({n} per class) in data/authentic/ and data/tampered/")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    main(n)
