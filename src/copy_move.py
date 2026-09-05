"""
Copy-move forgery detection via ORB keypoint self-matching.

The idea: if a region of an image was duplicated elsewhere in the *same*
image (e.g. a signature or an amount cloned to cover something else), then
that region's local keypoints will have near-identical descriptors to
keypoints somewhere else in the same image -- a real photo/scan almost
never has two regions that match each other that well by chance.

This catches exactly what ELA is blind to: a clone using pixels that were
already part of the same, single-compression-history image, so there's no
recompression signature for ELA to pick up on. That's why the two run in
parallel in the architecture rather than one replacing the other.

Recipe:
  1. Detect ORB keypoints + descriptors across the whole image.
  2. Match each keypoint against all others *in the same image* (self-match).
  3. Discard matches between keypoints that are close together (that's just
     a normal repeated local pattern, e.g. lines of text) -- keep only
     matches that are spatially far apart AND geometrically consistent
     (same relative offset), since a real copy-move is a rigid translation
     of one patch to another location.
  4. Cluster the surviving matched points; if a big enough cluster with a
     dominant offset exists, flag it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CopyMoveResult:
    matched_points_src: list[tuple[int, int]] = field(default_factory=list)
    matched_points_dst: list[tuple[int, int]] = field(default_factory=list)
    dominant_offset: tuple[float, float] | None = None
    inlier_count: int = 0
    score: float = 0.0  # 0-1, how confident we are a copy-move exists


def detect_copy_move(
    image_bgr: np.ndarray,
    min_distance: int = 40,
    ratio_thresh: float = 0.6,
    offset_cluster_tol: float = 4.0,
    min_inliers: int = 8,
) -> CopyMoveResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=4000)
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None or len(kps) < 20:
        return CopyMoveResult()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # k=3: best match is a keypoint against itself (distance 0), so we look
    # at the 2nd and 3rd nearest neighbors for genuine self-similarity.
    knn = bf.knnMatch(descs, descs, k=3)

    candidate_offsets = []
    candidate_pairs = []

    for matches in knn:
        if len(matches) < 3:
            continue
        # matches[0] is the point matching itself (distance ~0) -> skip it.
        m = matches[1]
        third = matches[2]

        if third.distance == 0:
            continue
        # Lowe's-style ratio test adapted for self-matching.
        if m.distance / (third.distance + 1e-6) > ratio_thresh:
            continue

        pt1 = np.array(kps[m.queryIdx].pt)
        pt2 = np.array(kps[m.trainIdx].pt)
        dist = np.linalg.norm(pt1 - pt2)
        if dist < min_distance:
            continue  # too close: likely just repeated local texture

        offset = pt2 - pt1
        candidate_offsets.append(offset)
        candidate_pairs.append((tuple(pt1.astype(int)), tuple(pt2.astype(int))))

    if not candidate_offsets:
        return CopyMoveResult()

    offsets = np.array(candidate_offsets)

    # Cluster offsets: a real copy-move produces MANY keypoint pairs that
    # all share (approximately) the same translation vector. Noise matches
    # scatter randomly across offset-space. So the densest cluster of
    # offsets is the signal.
    best_center = None
    best_inliers = np.zeros(len(offsets), dtype=bool)
    for i in range(len(offsets)):
        dists = np.linalg.norm(offsets - offsets[i], axis=1)
        inliers = dists < offset_cluster_tol
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_center = offsets[inliers].mean(axis=0)

    inlier_count = int(best_inliers.sum())
    if inlier_count < min_inliers or best_center is None:
        return CopyMoveResult(
            matched_points_src=[p[0] for p in candidate_pairs],
            matched_points_dst=[p[1] for p in candidate_pairs],
            inlier_count=inlier_count,
            score=min(inlier_count / (min_inliers * 3), 0.3),
        )

    src_pts = [candidate_pairs[i][0] for i in range(len(candidate_pairs)) if best_inliers[i]]
    dst_pts = [candidate_pairs[i][1] for i in range(len(candidate_pairs)) if best_inliers[i]]

    # Score saturates smoothly rather than jumping straight to 1.0 -- more
    # inliers and a wider spread both increase confidence.
    score = float(min(1.0, inlier_count / 40))

    return CopyMoveResult(
        matched_points_src=src_pts,
        matched_points_dst=dst_pts,
        dominant_offset=tuple(best_center),
        inlier_count=inlier_count,
        score=score,
    )


def visualize(image_bgr: np.ndarray, result: CopyMoveResult) -> np.ndarray:
    vis = image_bgr.copy()
    for (x1, y1), (x2, y2) in zip(result.matched_points_src, result.matched_points_dst):
        cv2.circle(vis, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 5, (0, 165, 255), -1)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return vis


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/tampered/sample_00_copymove.png"
    img = cv2.imread(path)
    result = detect_copy_move(img)
    out_path = "data/outputs/" + path.split("/")[-1].rsplit(".", 1)[0] + "_copymove_vis.png"
    cv2.imwrite(out_path, visualize(img, result))
    print(f"score: {result.score:.3f}  inliers: {result.inlier_count}  offset: {result.dominant_offset}  saved: {out_path}")
