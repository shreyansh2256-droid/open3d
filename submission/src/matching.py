"""
matching.py — Feature matching between image pairs with Lowe ratio test.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from .io_utils import ensure_dir, safe_json_dump

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Matcher creation
# ─────────────────────────────────────────────────────────────────────────────

def create_matcher(cfg: Dict, desc_type: str = "SIFT"):
    """Create BFMatcher or FLANN matcher."""
    method = cfg.get("matching", {}).get("method", "FLANN").upper()
    cross_check = cfg.get("matching", {}).get("cross_check", False)

    if desc_type == "ORB":
        # ORB uses binary descriptors → Hamming distance
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        logger.info("[INFO] Matcher: BFMatcher (Hamming, ORB)")
        return matcher, "BF"

    if method == "FLANN":
        index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        logger.info("[INFO] Matcher: FLANN (SIFT/float descriptors)")
        return matcher, "FLANN"
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
        logger.info("[INFO] Matcher: BFMatcher (L2)")
        return matcher, "BF"


# ─────────────────────────────────────────────────────────────────────────────
# Pair generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_pairs(
    image_names: List[str],
    strategy: str = "sequential",
    window: int = 10,
) -> List[Tuple[int, int]]:
    """Generate image pairs to match."""
    n = len(image_names)
    pairs = []
    if strategy == "exhaustive":
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
    else:  # sequential
        for i in range(n):
            for j in range(i + 1, min(i + 1 + window, n)):
                pairs.append((i, j))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair matching
# ─────────────────────────────────────────────────────────────────────────────

def match_pair(
    desc1: np.ndarray,
    desc2: np.ndarray,
    matcher,
    matcher_type: str,
    ratio_thresh: float = 0.75,
    cross_check: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match descriptors between two images.
    Returns (idx1, idx2) arrays of matched keypoint indices (after ratio test).
    """
    if desc1 is None or desc2 is None:
        return np.empty(0, int), np.empty(0, int)
    if len(desc1) < 2 or len(desc2) < 2:
        return np.empty(0, int), np.empty(0, int)

    try:
        if cross_check and matcher_type == "BF":
            # Use knnMatch with k=1 for cross-check
            matches_ab = matcher.match(desc1, desc2)
            matches_ba = matcher.match(desc2, desc1)
            ba_set = {m.trainIdx: m.queryIdx for m in matches_ba}
            good = []
            for m in matches_ab:
                if ba_set.get(m.trainIdx) == m.queryIdx:
                    good.append(m)
            if not good:
                return np.empty(0, int), np.empty(0, int)
            idx1 = np.array([m.queryIdx for m in good], dtype=int)
            idx2 = np.array([m.trainIdx for m in good], dtype=int)
        else:
            # kNN + Lowe ratio test
            knn = matcher.knnMatch(desc1, desc2, k=2)
            idx1_list, idx2_list = [], []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    idx1_list.append(m.queryIdx)
                    idx2_list.append(m.trainIdx)
            idx1 = np.array(idx1_list, dtype=int)
            idx2 = np.array(idx2_list, dtype=int)

        return idx1, idx2

    except cv2.error as e:
        logger.debug(f"Matching error: {e}")
        return np.empty(0, int), np.empty(0, int)


# ─────────────────────────────────────────────────────────────────────────────
# Full matching pass
# ─────────────────────────────────────────────────────────────────────────────

def match_all_pairs(
    image_names: List[str],
    features: Dict[str, Dict],
    cfg: Dict,
    output_dir: str,
    debug: bool = False,
) -> Dict[Tuple[str, str], Dict]:
    """
    Match all candidate image pairs.
    Returns dict: (name_i, name_j) → match info dict.
    """
    match_cfg = cfg.get("matching", {})
    strategy = match_cfg.get("strategy", "sequential")
    window = match_cfg.get("sequential_window", 10)
    ratio_thresh = match_cfg.get("ratio_threshold", 0.75)
    min_matches = match_cfg.get("min_matches", 30)
    cross_check = match_cfg.get("cross_check", False)

    # Determine descriptor type from first valid image
    desc_type = "SIFT"
    for nm in image_names:
        d = features.get(nm, {}).get("descriptors")
        if d is not None and d.dtype == np.uint8:
            desc_type = "ORB"
            break

    matcher, matcher_type = create_matcher(cfg, desc_type)
    pairs = generate_pairs(image_names, strategy, window)

    logger.info(f"[INFO] Matching {len(pairs)} image pairs (strategy={strategy}, window={window})...")
    t0 = time.time()

    results = {}
    good_pairs = 0

    for k, (i, j) in enumerate(pairs):
        name_i = image_names[i]
        name_j = image_names[j]
        feat_i = features.get(name_i, {})
        feat_j = features.get(name_j, {})

        if not feat_i.get("valid") or not feat_j.get("valid"):
            continue

        desc_i = feat_i["descriptors"]
        desc_j = feat_j["descriptors"]

        if desc_i is None or desc_j is None:
            continue

        idx1, idx2 = match_pair(desc_i, desc_j, matcher, matcher_type,
                                 ratio_thresh, cross_check)
        n_matches = len(idx1)

        if n_matches < min_matches:
            continue

        kp_i = feat_i["keypoints"]
        kp_j = feat_j["keypoints"]
        pts_i = kp_i[idx1] if len(kp_i) > 0 else np.empty((0, 2))
        pts_j = kp_j[idx2] if len(kp_j) > 0 else np.empty((0, 2))

        results[(name_i, name_j)] = {
            "idx1": idx1,
            "idx2": idx2,
            "pts1": pts_i,
            "pts2": pts_j,
            "n_raw": n_matches,
            "n_filtered": n_matches,
            "n_verified": 0,  # filled in geometric verification
            "inlier_mask": None,
            "inlier_ratio": 0.0,
        }
        good_pairs += 1

        if (k + 1) % 100 == 0:
            logger.info(f"  [{k+1}/{len(pairs)}] {good_pairs} good pairs so far")

    logger.info(f"[INFO] Matching done: {good_pairs}/{len(pairs)} pairs have ≥{min_matches} matches"
                f" ({time.time()-t0:.1f}s)")

    if debug:
        _save_match_vis(results, features, cfg, output_dir, max_vis=5)

    return results


def _save_match_vis(results, features, cfg, output_dir, max_vis=5):
    """Save match visualisations for the top pairs."""
    try:
        debug_dir = ensure_dir(os.path.join(output_dir, "debug"))
        from .io_utils import load_image_rgb
        max_dim = cfg.get("image", {}).get("resize_for_processing", 1600)
        for k, ((n1, n2), info) in enumerate(list(results.items())[:max_vis]):
            # Find paths lazily
            pass  # Skip visual save to keep code fast for hackathon
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_matching_report(matches: Dict, output_dir: str, cfg: Dict) -> Dict:
    n_raw = [v["n_raw"] for v in matches.values()]
    n_ver = [v["n_verified"] for v in matches.values()]
    report = {
        "total_pairs": len(matches),
        "mean_raw_matches": float(np.mean(n_raw)) if n_raw else 0,
        "mean_verified_matches": float(np.mean(n_ver)) if n_ver else 0,
        "max_raw_matches": int(max(n_raw)) if n_raw else 0,
        "pairs": {
            f"{n1}|{n2}": {
                "n_raw": v["n_raw"],
                "n_filtered": v["n_filtered"],
                "n_verified": v["n_verified"],
                "inlier_ratio": v["inlier_ratio"],
            }
            for (n1, n2), v in matches.items()
        },
    }
    safe_json_dump(report, os.path.join(output_dir, cfg.get("output", {}).get("matching_report", "matching_report.json")))
    return report
