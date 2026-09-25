"""
features.py — SIFT/ORB feature extraction per image.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .io_utils import load_image_gray, ensure_dir

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Detector creation
# ─────────────────────────────────────────────────────────────────────────────

def create_detector(cfg: Dict):
    """Create SIFT or ORB detector from config."""
    method = cfg.get("features", {}).get("method", "SIFT").upper()
    feat_cfg = cfg.get("features", {})

    if method == "SIFT":
        try:
            n = feat_cfg.get("sift_nfeatures", 4000)
            ct = feat_cfg.get("sift_contrast_threshold", 0.03)
            et = feat_cfg.get("sift_edge_threshold", 10)
            detector = cv2.SIFT_create(
                nfeatures=n,
                contrastThreshold=ct,
                edgeThreshold=et,
            )
            logger.info("[INFO] Feature detector: SIFT")
            return detector, "SIFT"
        except AttributeError:
            logger.warning("SIFT not available; falling back to ORB")
            method = "ORB"

    if method == "ORB":
        n = feat_cfg.get("orb_nfeatures", 6000)
        detector = cv2.ORB_create(nfeatures=n)
        logger.info("[INFO] Feature detector: ORB")
        return detector, "ORB"

    raise ValueError(f"Unknown feature method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_image(
    img_path: Path,
    detector,
    max_dim: int = 0,
) -> Tuple[Optional[List], Optional[np.ndarray]]:
    """Detect keypoints and compute descriptors for one image."""
    gray = load_image_gray(img_path, max_dim)
    if gray is None:
        return None, None

    try:
        keypoints, descriptors = detector.detectAndCompute(gray, None)
    except cv2.error as e:
        logger.warning(f"OpenCV error on {img_path.name}: {e}")
        return None, None

    if descriptors is None or len(keypoints) == 0:
        logger.warning(f"No features in {img_path.name}")
        return [], np.empty((0, 128), dtype=np.float32)

    return keypoints, descriptors


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def keypoints_to_array(keypoints) -> np.ndarray:
    """Convert list of cv2.KeyPoint → (N, 2) float32 array of (x, y)."""
    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def keypoints_to_full_array(keypoints) -> np.ndarray:
    """Convert list of cv2.KeyPoint → (N, 4) array: x, y, size, angle."""
    if not keypoints:
        return np.empty((0, 4), dtype=np.float32)
    return np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints], dtype=np.float32)


def array_to_keypoints(arr: np.ndarray):
    """Convert (N, 4) array back to list of cv2.KeyPoint."""
    kps = []
    for row in arr:
        kp = cv2.KeyPoint(x=float(row[0]), y=float(row[1]),
                          size=float(row[2]) if len(row) > 2 else 1.0,
                          angle=float(row[3]) if len(row) > 3 else 0.0)
        kps.append(kp)
    return kps


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: extract all images
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(
    images: List[Path],
    cfg: Dict,
    output_dir: str,
    debug: bool = False,
) -> Dict[str, Dict]:
    """
    Extract features for all images.
    Returns dict: image_name → {keypoints_arr, descriptors, kp_count}
    """
    feat_dir = ensure_dir(os.path.join(output_dir, "features"))
    feat_cfg = cfg.get("features", {})
    max_dim = cfg.get("image", {}).get("resize_for_processing", 1600)

    detector, method_name = create_detector(cfg)
    features = {}
    total_kp = 0

    logger.info(f"[INFO] Extracting features from {len(images)} images...")
    t0 = time.time()

    for i, img_path in enumerate(images):
        name = img_path.name
        kps, descs = extract_features_image(img_path, detector, max_dim)

        if kps is None:
            logger.warning(f"  Skipping {name} (load failed)")
            features[name] = {"keypoints": np.empty((0, 2), dtype=np.float32),
                              "descriptors": None,
                              "kp_count": 0,
                              "valid": False}
            continue

        kp_arr = keypoints_to_array(kps)
        kp_full = keypoints_to_full_array(kps)
        n = len(kps)
        total_kp += n
        features[name] = {
            "keypoints": kp_arr,          # (N, 2) for matching
            "keypoints_full": kp_full,    # (N, 4) for serialisation
            "descriptors": descs,
            "kp_count": n,
            "valid": n >= 10,
        }

        if (i + 1) % 10 == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{len(images)}] {name}: {n} keypoints  ({elapsed:.1f}s)")

        # Debug feature vis
        if debug and feat_cfg.get("save_visualizations", False):
            _save_feature_vis(img_path, kps, feat_dir, max_dim)

    logger.info(f"[INFO] Feature extraction done: {total_kp} total keypoints in {time.time()-t0:.1f}s")
    return features


def _save_feature_vis(img_path: Path, kps, feat_dir: Path, max_dim: int):
    """Save a keypoint visualisation image."""
    try:
        import cv2
        gray = load_image_gray(img_path, max_dim)
        if gray is None:
            return
        vis = cv2.drawKeypoints(gray, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_path = feat_dir / f"feat_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
    except Exception as e:
        logger.debug(f"Feature vis save error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_report(features: Dict, output_dir: str, cfg: Dict) -> Dict:
    from .io_utils import safe_json_dump
    counts = [v["kp_count"] for v in features.values()]
    valid = [k for k, v in features.items() if v["valid"]]
    report = {
        "total_images": len(features),
        "valid_images": len(valid),
        "total_keypoints": int(sum(counts)),
        "mean_keypoints": float(np.mean(counts)) if counts else 0,
        "min_keypoints": int(min(counts)) if counts else 0,
        "max_keypoints": int(max(counts)) if counts else 0,
        "per_image": {k: v["kp_count"] for k, v in features.items()},
    }
    safe_json_dump(report, os.path.join(output_dir, cfg.get("output", {}).get("feature_report", "feature_report.json")))
    return report
