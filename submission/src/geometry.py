"""
geometry.py — Geometric verification: fundamental/essential matrix, pose recovery.
"""

import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric verification per pair
# ─────────────────────────────────────────────────────────────────────────────

def verify_pair(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    cfg: Dict,
) -> Dict:
    """
    Estimate fundamental + essential matrix, recover relative pose.
    Returns a result dict with R, t, inlier_mask, etc.
    """
    geo_cfg = cfg.get("geometry", {})
    f_thresh = geo_cfg.get("fundamental_ransac_threshold", 1.5)
    f_conf   = geo_cfg.get("fundamental_confidence", 0.999)
    e_thresh = geo_cfg.get("essential_ransac_threshold", 1.0)
    e_conf   = geo_cfg.get("essential_confidence", 0.999)
    min_inliers = geo_cfg.get("min_inliers", 20)
    min_ratio   = geo_cfg.get("min_inlier_ratio", 0.20)

    result = {
        "success": False,
        "R": None, "t": None,
        "F": None, "E": None,
        "inlier_mask": None,
        "n_inliers": 0,
        "inlier_ratio": 0.0,
    }

    n = len(pts1)
    if n < 8:
        return result

    pts1f = pts1.astype(np.float64)
    pts2f = pts2.astype(np.float64)

    # ── Fundamental matrix ───────────────────────────────────────────────────
    try:
        F, mask_f = cv2.findFundamentalMat(
            pts1f, pts2f,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=f_thresh,
            confidence=f_conf,
        )
    except cv2.error as e:
        logger.debug(f"findFundamentalMat failed: {e}")
        return result

    if F is None or mask_f is None:
        return result

    mask_f = mask_f.ravel().astype(bool)
    n_f = int(mask_f.sum())

    if n_f < min_inliers or n_f / n < min_ratio:
        return result

    result["F"] = F
    result["inlier_mask"] = mask_f

    # ── Essential matrix from calibrated camera ───────────────────────────────
    pts1_in = pts1f[mask_f]
    pts2_in = pts2f[mask_f]

    try:
        E, mask_e = cv2.findEssentialMat(
            pts1_in, pts2_in, K,
            method=cv2.RANSAC,
            prob=e_conf,
            threshold=e_thresh,
        )
    except cv2.error as e:
        logger.debug(f"findEssentialMat failed: {e}")
        return result

    if E is None or mask_e is None:
        return result

    mask_e = mask_e.ravel().astype(bool)
    n_e = int(mask_e.sum())

    if n_e < min_inliers:
        return result

    result["E"] = E

    # Combined inlier mask (in original pts1/pts2 space)
    combined_mask = np.zeros(n, dtype=bool)
    idx_f = np.where(mask_f)[0]
    combined_mask[idx_f[mask_e]] = True
    result["inlier_mask"] = combined_mask
    result["n_inliers"] = int(combined_mask.sum())
    result["inlier_ratio"] = result["n_inliers"] / n

    # ── Recover pose ─────────────────────────────────────────────────────────
    pts1_ee = pts1f[combined_mask]
    pts2_ee = pts2f[combined_mask]

    try:
        n_pos, R, t, mask_rp = cv2.recoverPose(E, pts1_ee, pts2_ee, K)
    except cv2.error as e:
        logger.debug(f"recoverPose failed: {e}")
        return result

    if R is None or t is None:
        return result

    result["R"] = R
    result["t"] = t.ravel()
    result["success"] = True

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Verify all pairs
# ─────────────────────────────────────────────────────────────────────────────

def verify_all_pairs(
    matches: Dict,
    cameras: Dict,
    cfg: Dict,
    image_names: List[str],
) -> Dict:
    """
    Run geometric verification on all matched pairs.
    Updates match dicts in-place and returns verified_pairs dict.
    """
    verified = {}
    total = len(matches)
    n_ok = 0

    # Use K from first registered camera (shared intrinsics assumption)
    # In a multi-camera system each pair would use per-camera K
    K = None
    for nm in image_names:
        cam = cameras.get(nm)
        if cam:
            K = cam.K
            break
    if K is None:
        logger.error("No camera found for geometric verification!")
        return verified

    logger.info(f"[INFO] Geometric verification of {total} pairs...")

    for (n1, n2), info in matches.items():
        pts1 = info["pts1"]
        pts2 = info["pts2"]

        if len(pts1) < 8:
            continue

        # Use pair-specific K if cameras differ; use shared K otherwise
        K1 = cameras.get(n1, cameras.get(list(cameras.keys())[0])).K
        # Use average K for fundamental/essential (shared camera assumption)
        res = verify_pair(pts1, pts2, K1, cfg)

        if not res["success"]:
            continue

        # Update match info
        mask = res["inlier_mask"]
        info["inlier_mask"] = mask
        info["n_verified"] = res["n_inliers"]
        info["inlier_ratio"] = res["inlier_ratio"]
        info["R_rel"] = res["R"]
        info["t_rel"] = res["t"]
        info["F"] = res["F"]
        info["E"] = res["E"]

        verified[(n1, n2)] = info
        n_ok += 1

    logger.info(f"[INFO] Geometric verification: {n_ok}/{total} pairs verified")
    return verified


# ─────────────────────────────────────────────────────────────────────────────
# Reprojection error
# ─────────────────────────────────────────────────────────────────────────────

def reprojection_error(pts2d: np.ndarray, pts3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute per-point reprojection error."""
    if len(pts3d) == 0:
        return np.array([])
    X_cam = (R @ pts3d.T).T + t            # (N, 3)
    valid = X_cam[:, 2] > 1e-6
    uvw = (K @ X_cam.T).T                  # (N, 3)
    proj = np.full((len(pts3d), 2), np.nan)
    proj[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    errors = np.linalg.norm(proj - pts2d, axis=1)
    errors[~valid] = np.inf
    return errors
