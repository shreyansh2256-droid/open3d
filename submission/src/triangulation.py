"""
triangulation.py — DLT triangulation, initial pair, point filtering.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from .camera import Camera
from .geometry import reprojection_error

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DLT Triangulation
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """
    Triangulate (N,2) point correspondences using DLT.
    Returns (N, 3) array of 3D points (may contain invalid points).
    """
    if len(pts1) == 0:
        return np.empty((0, 3))

    pts1h = pts1.T.astype(np.float64)   # (2, N)
    pts2h = pts2.T.astype(np.float64)

    pts4d = cv2.triangulatePoints(P1.astype(np.float64), P2.astype(np.float64), pts1h, pts2h)
    # Homogeneous → Euclidean
    pts4d /= pts4d[3:4, :]
    pts3d = pts4d[:3, :].T              # (N, 3)
    return pts3d


def filter_triangulated_points(
    pts3d: np.ndarray,
    pts2d_ref: np.ndarray,
    pts2d_other: np.ndarray,
    cam_ref: Camera,
    cam_other: Camera,
    cfg: Dict,
) -> np.ndarray:
    """
    Return boolean mask of valid triangulated points.
    Rejects: negative depth, large reprojection error, extreme depths.
    """
    tri_cfg = cfg.get("triangulation", {})
    max_reproj = tri_cfg.get("max_reprojection_error", 4.0)
    min_depth  = tri_cfg.get("min_depth", 0.01)
    max_depth  = tri_cfg.get("max_depth", 1000.0)
    min_par    = tri_cfg.get("min_parallax_degrees", 1.0)

    N = len(pts3d)
    if N == 0:
        return np.zeros(0, dtype=bool)

    mask = np.ones(N, dtype=bool)

    # ── Depth check in both cameras ──────────────────────────────────────────
    for cam in [cam_ref, cam_other]:
        if cam.R is None:
            continue
        X_cam = (cam.R @ pts3d.T).T + cam.t  # (N, 3)
        depth = X_cam[:, 2]
        mask &= (depth > min_depth) & (depth < max_depth)

    # ── Reprojection error ────────────────────────────────────────────────────
    if cam_ref.R is not None:
        err1 = reprojection_error(pts2d_ref, pts3d, cam_ref.R, cam_ref.t, cam_ref.K)
        mask &= (err1 < max_reproj)
    if cam_other.R is not None:
        err2 = reprojection_error(pts2d_other, pts3d, cam_other.R, cam_other.t, cam_other.K)
        mask &= (err2 < max_reproj)

    # ── Parallax check ────────────────────────────────────────────────────────
    if cam_ref.R is not None and cam_other.R is not None:
        c1 = cam_ref.camera_center()
        c2 = cam_other.camera_center()
        v1 = pts3d - c1  # (N,3)
        v2 = pts3d - c2
        cos_angle = (v1 * v2).sum(axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-10
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        mask &= (angle_deg > min_par)

    # ── Finite check ──────────────────────────────────────────────────────────
    mask &= np.all(np.isfinite(pts3d), axis=1)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Initial pair selection
# ─────────────────────────────────────────────────────────────────────────────

def score_pair_for_init(info: Dict) -> float:
    """
    Score a verified pair for use as initial pair.
    Balances inlier count and inlier ratio (not just raw count).
    """
    n_ver = info.get("n_verified", 0)
    ratio = info.get("inlier_ratio", 0.0)
    # Prefer pairs with many inliers AND good ratio
    return n_ver * (0.5 + 0.5 * ratio)


def select_initial_pair(
    verified: Dict,
    cfg: Dict,
    image_names: List[str],
) -> Optional[Tuple[str, str]]:
    """Choose the best initial camera pair for SfM seed."""
    init_cfg = cfg.get("init", {})
    min_inliers = init_cfg.get("min_initial_inliers", 50)
    top_n = init_cfg.get("max_initial_score_pairs", 10)

    # Filter pairs with enough inliers
    candidates = [(k, v) for k, v in verified.items()
                  if v.get("n_verified", 0) >= min_inliers
                  and v.get("R_rel") is not None]

    if not candidates:
        logger.warning("No candidate pair meets min_initial_inliers threshold; relaxing...")
        candidates = [(k, v) for k, v in verified.items() if v.get("R_rel") is not None]
        if not candidates:
            return None

    # Sort by composite score
    candidates.sort(key=lambda x: score_pair_for_init(x[1]), reverse=True)
    top = candidates[:top_n]

    # Among top, prefer pairs that are not too close in index (have parallax)
    best_pair = None
    best_score = -1
    for (n1, n2), info in top:
        i1 = image_names.index(n1) if n1 in image_names else 0
        i2 = image_names.index(n2) if n2 in image_names else 0
        idx_sep = abs(i2 - i1)
        # Mild preference for pairs separated by some frames
        sep_bonus = min(idx_sep / 5.0, 1.0)
        score = score_pair_for_init(info) * (0.8 + 0.2 * sep_bonus)
        if score > best_score:
            best_score = score
            best_pair = (n1, n2)

    if best_pair:
        logger.info(f"[INFO] Initial pair: {best_pair[0]} <-> {best_pair[1]}"
                    f"  (inliers={verified[best_pair]['n_verified']})")
    return best_pair
