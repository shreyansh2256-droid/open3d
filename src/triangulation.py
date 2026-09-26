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
    depth_fails = 0
    for cam in [cam_ref, cam_other]:
        if cam.R is None:
            continue
        X_cam = (cam.R @ pts3d.T).T + cam.t  # (N, 3)
        depth = X_cam[:, 2]
        dmask = (depth > min_depth) & (depth < max_depth)
        depth_fails += np.sum(~dmask)
        mask &= dmask

    # ── Reprojection error ────────────────────────────────────────────────────
    reproj_fails = 0
    if cam_ref.R is not None:
        err1 = reprojection_error(pts2d_ref, pts3d, cam_ref.R, cam_ref.t, cam_ref.K)
        rmask1 = (err1 < max_reproj)
        reproj_fails += np.sum(~rmask1)
        mask &= rmask1
    if cam_other.R is not None:
        err2 = reprojection_error(pts2d_other, pts3d, cam_other.R, cam_other.t, cam_other.K)
        rmask2 = (err2 < max_reproj)
        reproj_fails += np.sum(~rmask2)
        mask &= rmask2

    # ── Parallax check ────────────────────────────────────────────────────────
    par_fails = 0
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
        pmask = (angle_deg > min_par)
        par_fails += np.sum(~pmask)
        mask &= pmask

    # ── Finite check ──────────────────────────────────────────────────────────
    fmask = np.all(np.isfinite(pts3d), axis=1)
    finite_fails = np.sum(~fmask)
    mask &= fmask

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
    cameras: Optional[Dict] = None,
) -> Optional[Tuple[str, str]]:
    """Choose the best initial camera pair for SfM seed by actively testing triangulation."""
    init_cfg = cfg.get("init", {})
    min_inliers = init_cfg.get("min_initial_inliers", 30)
    top_n = init_cfg.get("max_initial_score_pairs", 20)

    candidates = [(k, v) for k, v in verified.items()
                  if v.get("n_verified", 0) >= min_inliers
                  and v.get("R_rel") is not None]

    if not candidates:
        logger.warning("No candidate pair meets min_initial_inliers threshold; relaxing...")
        candidates = [(k, v) for k, v in verified.items() if v.get("R_rel") is not None]
        if not candidates:
            return None

    candidates.sort(key=lambda x: score_pair_for_init(x[1]), reverse=True)
    top = candidates[:top_n]

    best_pair = None
    best_valid = -1

    for (n1, n2), info in top:
        if cameras and n1 in cameras and n2 in cameras:
            # Test triangulation!
            cam1, cam2 = cameras[n1], cameras[n2]
            
            # Temporary setup
            old_R1, old_t1, old_reg1 = cam1.R, cam1.t, getattr(cam1, 'registered', False)
            old_R2, old_t2, old_reg2 = cam2.R, cam2.t, getattr(cam2, 'registered', False)
            
            cam1.set_identity_pose()
            cam2.set_pose(info["R_rel"], info["t_rel"])
            
            mask = info.get("inlier_mask")
            pts1, pts2 = info["pts1"], info["pts2"]
            if mask is not None and mask.any():
                pts1_in, pts2_in = pts1[mask], pts2[mask]
            else:
                pts1_in, pts2_in = pts1, pts2
                
            pts3d = triangulate_points(pts1_in, pts2_in, cam1.P, cam2.P)
            valid_mask = filter_triangulated_points(pts3d, pts1_in, pts2_in, cam1, cam2, cfg)
            n_valid = int(valid_mask.sum())
            
            # Restore
            cam1.R, cam1.t, cam1.registered = old_R1, old_t1, old_reg1
            cam2.R, cam2.t, cam2.registered = old_R2, old_t2, old_reg2
            
            if n_valid > best_valid:
                best_valid = n_valid
                best_pair = (n1, n2)
        else:
            # Fallback if cameras not passed
            score = score_pair_for_init(info)
            if score > best_valid:
                best_valid = score
                best_pair = (n1, n2)

    if best_pair:
        logger.info(f"[INFO] Initial pair: {best_pair[0]} <-> {best_pair[1]}  (valid_triangulated={best_valid})")
    return best_pair
