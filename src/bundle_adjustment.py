"""
bundle_adjustment.py — Optional bundle adjustment using SciPy least_squares.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares

from .sfm import SfMState
from .camera import Camera

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter packing/unpacking
# ─────────────────────────────────────────────────────────────────────────────

def pack_params(cameras: List[Camera], points: np.ndarray) -> np.ndarray:
    """Pack camera params (rvec 3, tvec 3 per camera) + point XYZs into 1D."""
    import cv2
    cam_params = []
    for cam in cameras:
        if cam.R is None:
            rvec = np.zeros(3)
        else:
            rvec, _ = cv2.Rodrigues(cam.R)
            rvec = rvec.ravel()
        tvec = cam.t.ravel() if cam.t is not None else np.zeros(3)
        cam_params.append(np.concatenate([rvec, tvec]))
    cam_arr = np.concatenate(cam_params)  # (n_cams * 6,)
    pt_arr = points.ravel()               # (n_pts * 3,)
    return np.concatenate([cam_arr, pt_arr])


def unpack_params(params: np.ndarray, n_cams: int, n_pts: int):
    import cv2
    cam_data = params[: n_cams * 6].reshape(n_cams, 6)
    Rs = []
    ts = []
    for c in cam_data:
        R, _ = cv2.Rodrigues(c[:3])
        Rs.append(R)
        ts.append(c[3:6])
    pts = params[n_cams * 6:].reshape(n_pts, 3)
    return Rs, ts, pts


# ─────────────────────────────────────────────────────────────────────────────
# Residuals
# ─────────────────────────────────────────────────────────────────────────────

def ba_residuals(params, n_cams, n_pts, Ks, cam_indices, pt_indices, pts2d_obs):
    """Residual function for bundle adjustment."""
    import cv2
    Rs, ts, pts3d = unpack_params(params, n_cams, n_pts)
    residuals = []
    for ci, pi, obs in zip(cam_indices, pt_indices, pts2d_obs):
        R = Rs[ci]
        t = ts[ci]
        K = Ks[ci]
        X = pts3d[pi]
        X_cam = R @ X + t
        if X_cam[2] < 1e-6:
            residuals.extend([1000.0, 1000.0])
            continue
        uvw = K @ X_cam
        u = uvw[0] / uvw[2]
        v = uvw[1] / uvw[2]
        residuals.extend([u - obs[0], v - obs[1]])
    return np.array(residuals)


# ─────────────────────────────────────────────────────────────────────────────
# Build observation index
# ─────────────────────────────────────────────────────────────────────────────

def build_ba_observations(
    state: SfMState,
    features: Dict,
    max_points: int = 5000,
) -> Optional[Tuple]:
    """Build arrays needed for bundle adjustment."""
    reg_cams = [cam for cam in state.cameras.values() if cam.registered]
    if len(reg_cams) < 2:
        return None

    cam_name_to_idx = {cam.image_name: i for i, cam in enumerate(reg_cams)}
    Ks = [cam.K for cam in reg_cams]

    # Collect observations
    lm_ids = list(state.landmarks.keys())
    if len(lm_ids) > max_points:
        import random
        lm_ids = random.sample(lm_ids, max_points)

    lm_id_to_idx = {lm_id: i for i, lm_id in enumerate(lm_ids)}
    points3d = np.array([state.landmarks[lm_id].xyz for lm_id in lm_ids])

    cam_indices = []
    pt_indices = []
    pts2d_obs = []

    for lm_id in lm_ids:
        lm = state.landmarks[lm_id]
        pi = lm_id_to_idx[lm_id]
        for img_name, kp_idx in lm.observations.items():
            ci = cam_name_to_idx.get(img_name)
            if ci is None:
                continue
            kps = features.get(img_name, {}).get("keypoints")
            if kps is None or kp_idx >= len(kps):
                continue
            pt2d = kps[kp_idx]
            cam_indices.append(ci)
            pt_indices.append(pi)
            pts2d_obs.append(pt2d)

    if len(pts2d_obs) < 10:
        return None

    return (reg_cams, Ks, cam_indices, pt_indices,
            np.array(pts2d_obs), points3d, lm_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Run bundle adjustment
# ─────────────────────────────────────────────────────────────────────────────

def run_bundle_adjustment(
    state: SfMState,
    features: Dict,
    cfg: Dict,
) -> bool:
    """
    Run bundle adjustment. Updates camera poses and 3D point positions in state.
    Returns True if BA converged successfully.
    """
    ba_cfg = cfg.get("bundle_adjustment", {})
    if not ba_cfg.get("enabled", False):
        logger.info("[INFO] Bundle adjustment disabled")
        return False

    max_pts = ba_cfg.get("max_points_for_ba", 5000)
    max_iter = ba_cfg.get("max_iterations", 50)
    ftol = ba_cfg.get("ftol", 1e-4)
    xtol = ba_cfg.get("xtol", 1e-4)
    gtol = ba_cfg.get("gtol", 1e-8)
    loss_fn = ba_cfg.get("loss_function", "huber")

    logger.info(f"[INFO] Running bundle adjustment (max_points={max_pts})...")
    t0 = time.time()

    obs = build_ba_observations(state, features, max_pts)
    if obs is None:
        logger.warning("[BA] Not enough observations for bundle adjustment")
        return False

    reg_cams, Ks, cam_indices, pt_indices, pts2d_obs, points3d, lm_ids = obs
    n_cams = len(reg_cams)
    n_pts = len(lm_ids)

    logger.info(f"[BA] {n_cams} cameras, {n_pts} points, {len(pts2d_obs)} observations")

    x0 = pack_params(reg_cams, points3d)

    # Pre-BA error
    r0 = ba_residuals(x0, n_cams, n_pts, Ks, cam_indices, pt_indices, pts2d_obs)
    pre_err = np.sqrt(np.mean(r0**2))
    logger.info(f"[BA] Pre-BA RMS: {pre_err:.3f} px")

    try:
        result = least_squares(
            ba_residuals,
            x0,
            args=(n_cams, n_pts, Ks, cam_indices, pt_indices, pts2d_obs),
            method="trf",
            loss=loss_fn,
            ftol=ftol, xtol=xtol, gtol=gtol,
            max_nfev=max_iter * len(x0),
            verbose=0,
        )
        success = result.success or result.cost < 1e6
    except Exception as e:
        logger.warning(f"[BA] least_squares failed: {e}")
        return False

    if not success:
        logger.warning(f"[BA] Bundle adjustment did not converge: {result.message}")
        return False

    post_err = np.sqrt(np.mean(result.fun**2))
    logger.info(f"[BA] Post-BA RMS: {post_err:.3f} px  (improvement: {pre_err-post_err:.3f})")
    logger.info(f"[BA] Time: {time.time()-t0:.1f}s")

    # Update state
    import cv2
    Rs, ts, pts3d_opt = unpack_params(result.x, n_cams, n_pts)
    for i, cam in enumerate(reg_cams):
        cam.set_pose(Rs[i], ts[i])
    for i, lm_id in enumerate(lm_ids):
        state.landmarks[lm_id].xyz = pts3d_opt[i]

    return True
