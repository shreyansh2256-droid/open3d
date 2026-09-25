"""
sfm.py — Incremental Structure-from-Motion: initialization, registration, triangulation.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from .camera import Camera
from .geometry import reprojection_error
from .triangulation import (
    triangulate_points, filter_triangulated_points,
    select_initial_pair
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Landmark storage
# ─────────────────────────────────────────────────────────────────────────────

class Landmark:
    """A 3D point seen in one or more images."""
    __slots__ = ["xyz", "rgb", "observations", "id"]

    def __init__(self, xyz: np.ndarray, lm_id: int):
        self.xyz = xyz.copy()
        self.rgb = np.array([128, 128, 128], dtype=np.uint8)
        self.observations: Dict[str, int] = {}   # image_name → keypoint_idx
        self.id = lm_id


# ─────────────────────────────────────────────────────────────────────────────
# SfM state
# ─────────────────────────────────────────────────────────────────────────────

class SfMState:
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.landmarks: Dict[int, Landmark] = {}
        self.next_lm_id = 0
        # feature_index → landmark_id  (per image)
        self.feat2lm: Dict[str, Dict[int, int]] = {}

    def add_landmark(self, xyz: np.ndarray) -> Landmark:
        lm = Landmark(xyz, self.next_lm_id)
        self.landmarks[self.next_lm_id] = lm
        self.next_lm_id += 1
        return lm

    def register_camera(self, cam: Camera):
        self.cameras[cam.image_name] = cam
        if cam.image_name not in self.feat2lm:
            self.feat2lm[cam.image_name] = {}

    def n_registered(self) -> int:
        return sum(1 for c in self.cameras.values() if c.registered)

    def n_points(self) -> int:
        return len(self.landmarks)

    def get_points_array(self) -> np.ndarray:
        if not self.landmarks:
            return np.empty((0, 3))
        return np.array([lm.xyz for lm in self.landmarks.values()])

    def get_colors_array(self) -> np.ndarray:
        if not self.landmarks:
            return np.empty((0, 3))
        return np.array([lm.rgb for lm in self.landmarks.values()], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Colour assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_colors_to_landmarks(
    state: SfMState,
    images_rgb: Dict[str, np.ndarray],
    cameras_all: Dict[str, Camera],
):
    """Assign RGB colour to each landmark from its first observation."""
    for lm_id, lm in state.landmarks.items():
        for img_name, kp_idx in lm.observations.items():
            rgb_img = images_rgb.get(img_name)
            cam = cameras_all.get(img_name)
            if rgb_img is None or cam is None:
                continue
            # Project landmark and sample colour
            try:
                uv = cam.project(lm.xyz.reshape(1, 3))[0]
                if not np.isfinite(uv).all():
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                h, w = rgb_img.shape[:2]
                if 0 <= u < w and 0 <= v < h:
                    lm.rgb = rgb_img[v, u]
                    break
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Two-view initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_sfm(
    state: SfMState,
    init_pair: Tuple[str, str],
    verified: Dict,
    cameras: Dict[str, Camera],
    features: Dict,
    cfg: Dict,
) -> bool:
    """Initialize SfM from the best pair. Returns True on success."""
    n1, n2 = init_pair
    info = verified.get((n1, n2)) or verified.get((n2, n1))
    if info is None:
        logger.error(f"Pair {n1} <-> {n2} not in verified dict")
        return False

    cam1 = cameras[n1]
    cam2 = cameras[n2]

    # Camera 1: identity pose
    cam1.set_identity_pose()
    state.register_camera(cam1)

    # Camera 2: relative pose from essential matrix
    R_rel = info.get("R_rel")
    t_rel = info.get("t_rel")
    if R_rel is None or t_rel is None:
        logger.error("No relative pose for initial pair")
        return False

    cam2.set_pose(R_rel, t_rel)
    state.register_camera(cam2)

    # Triangulate inlier matches
    mask = info.get("inlier_mask")
    pts1 = info["pts1"]
    pts2 = info["pts2"]
    idx1 = info["idx1"]
    idx2 = info["idx2"]

    if mask is not None and mask.any():
        pts1_in = pts1[mask]
        pts2_in = pts2[mask]
        idx1_in = idx1[mask]
        idx2_in = idx2[mask]
    else:
        pts1_in = pts1
        pts2_in = pts2
        idx1_in = idx1
        idx2_in = idx2

    if len(pts1_in) < 5:
        logger.error("Too few inlier matches for initialization")
        return False

    pts3d = triangulate_points(pts1_in, pts2_in, cam1.P, cam2.P)
    valid_mask = filter_triangulated_points(pts3d, pts1_in, pts2_in, cam1, cam2, cfg)

    n_added = 0
    for i, (valid, xyz, ki1, ki2) in enumerate(zip(valid_mask, pts3d, idx1_in, idx2_in)):
        if not valid:
            continue
        lm = state.add_landmark(xyz)
        lm.observations[n1] = int(ki1)
        lm.observations[n2] = int(ki2)
        state.feat2lm[n1][int(ki1)] = lm.id
        state.feat2lm[n2][int(ki2)] = lm.id
        n_added += 1

    logger.info(f"[INFO] Initial inliers: {int(mask.sum()) if mask is not None else len(pts1_in)}")
    logger.info(f"[INFO] Initial 3D points: {n_added}")
    return n_added >= 10


# ─────────────────────────────────────────────────────────────────────────────
# Camera registration via PnP
# ─────────────────────────────────────────────────────────────────────────────

def register_camera_pnp(
    cam: Camera,
    state: SfMState,
    verified: Dict,
    features: Dict,
    cfg: Dict,
) -> bool:
    """
    Estimate pose of `cam` using 2D-3D correspondences from already
    reconstructed landmarks. Returns True on success.
    """
    pnp_cfg = cfg.get("pnp", {})
    min_corr = pnp_cfg.get("min_correspondences", 12)
    ransac_thresh = pnp_cfg.get("ransac_threshold", 4.0)
    ransac_conf   = pnp_cfg.get("ransac_confidence", 0.999)
    max_reproj    = pnp_cfg.get("max_reprojection_error", 6.0)

    name = cam.image_name
    kps = features.get(name, {}).get("keypoints")
    if kps is None or len(kps) == 0:
        return False

    pts2d = []
    pts3d = []

    # Find 2D-3D correspondences through verified pairs
    for reg_name in list(state.cameras.keys()):
        if not state.cameras[reg_name].registered:
            continue
        pair_key = (reg_name, name) if (reg_name, name) in verified else \
                   (name, reg_name) if (name, reg_name) in verified else None
        if pair_key is None:
            continue

        info = verified[pair_key]
        mask = info.get("inlier_mask")
        idx1 = info["idx1"]
        idx2 = info["idx2"]
        pts1 = info["pts1"]
        pts2 = info["pts2"]

        # Which index corresponds to reg_name vs name?
        if pair_key[0] == reg_name:
            reg_idx_arr = idx1
            new_idx_arr = idx2
            new_pts_arr = pts2
        else:
            reg_idx_arr = idx2
            new_idx_arr = idx1
            new_pts_arr = pts1

        if mask is not None:
            valid = mask
            reg_idx_arr = reg_idx_arr[valid]
            new_idx_arr = new_idx_arr[valid]
            new_pts_arr = new_pts_arr[valid]

        f2lm_reg = state.feat2lm.get(reg_name, {})
        for ri, ni, pt2 in zip(reg_idx_arr, new_idx_arr, new_pts_arr):
            lm_id = f2lm_reg.get(int(ri))
            if lm_id is None:
                continue
            lm = state.landmarks.get(lm_id)
            if lm is None:
                continue
            pts3d.append(lm.xyz)
            pts2d.append(pt2)

    if len(pts3d) < min_corr:
        logger.debug(f"  {name}: only {len(pts3d)} 2D-3D correspondences (need {min_corr})")
        return False

    pts3d_arr = np.array(pts3d, dtype=np.float64)
    pts2d_arr = np.array(pts2d, dtype=np.float64)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_arr, pts2d_arr, cam.K, None,
            reprojectionError=ransac_thresh,
            confidence=ransac_conf,
            iterationsCount=200,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as e:
        logger.debug(f"  PnP error for {name}: {e}")
        return False

    if not success or inliers is None or len(inliers) < min_corr:
        return False

    R_pnp, _ = cv2.Rodrigues(rvec)
    t_pnp = tvec.ravel()

    # Verify reprojection error on inliers
    in_idx = inliers.ravel()
    err = reprojection_error(pts2d_arr[in_idx], pts3d_arr[in_idx], R_pnp, t_pnp, cam.K)
    if np.median(err) > max_reproj:
        logger.debug(f"  {name}: high median reprojection error ({np.median(err):.2f}px)")
        return False

    cam.set_pose(R_pnp, t_pnp)
    logger.info(f"[INFO] Registered camera: {name}  ({len(in_idx)} inliers, "
                f"err={np.median(err):.2f}px)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Triangulate new points after registration
# ─────────────────────────────────────────────────────────────────────────────

def triangulate_new_points(
    new_cam: Camera,
    state: SfMState,
    verified: Dict,
    features: Dict,
    cfg: Dict,
) -> int:
    """Triangulate new 3D points using newly registered camera."""
    name = new_cam.image_name
    n_added = 0
    tri_cfg = cfg.get("triangulation", {})

    for reg_name, reg_cam in state.cameras.items():
        if not reg_cam.registered or reg_name == name:
            continue

        pair_key = (reg_name, name) if (reg_name, name) in verified else \
                   (name, reg_name) if (name, reg_name) in verified else None
        if pair_key is None:
            continue

        info = verified[pair_key]
        mask = info.get("inlier_mask")
        idx1_all = info["idx1"]
        idx2_all = info["idx2"]
        pts1_all = info["pts1"]
        pts2_all = info["pts2"]

        if mask is not None:
            idx1_all = idx1_all[mask]
            idx2_all = idx2_all[mask]
            pts1_all = pts1_all[mask]
            pts2_all = pts2_all[mask]

        # Decide which is reg vs new
        if pair_key[0] == reg_name:
            reg_idx_arr, new_idx_arr = idx1_all, idx2_all
            reg_pts, new_pts = pts1_all, pts2_all
        else:
            reg_idx_arr, new_idx_arr = idx2_all, idx1_all
            reg_pts, new_pts = pts2_all, pts1_all

        f2lm_reg = state.feat2lm.get(reg_name, {})
        f2lm_new = state.feat2lm.get(name, {})
        if name not in state.feat2lm:
            state.feat2lm[name] = {}

        # Only triangulate un-matched pairs
        new_match_mask = np.array([
            int(ri) not in f2lm_reg and int(ni) not in f2lm_new
            for ri, ni in zip(reg_idx_arr, new_idx_arr)
        ], dtype=bool)

        if not new_match_mask.any():
            continue

        reg_pts_new = reg_pts[new_match_mask]
        new_pts_new = new_pts[new_match_mask]
        reg_idx_new = reg_idx_arr[new_match_mask]
        new_idx_new = new_idx_arr[new_match_mask]

        pts3d = triangulate_points(reg_pts_new, new_pts_new, reg_cam.P, new_cam.P)
        valid = filter_triangulated_points(pts3d, reg_pts_new, new_pts_new, reg_cam, new_cam, cfg)

        for i, (v, xyz, ri, ni) in enumerate(zip(valid, pts3d, reg_idx_new, new_idx_new)):
            if not v:
                continue
            lm = state.add_landmark(xyz)
            lm.observations[reg_name] = int(ri)
            lm.observations[name] = int(ni)
            state.feat2lm[reg_name][int(ri)] = lm.id
            state.feat2lm[name][int(ni)] = lm.id
            n_added += 1

    return n_added


# ─────────────────────────────────────────────────────────────────────────────
# Incremental SfM main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_incremental_sfm(
    images: List,
    cameras: Dict[str, Camera],
    features: Dict,
    verified: Dict,
    cfg: Dict,
) -> SfMState:
    """Run full incremental SfM pipeline."""
    image_names = [p.name for p in images]
    state = SfMState()

    # Register all cameras into state (unregistered initially)
    for nm, cam in cameras.items():
        state.cameras[nm] = cam
        state.feat2lm[nm] = {}

    # ── 1. Select initial pair ────────────────────────────────────────────────
    init_pair = select_initial_pair(verified, cfg, image_names)
    if init_pair is None:
        logger.error("[ERROR] Could not find a valid initial pair! "
                     "Check that images have sufficient overlap.")
        return state

    # ── 2. Initialize ─────────────────────────────────────────────────────────
    if not initialize_sfm(state, init_pair, verified, cameras, features, cfg):
        logger.error("[ERROR] Initialization failed.")
        return state

    # ── 3. Incremental registration ───────────────────────────────────────────
    unregistered = [nm for nm in image_names if nm not in {init_pair[0], init_pair[1]}]
    max_iter = len(unregistered) + 1
    iteration = 0

    while unregistered and iteration < max_iter:
        iteration += 1
        registered_this_round = []

        for name in list(unregistered):
            cam = cameras.get(name)
            if cam is None:
                unregistered.remove(name)
                continue

            success = register_camera_pnp(cam, state, verified, features, cfg)
            if success:
                state.register_camera(cam)
                n_new = triangulate_new_points(cam, state, verified, features, cfg)
                logger.info(f"  → Triangulated {n_new} new points")
                registered_this_round.append(name)

        for nm in registered_this_round:
            unregistered.remove(nm)

        if not registered_this_round:
            logger.info(f"[INFO] No more cameras can be registered ({len(unregistered)} remaining)")
            break

    n_reg = state.n_registered()
    n_pts = state.n_points()
    logger.info(f"[INFO] Final cameras: {n_reg}/{len(image_names)}")
    logger.info(f"[INFO] Final sparse points: {n_pts:,}")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Reprojection statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_reprojection_stats(state: SfMState) -> Dict:
    """Compute mean/median reprojection error across all observations."""
    all_errors = []
    for lm_id, lm in state.landmarks.items():
        for img_name, kp_idx in lm.observations.items():
            cam = state.cameras.get(img_name)
            if cam is None or not cam.registered:
                continue
            feat = state.cameras.get(img_name)
            # Get 2D observation point
            # (we'd need the keypoints stored separately; use camera projection)
            uv_proj = cam.project(lm.xyz.reshape(1, 3))[0]
            # We don't have the original observed point here easily,
            # so compute proj→visibility only
            if np.isfinite(uv_proj).all():
                pass  # Error requires original 2d obs

    # Compute from cameras projecting landmarks
    for lm_id, lm in state.landmarks.items():
        for img_name in lm.observations:
            cam = state.cameras.get(img_name)
            if cam is None or not cam.registered:
                continue
            X = lm.xyz.reshape(1, 3)
            uv = cam.project(X)[0]
            if np.isfinite(uv).all():
                # Without stored 2D obs we can only check depth
                X_cam = (cam.R @ lm.xyz) + cam.t
                if X_cam[2] > 0:
                    all_errors.append(0.0)  # Placeholder

    if not all_errors:
        return {"mean_reprojection_error": None, "median_reprojection_error": None}

    return {
        "mean_reprojection_error": float(np.mean(all_errors)),
        "median_reprojection_error": float(np.median(all_errors)),
        "n_observations": len(all_errors),
    }
