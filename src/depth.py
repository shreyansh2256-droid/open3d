"""
depth.py — Fast semi-dense depth estimation using vectorised OpenCV operations.

Strategy:
  For each reference image + neighbours, compute a depth map by:
  1. Computing the homography-warped depth-plane sweep (plane-sweep stereo)
  2. Using OpenCV's matchTemplate (NCC) vectorised across the full image
  3. Selecting the depth hypothesis with best multi-view score
  4. Filling gaps with bilateral-filter smoothing

This is ~100x faster than pixel-loop NCC.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .camera import Camera
from .io_utils import ensure_dir, load_image_rgb

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Depth estimation via semi-global matching (SGM via OpenCV StereoBM/StereoSGBM)
# fallback: plane-sweep with OpenCV matchTemplate
# ─────────────────────────────────────────────────────────────────────────────

def compute_homography_plane(
    K1: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
    depth: float,
    normal: np.ndarray = None,
) -> np.ndarray:
    """
    Compute homography mapping ref image pixels (at given depth) to neighbour image.
    H = K2 * (R_rel + t_rel * n^T / depth) * K1_inv
    """
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    H = K2 @ (R_rel + np.outer(t_rel, normal) / depth) @ np.linalg.inv(K1)
    return H


def warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp image using homography."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def compute_zncc_image(ref: np.ndarray, warped: np.ndarray, win: int = 9) -> np.ndarray:
    """
    Compute per-pixel Zero-Mean NCC between ref and warped images.
    Uses integral images for O(1) per pixel cost.
    Returns a float32 score map in [-1, 1].
    """
    ref_f   = ref.astype(np.float32)
    warp_f  = warped.astype(np.float32)

    # Local mean via boxFilter
    mu1  = cv2.boxFilter(ref_f,   -1, (win, win))
    mu2  = cv2.boxFilter(warp_f,  -1, (win, win))

    ref_c  = ref_f  - mu1
    warp_c = warp_f - mu2

    sigma1_sq = cv2.boxFilter(ref_c  * ref_c,  -1, (win, win))
    sigma2_sq = cv2.boxFilter(warp_c * warp_c, -1, (win, win))
    sigma12   = cv2.boxFilter(ref_c  * warp_c, -1, (win, win))

    eps = 1e-6
    zncc = sigma12 / (np.sqrt(sigma1_sq * sigma2_sq) + eps)
    return np.clip(zncc, -1.0, 1.0)


def estimate_depth_for_image(
    ref_name: str,
    neighbor_names: List[str],
    cameras: Dict[str, Camera],
    images_gray: Dict[str, np.ndarray],
    cfg: Dict,
) -> Optional[np.ndarray]:
    """
    Estimate a depth map for `ref_name` using plane-sweep with vectorised ZNCC.
    Returns depth array (H, W), 0 = invalid.
    """
    depth_cfg    = cfg.get("depth", {})
    n_hyp        = depth_cfg.get("num_depth_hypotheses", 32)
    win          = depth_cfg.get("patch_size", 9)
    ncc_thresh   = depth_cfg.get("ncc_threshold", 0.2)
    d_min        = float(depth_cfg.get("d_min", 0.5))
    d_max        = float(depth_cfg.get("d_max", 50.0))

    cam_ref  = cameras.get(ref_name)
    ref_gray = images_gray.get(ref_name)
    if cam_ref is None or not cam_ref.registered or ref_gray is None:
        return None

    K1  = cam_ref.K
    R1  = cam_ref.R
    t1  = cam_ref.t

    # Depth range
    depth_hyps = np.linspace(d_min, d_max, n_hyp)

    H_img, W_img = ref_gray.shape

    # Accumulate ZNCC score across hypotheses
    best_score = np.full((H_img, W_img), -1.0, dtype=np.float32)
    best_depth = np.zeros((H_img, W_img), dtype=np.float32)

    # Collect valid neighbour cameras
    nei_list = []
    for nm in neighbor_names:
        cam_n = cameras.get(nm)
        img_n = images_gray.get(nm)
        if cam_n is not None and cam_n.registered and img_n is not None:
            nei_list.append((nm, cam_n, img_n))

    if not nei_list:
        return None

    for d in depth_hyps:
        # Average ZNCC across all neighbours
        agg_score = np.zeros((H_img, W_img), dtype=np.float32)
        valid_count = 0

        for nm, cam_n, img_n in nei_list:
            K2 = cam_n.K
            R2 = cam_n.R
            t2 = cam_n.t

            # Need to match image sizes
            if img_n.shape != (H_img, W_img):
                img_n_rs = cv2.resize(img_n, (W_img, H_img))
            else:
                img_n_rs = img_n

            try:
                H_hom = compute_homography_plane(K1, R1, t1, K2, R2, t2, d)
                warped = warp_image(img_n_rs, H_hom)
                # Only score where warped is non-zero (valid warp)
                warped_valid = (warped > 0).astype(np.float32)
                zncc = compute_zncc_image(ref_gray, warped, win)
                zncc *= warped_valid  # Zero out invalid regions
                agg_score += zncc
                valid_count += 1
            except Exception as e:
                logger.debug("Homography computation failed for depth %.2f: %s", d, e)
                continue

        if valid_count == 0:
            continue

        agg_score /= valid_count

        # Winner-takes-all depth selection
        better = agg_score > best_score
        best_score[better] = agg_score[better]
        best_depth[better] = d

    # Threshold by minimum NCC score
    valid_mask = best_score >= ncc_thresh
    depth_map = np.where(valid_mask, best_depth, 0.0).astype(np.float32)

    # Diagnose depth quality BEFORE fill
    n_valid_pre_fill = valid_mask.sum()
    pct_valid_pre_fill = 100.0 * n_valid_pre_fill / valid_mask.size
    if n_valid_pre_fill > 0:
        d_vals = depth_map[valid_mask]
        pct_at_dmin = 100.0 * (d_vals <= d_min + 0.05).sum() / len(d_vals)
        pct_at_dmax = 100.0 * (d_vals >= d_max - 0.05).sum() / len(d_vals)
        logger.debug(
            "  [%s] Pre-fill: valid=%.1f%%  dmin=%.1f%%  dmax=%.1f%%  "
            "median=%.2f  std=%.2f",
            ref_name, pct_valid_pre_fill, pct_at_dmin, pct_at_dmax,
            float(np.median(d_vals)), float(d_vals.std()))
        if pct_at_dmin > 50:
            logger.warning("  [%s] WARNING: >50%% pixels at d_min=%.1f (d_range may be wrong)",
                           ref_name, d_min)
        if pct_at_dmax > 50:
            logger.warning("  [%s] WARNING: >50%% pixels at d_max=%.1f (d_range may be wrong)",
                           ref_name, d_max)

    # Fill gaps (only extend measured values, don't synthesize new geometry)
    if depth_cfg.get("depth_fill", True):
        depth_map = fill_depth(depth_map, valid_mask)

    return depth_map


def fill_depth(depth_map: np.ndarray,
               valid_mask: np.ndarray = None) -> np.ndarray:
    """Fill invalid (zero) depth pixels via morphological dilation from valid neighbours."""
    filled = depth_map.copy()
    if valid_mask is not None and not valid_mask.any():
        return filled   # Nothing to fill from

    kernel = np.ones((5, 5), np.uint8)
    for _ in range(6):
        dilated = cv2.dilate(filled, kernel)
        filled = np.where(filled == 0, dilated, filled)

    # Smooth with bilateral filter to preserve edges
    if filled.max() > 0:
        filled_u8 = np.clip(filled / filled.max() * 255, 0, 255).astype(np.uint8)
        smoothed_u8 = cv2.bilateralFilter(filled_u8, 9, 75, 75)
        filled = smoothed_u8.astype(np.float32) / 255.0 * filled.max()
    return filled


# ─────────────────────────────────────────────────────────────────────────────
# Choose reference images
# ─────────────────────────────────────────────────────────────────────────────

def choose_reference_images(
    state,
    image_names: List[str],
    max_depth_images: int,
    n_neighbors: int,
) -> List[Tuple[str, List[str]]]:
    """Choose reference images and their neighbours."""
    registered_names = [nm for nm in image_names
                        if state.cameras.get(nm) and state.cameras[nm].registered]

    if not registered_names:
        return []

    # Score cameras by landmark visibility
    lm_per_cam = {nm: 0 for nm in registered_names}
    for lm in state.landmarks.values():
        for nm in lm.observations:
            if nm in lm_per_cam:
                lm_per_cam[nm] += 1

    sorted_cams = sorted(registered_names, key=lambda x: lm_per_cam.get(x, 0), reverse=True)

    if max_depth_images > 0:
        step = max(1, len(sorted_cams) // max_depth_images)
        refs = sorted_cams[::step][:max_depth_images]
    else:
        refs = sorted_cams

    results = []
    for ref in refs:
        ref_idx = registered_names.index(ref) if ref in registered_names else 0
        neighbors = []
        for offset in range(1, len(registered_names)):
            for sign in [1, -1]:
                ni = ref_idx + sign * offset
                if 0 <= ni < len(registered_names) and registered_names[ni] != ref:
                    neighbors.append(registered_names[ni])
                if len(neighbors) >= n_neighbors:
                    break
            if len(neighbors) >= n_neighbors:
                break
        if neighbors:
            results.append((ref, neighbors[:n_neighbors]))

    return results


def load_image_gray_direct(img_path: Path, max_dim: int = 0) -> Optional[np.ndarray]:
    """Load and optionally resize a grayscale image."""
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if max_dim > 0:
            h, w = img.shape
            scale = min(max_dim / max(w, h), 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Run depth estimation for all references
# ─────────────────────────────────────────────────────────────────────────────

def run_depth_estimation(
    images: List[Path],
    state,
    cameras: Dict[str, Camera],
    cfg: Dict,
    output_dir: str,
) -> Dict[str, np.ndarray]:
    """Estimate depth maps for selected reference images."""
    depth_cfg = cfg.get("depth", {})
    if not depth_cfg.get("enabled", True):
        logger.info("[INFO] Depth estimation disabled")
        return {}

    depth_dir   = ensure_dir(os.path.join(output_dir, "depth"))
    preview_dir = ensure_dir(os.path.join(output_dir, "depth_preview"))

    # Use smaller images for depth estimation (faster)
    depth_max_dim = min(cfg.get("image", {}).get("resize_for_processing", 1600), 600)
    max_refs    = depth_cfg.get("max_depth_images", 8)
    n_neighbors = depth_cfg.get("num_neighbors", 2)

    image_names = [p.name for p in images]

    # Load grayscale images at smaller size for depth
    logger.info(f"[INFO] Loading images for depth estimation (max_dim={depth_max_dim})...")
    images_gray: Dict[str, np.ndarray] = {}
    img_path_dict = {p.name: p for p in images}

    for nm in image_names:
        if not (state.cameras.get(nm) and state.cameras[nm].registered):
            continue
        p = img_path_dict.get(nm)
        if p:
            gray = load_image_gray_direct(p, depth_max_dim)
            if gray is not None:
                images_gray[nm] = gray

    # Scale camera intrinsics to match smaller images
    scaled_cameras: Dict[str, Camera] = {}
    for nm, cam in cameras.items():
        if nm not in images_gray:
            continue
        gray = images_gray[nm]
        dh, dw = gray.shape
        # Compute scale from original camera width
        scale_x = dw / cam.width
        scale_y = dh / cam.height
        from .camera import Camera as Cam
        c = Cam(
            image_name=cam.image_name,
            width=dw, height=dh,
            fx=cam.fx * scale_x, fy=cam.fy * scale_y,
            cx=cam.cx * scale_x, cy=cam.cy * scale_y,
        )
        c.R = cam.R
        c.t = cam.t
        c.registered = cam.registered
        scaled_cameras[nm] = c

    refs_and_neighbors = choose_reference_images(state, image_names, max_refs, n_neighbors)
    logger.info(f"[INFO] Estimating depth for {len(refs_and_neighbors)} images "
                f"at {depth_max_dim}px (plane-sweep ZNCC)...")

    depth_maps = {}
    t0 = time.time()

    for i, (ref_name, nei_names) in enumerate(refs_and_neighbors):
        logger.info(f"  [{i+1}/{len(refs_and_neighbors)}] Depth: {ref_name} "
                    f"(neighbors: {nei_names})")
        t_ref = time.time()

        depth_map = estimate_depth_for_image(
            ref_name, nei_names, scaled_cameras, images_gray, cfg
        )

        if depth_map is None or depth_map.max() == 0:
            logger.warning(f"  Depth estimation failed/empty for {ref_name}")
            continue

        valid_pct = 100.0 * (depth_map > 0).sum() / depth_map.size
        logger.info(f"  → depth range [{depth_map[depth_map>0].min():.2f}, "
                    f"{depth_map[depth_map>0].max():.2f}], "
                    f"valid={valid_pct:.1f}% ({time.time()-t_ref:.1f}s)")

        depth_maps[ref_name] = depth_map

        # Save raw .npy
        stem = Path(ref_name).stem
        npy_path = str(depth_dir / f"depth_map_{i+1:02d}_{stem}.npy")
        np.save(npy_path, depth_map)

        # Save PNG (16-bit)
        png_path = str(depth_dir / f"depth_map_{i+1:02d}_{stem}.png")
        _save_depth_png(depth_map, png_path)

        # Save preview
        prev_path = str(preview_dir / f"depth_preview_{i+1:02d}_{stem}.png")
        _save_depth_preview(depth_map, ref_name, prev_path)

    logger.info(f"[INFO] Depth estimation done: {len(depth_maps)} maps in {time.time()-t0:.1f}s")
    return depth_maps


def _save_depth_png(depth: np.ndarray, path: str):
    """Save normalised depth as 16-bit PNG."""
    valid = depth > 0
    if not valid.any():
        return
    d_min = depth[valid].min()
    d_max = depth[valid].max()
    if d_max == d_min:
        return
    norm = np.zeros_like(depth, dtype=np.uint16)
    norm[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    cv2.imwrite(path, norm)


def _save_depth_preview(depth: np.ndarray, ref_name: str, path: str):
    """Save a colourised depth preview PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid = depth > 0
        if not valid.any():
            return

        d_vis = depth.copy().astype(np.float32)
        d_vis[~valid] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(d_vis, cmap="plasma", aspect="auto")
        plt.colorbar(im, ax=ax, label="Relative depth (scene units)")
        ax.set_title(f"Depth map: {ref_name}\nValid: {100*valid.mean():.1f}%")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.debug(f"Depth preview error: {e}")
