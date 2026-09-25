"""
depth.py — Semi-dense depth estimation using sparse-SfM-guided plane sweep.

The dense stage is deliberately constrained by the sparse reconstruction:
  1. choose reference/neighbor cameras with actual shared landmarks;
  2. derive a per-camera depth interval from visible sparse 3D landmarks;
  3. sweep that interval at higher resolution;
  4. keep only pixels with both a good NCC score and a decisive best-vs-second-best margin;
  5. avoid aggressive depth filling so invalid stereo pixels do not become fake geometry.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .camera import Camera
from .io_utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_homography_plane(
    K1: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
    depth: float,
    normal: np.ndarray = None,
) -> np.ndarray:
    """Compute homography mapping ref-image pixels at Z=depth to the neighbour."""
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    H = K2 @ (R_rel + np.outer(t_rel, normal) / depth) @ np.linalg.inv(K1)
    return H


def warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp neighbour image into the reference image coordinate frame."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compute_zncc_image(ref: np.ndarray, warped: np.ndarray, win: int = 9) -> np.ndarray:
    """Compute per-pixel zero-mean normalized cross correlation."""
    ref_f = ref.astype(np.float32)
    warp_f = warped.astype(np.float32)
    mu1 = cv2.boxFilter(ref_f, -1, (win, win))
    mu2 = cv2.boxFilter(warp_f, -1, (win, win))
    ref_c = ref_f - mu1
    warp_c = warp_f - mu2
    sigma1_sq = cv2.boxFilter(ref_c * ref_c, -1, (win, win))
    sigma2_sq = cv2.boxFilter(warp_c * warp_c, -1, (win, win))
    sigma12 = cv2.boxFilter(ref_c * warp_c, -1, (win, win))
    eps = 1e-6
    zncc = sigma12 / (np.sqrt(sigma1_sq * sigma2_sq) + eps)
    return np.clip(zncc, -1.0, 1.0)


def _camera_depth_range_from_landmarks(
    ref_name: str,
    cam: Camera,
    state,
    depth_cfg: Dict,
) -> Tuple[float, float]:
    """Estimate a useful Z-depth interval from sparse SfM landmarks visible in a camera."""
    global_min = float(depth_cfg.get("d_min", 0.5))
    global_max = float(depth_cfg.get("d_max", 80.0))
    depths = []

    for lm in state.landmarks.values():
        if ref_name not in lm.observations:
            continue
        try:
            z = float((cam.R @ lm.xyz + cam.t)[2])
            if np.isfinite(z) and z > 0:
                depths.append(z)
        except Exception:
            continue

    if len(depths) < 20:
        logger.warning(
            "  [%s] only %d sparse depth samples; using configured fallback [%.2f, %.2f]",
            ref_name, len(depths), global_min, global_max,
        )
        return global_min, global_max

    vals = np.asarray(depths, dtype=np.float64)
    p02, p98 = np.percentile(vals, [2.0, 98.0])
    span = max(float(p98 - p02), 1e-3)
    pad = max(0.10 * span, 0.5)

    d_lo = max(global_min, float(p02 - pad))
    d_hi = min(global_max, float(p98 + pad))

    # Never allow a nearly-zero sweep interval.
    if d_hi <= d_lo + 1.0:
        mid = float(np.median(vals))
        half = max(1.0, 0.25 * span)
        d_lo = max(global_min, mid - half)
        d_hi = min(global_max, mid + half)

    logger.info(
        "  [%s] sparse-guided depth range [%.2f, %.2f] from %d landmarks "
        "(p02=%.2f p98=%.2f)",
        ref_name, d_lo, d_hi, len(vals), p02, p98,
    )
    return d_lo, d_hi


def estimate_depth_for_image(
    ref_name: str,
    neighbor_names: List[str],
    cameras: Dict[str, Camera],
    images_gray: Dict[str, np.ndarray],
    cfg: Dict,
    state=None,
) -> Optional[np.ndarray]:
    """Estimate a sparse-SfM-guided depth map using plane-sweep ZNCC."""
    depth_cfg = cfg.get("depth", {})
    n_hyp = int(depth_cfg.get("num_depth_hypotheses", 72))
    win = int(depth_cfg.get("patch_size", 9))
    ncc_thresh = float(depth_cfg.get("ncc_threshold", 0.35))
    margin_thresh = float(depth_cfg.get("confidence_margin", 0.08))

    cam_ref = cameras.get(ref_name)
    ref_gray = images_gray.get(ref_name)
    if cam_ref is None or not cam_ref.registered or ref_gray is None:
        return None

    if state is not None:
        d_min, d_max = _camera_depth_range_from_landmarks(
            ref_name, cam_ref, state, depth_cfg
        )
    else:
        d_min = float(depth_cfg.get("d_min", 0.5))
        d_max = float(depth_cfg.get("d_max", 80.0))

    depth_hyps = np.linspace(d_min, d_max, max(8, n_hyp), dtype=np.float32)
    H_img, W_img = ref_gray.shape

    best_score = np.full((H_img, W_img), -1.0, dtype=np.float32)
    second_score = np.full((H_img, W_img), -1.0, dtype=np.float32)
    best_depth = np.zeros((H_img, W_img), dtype=np.float32)

    nei_list = []
    for nm in neighbor_names:
        cam_n = cameras.get(nm)
        img_n = images_gray.get(nm)
        if cam_n is not None and cam_n.registered and img_n is not None:
            nei_list.append((nm, cam_n, img_n))

    if not nei_list:
        return None

    for d in depth_hyps:
        agg_score = np.zeros((H_img, W_img), dtype=np.float32)
        valid_count = 0

        for nm, cam_n, img_n in nei_list:
            if img_n.shape != (H_img, W_img):
                img_n_rs = cv2.resize(img_n, (W_img, H_img), interpolation=cv2.INTER_AREA)
            else:
                img_n_rs = img_n

            try:
                H_hom = compute_homography_plane(
                    cam_ref.K, cam_ref.R, cam_ref.t,
                    cam_n.K, cam_n.R, cam_n.t, float(d)
                )
                warped = warp_image(img_n_rs, H_hom)

                # Warp validity: border pixels are zero. Use grayscale > 0 rather
                # than treating black image content as valid stereo support.
                warped_valid = (warped > 0).astype(np.float32)
                zncc = compute_zncc_image(ref_gray, warped, win)
                agg_score += zncc * warped_valid
                valid_count += 1
            except Exception as e:
                logger.debug(
                    "Homography computation failed for %s depth %.3f: %s",
                    ref_name, float(d), e,
                )

        if valid_count == 0:
            continue

        agg_score /= float(valid_count)

        better = agg_score > best_score
        second_score[better] = best_score[better]
        best_score[better] = agg_score[better]
        best_depth[better] = float(d)

        between = (~better) & (agg_score > second_score)
        second_score[between] = agg_score[between]

    confidence = best_score - second_score
    valid_mask = (
        (best_score >= ncc_thresh)
        & (confidence >= margin_thresh)
        & (best_depth >= d_min)
        & (best_depth <= d_max)
    )

    depth_map = np.where(valid_mask, best_depth, 0.0).astype(np.float32)

    n_valid = int(valid_mask.sum())
    pct_valid = 100.0 * n_valid / valid_mask.size
    if n_valid:
        d_vals = depth_map[valid_mask]
        pct_at_min = 100.0 * (d_vals <= d_min + 0.05 * max(d_max - d_min, 1.0)).sum() / n_valid
        pct_at_max = 100.0 * (d_vals >= d_max - 0.05 * max(d_max - d_min, 1.0)).sum() / n_valid
        logger.info(
            "  [%s] Raw depth: valid=%.1f%% median=%.2f std=%.2f "
            "score_med=%.3f margin_med=%.3f edge[min,max]=%.1f%%/%.1f%%",
            ref_name, pct_valid, float(np.median(d_vals)), float(d_vals.std()),
            float(np.median(best_score[valid_mask])),
            float(np.median(confidence[valid_mask])),
            pct_at_min, pct_at_max,
        )
    else:
        logger.warning("  [%s] Raw depth has no confident pixels", ref_name)

    # Filling is deliberately conservative and disabled by default. It is
    # safer to fuse true stereo support than to turn invalid pixels into
    # synthetic geometry.
    if depth_cfg.get("depth_fill", False) and n_valid:
        depth_map = fill_depth(depth_map, valid_mask)

    return depth_map


def fill_depth(depth_map: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """Conservative one/two-pixel gap fill from measured depth only."""
    filled = depth_map.copy()
    if valid_mask is not None and not valid_mask.any():
        return filled

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(2):
        dilated = cv2.dilate(filled, kernel)
        candidate = (filled == 0) & (dilated > 0)
        filled[candidate] = dilated[candidate]

    return filled


def _shared_landmark_count(state, ref_name: str, candidate: str) -> int:
    count = 0
    for lm in state.landmarks.values():
        obs = lm.observations
        if ref_name in obs and candidate in obs:
            count += 1
    return count


def choose_reference_images(
    state,
    image_names: List[str],
    max_depth_images: int,
    n_neighbors: int,
) -> List[Tuple[str, List[str]]]:
    """Choose references by landmark support and neighbours by actual overlap."""
    registered_names = [
        nm for nm in image_names
        if state.cameras.get(nm) and state.cameras[nm].registered
    ]
    if not registered_names:
        return []

    lm_per_cam = {nm: 0 for nm in registered_names}
    for lm in state.landmarks.values():
        for nm in lm.observations:
            if nm in lm_per_cam:
                lm_per_cam[nm] += 1

    sorted_cams = sorted(
        registered_names,
        key=lambda x: lm_per_cam.get(x, 0),
        reverse=True,
    )

    if max_depth_images > 0:
        step = max(1, len(sorted_cams) // max_depth_images)
        refs = sorted_cams[::step][:max_depth_images]
    else:
        refs = sorted_cams

    results = []
    for ref in refs:
        candidates = []
        cam_ref = state.cameras[ref]
        C_ref = cam_ref.center

        for nm in registered_names:
            if nm == ref:
                continue
            shared = _shared_landmark_count(state, ref, nm)
            if shared < 8:
                continue
            cam_n = state.cameras[nm]
            baseline = float(np.linalg.norm(cam_n.center - C_ref))
            candidates.append((shared, baseline, nm))

        # Prefer real image overlap first, then a useful non-zero baseline.
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        neighbors = [nm for _, _, nm in candidates[:n_neighbors]]

        if neighbors:
            results.append((ref, neighbors))
        else:
            logger.warning("  [%s] no overlapping registered neighbours found", ref)

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
                img = cv2.resize(
                    img, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
        return img
    except Exception:
        return None


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

    depth_dir = ensure_dir(os.path.join(output_dir, "depth"))
    preview_dir = ensure_dir(os.path.join(output_dir, "depth_preview"))

    depth_max_dim = min(
        cfg.get("image", {}).get("resize_for_processing", 1600), 600
    )
    max_refs = int(depth_cfg.get("max_depth_images", 10))
    n_neighbors = int(depth_cfg.get("num_neighbors", 2))

    image_names = [p.name for p in images]
    logger.info(
        "[INFO] Loading images for depth estimation (max_dim=%d)...",
        depth_max_dim,
    )

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

    scaled_cameras: Dict[str, Camera] = {}
    for nm, cam in cameras.items():
        if nm not in images_gray:
            continue
        gray = images_gray[nm]
        dh, dw = gray.shape
        scale_x = dw / cam.width
        scale_y = dh / cam.height
        c = Camera(
            image_name=cam.image_name,
            width=dw, height=dh,
            fx=cam.fx * scale_x, fy=cam.fy * scale_y,
            cx=cam.cx * scale_x, cy=cam.cy * scale_y,
        )
        c.R = cam.R
        c.t = cam.t
        c.registered = cam.registered
        scaled_cameras[nm] = c

    refs_and_neighbors = choose_reference_images(
        state, image_names, max_refs, n_neighbors
    )
    logger.info(
        "[INFO] Estimating depth for %d images at %dpx "
        "(sparse-guided plane-sweep ZNCC)...",
        len(refs_and_neighbors), depth_max_dim,
    )

    depth_maps: Dict[str, np.ndarray] = {}
    t0 = time.time()

    for i, (ref_name, nei_names) in enumerate(refs_and_neighbors):
        logger.info(
            "  [%d/%d] Depth: %s (neighbors: %s)",
            i + 1, len(refs_and_neighbors), ref_name, nei_names,
        )
        t_ref = time.time()

        depth_map = estimate_depth_for_image(
            ref_name, nei_names, scaled_cameras, images_gray, cfg, state=state
        )
        if depth_map is None or depth_map.max() == 0:
            logger.warning("  Depth estimation failed/empty for %s", ref_name)
            continue

        valid = depth_map > 0
        valid_pct = 100.0 * valid.sum() / valid.size
        logger.info(
            "  -> depth range [%.2f, %.2f], valid=%.1f%% (%.1fs)",
            float(depth_map[valid].min()),
            float(depth_map[valid].max()),
            valid_pct,
            time.time() - t_ref,
        )

        depth_maps[ref_name] = depth_map
        stem = Path(ref_name).stem
        npy_path = str(depth_dir / f"depth_map_{i+1:02d}_{stem}.npy")
        np.save(npy_path, depth_map)
        png_path = str(depth_dir / f"depth_map_{i+1:02d}_{stem}.png")
        _save_depth_png(depth_map, png_path)
        prev_path = str(preview_dir / f"depth_preview_{i+1:02d}_{stem}.png")
        _save_depth_preview(depth_map, ref_name, prev_path)

    logger.info(
        "[INFO] Depth estimation done: %d maps in %.1fs",
        len(depth_maps), time.time() - t0,
    )
    return depth_maps


def _save_depth_png(depth: np.ndarray, path: str):
    valid = depth > 0
    if not valid.any():
        return
    d_min = depth[valid].min()
    d_max = depth[valid].max()
    if d_max == d_min:
        return
    norm = np.zeros_like(depth, dtype=np.uint16)
    norm[valid] = (
        (depth[valid] - d_min) / (d_max - d_min) * 65535
    ).astype(np.uint16)
    cv2.imwrite(path, norm)


def _save_depth_preview(depth: np.ndarray, ref_name: str, path: str):
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
        ax.set_title(
            f"Depth map: {ref_name}\nValid: {100*valid.mean():.1f}%"
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.debug("Depth preview error: %s", e)
