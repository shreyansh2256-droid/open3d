"""Semi-dense depth estimation for the PIXEL-OPS photogrammetry pipeline.

Uses a vectorised plane-sweep ZNCC stereo method.  Depth hypotheses are
constrained by the sparse SfM landmarks visible in each reference camera.
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
    K1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    K2: np.ndarray, R2: np.ndarray, t2: np.ndarray,
    depth: float, normal: np.ndarray = None,
) -> np.ndarray:
    """Homography mapping reference pixels on a fronto-parallel plane to neighbour."""
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if not np.isfinite(depth) or depth <= 0.0:
        raise ValueError(f"Depth hypothesis must be positive and finite; got {depth!r}")

    K1 = np.asarray(K1, dtype=np.float64)
    K2 = np.asarray(K2, dtype=np.float64)
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64).ravel()
    t2 = np.asarray(t2, dtype=np.float64).ravel()
    normal = np.asarray(normal, dtype=np.float64).ravel()

    if not (np.all(np.isfinite(K1)) and np.all(np.isfinite(K2)) and np.all(np.isfinite(R1))
            and np.all(np.isfinite(R2)) and np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))
            and np.all(np.isfinite(normal))):
        raise ValueError("Non-finite camera or plane parameters for homography")

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    H = K2 @ (R_rel + np.outer(t_rel, normal) / depth) @ np.linalg.inv(K1)
    if not np.all(np.isfinite(H)):
        raise ValueError("Non-finite homography produced for depth hypothesis")
    return H


def warp_image(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp neighbour image into reference-image coordinates."""
    h, w = img.shape[:2]
    return cv2.warpPerspective(
        img, H, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def compute_zncc_image(ref: np.ndarray, warped: np.ndarray, win: int = 9) -> np.ndarray:
    """Compute a per-pixel local zero-mean normalized cross correlation map."""
    ref_f = np.asarray(ref, dtype=np.float32)
    warp_f = np.asarray(warped, dtype=np.float32)

    if ref_f.shape != warp_f.shape:
        raise ValueError(f"ZNCC inputs must have the same shape; got {ref_f.shape} and {warp_f.shape}")

    mu1 = cv2.boxFilter(ref_f, -1, (win, win))
    mu2 = cv2.boxFilter(warp_f, -1, (win, win))
    ref_c = ref_f - mu1
    warp_c = warp_f - mu2

    sigma1_sq = cv2.boxFilter(ref_c * ref_c, -1, (win, win)).astype(np.float32)
    sigma2_sq = cv2.boxFilter(warp_c * warp_c, -1, (win, win)).astype(np.float32)
    sigma12 = cv2.boxFilter(ref_c * warp_c, -1, (win, win)).astype(np.float32)

    # Local variances are non-negative in exact arithmetic. Small negative values
    # here are numerical noise from floating point and must not be fed to sqrt().
    sigma1_sq = np.nan_to_num(sigma1_sq, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2_sq = np.nan_to_num(sigma2_sq, nan=0.0, posinf=0.0, neginf=0.0)
    sigma12 = np.nan_to_num(sigma12, nan=0.0, posinf=0.0, neginf=0.0)
    sigma1_sq = np.clip(sigma1_sq, 0.0, None)
    sigma2_sq = np.clip(sigma2_sq, 0.0, None)

    denom = np.sqrt(np.maximum(np.asarray(sigma1_sq * sigma2_sq, dtype=np.float32), 0.0)) + 1e-8
    denom = np.where(denom > 1e-8, denom, 1e-8)
    zncc = np.divide(
        sigma12,
        denom,
        out=np.zeros_like(sigma12, dtype=np.float32),
        where=np.isfinite(denom),
    )
    zncc = np.nan_to_num(zncc, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(zncc, -1.0, 1.0)


def _adaptive_depth_range(
    ref_name: str,
    cam: Camera,
    state,
    cfg: Dict,
) -> Tuple[float, float, int]:
    """Estimate a robust per-camera Z range from sparse SfM landmarks."""
    dc = cfg.get("depth", {})
    fallback_lo = float(dc.get("d_min", 0.5))
    fallback_hi = float(dc.get("d_max", 50.0))
    if not dc.get("adaptive_depth_range", True) or state is None:
        return fallback_lo, fallback_hi, 0

    zs = []
    for lm in state.landmarks.values():
        if ref_name not in lm.observations:
            continue
        try:
            X_cam = cam.R @ lm.xyz + cam.t
            z = float(X_cam[2])
            if z <= 0 or not np.isfinite(z):
                continue
            uv = cam.project(lm.xyz.reshape(1, 3))[0]
            if not np.isfinite(uv).all():
                continue
            if 0 <= uv[0] < cam.width and 0 <= uv[1] < cam.height:
                zs.append(z)
        except Exception:
            continue

    min_landmarks = int(dc.get("adaptive_min_landmarks", 20))
    if len(zs) < min_landmarks:
        return fallback_lo, fallback_hi, len(zs)

    zs = np.asarray(zs, dtype=np.float64)
    p_lo, p_hi = dc.get("depth_range_percentiles", [5.0, 95.0])
    lo = float(np.percentile(zs, p_lo))
    hi = float(np.percentile(zs, p_hi))
    span = max(hi - lo, 1e-6)
    pad_fraction = float(dc.get("depth_range_padding", 0.20))
    pad = max(span * pad_fraction, float(np.median(zs)) * 0.03)
    lo = max(0.05, lo - pad)
    hi = hi + pad

    min_span = float(dc.get("min_depth_range", 2.0))
    if hi - lo < min_span:
        mid = 0.5 * (hi + lo)
        lo = max(0.05, mid - min_span / 2)
        hi = mid + min_span / 2

    max_hi = float(dc.get("max_adaptive_depth", 2000.0))
    hi = min(hi, max_hi)
    return lo, hi, len(zs)


def estimate_depth_for_image(
    ref_name: str,
    neighbor_names: List[str],
    cameras: Dict[str, Camera],
    images_gray: Dict[str, np.ndarray],
    cfg: Dict,
    state=None,
) -> Optional[np.ndarray]:
    """Estimate a depth map with an adaptive plane-sweep and confidence test."""
    depth_cfg = cfg.get("depth", {})
    n_hyp = int(depth_cfg.get("num_depth_hypotheses", 96))
    win = int(depth_cfg.get("patch_size", 9))
    ncc_thresh = float(depth_cfg.get("ncc_threshold", 0.35))
    confidence_margin = float(depth_cfg.get("confidence_margin", 0.04))

    cam_ref = cameras.get(ref_name)
    ref_gray = images_gray.get(ref_name)
    if cam_ref is None or not cam_ref.registered or ref_gray is None:
        return None

    d_min, d_max, n_landmarks = _adaptive_depth_range(ref_name, cam_ref, state, cfg)
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
            if img_n.shape != (H_img, W_img):
                img_n = cv2.resize(img_n, (W_img, H_img), interpolation=cv2.INTER_AREA)
            nei_list.append((nm, cam_n, img_n))
    if not nei_list:
        return None

    for d in depth_hyps:
        agg_score = np.zeros((H_img, W_img), dtype=np.float32)
        valid_count = 0
        for nm, cam_n, img_n in nei_list:
            try:
                H_hom = compute_homography_plane(
                    cam_ref.K, cam_ref.R, cam_ref.t,
                    cam_n.K, cam_n.R, cam_n.t, float(d)
                )
                warped = warp_image(img_n, H_hom)
                valid_warp = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                              if warped.ndim == 3 else warped) > 0
                zncc = compute_zncc_image(ref_gray, warped, win)
                zncc = np.nan_to_num(zncc, nan=0.0, posinf=0.0, neginf=0.0)
                agg_score += np.where(valid_warp & np.isfinite(zncc), zncc, 0.0)
                valid_count += 1
            except Exception as exc:
                logger.debug("Depth homography failed at %.3f: %s", d, exc)

        if valid_count == 0:
            continue
        agg_score = np.nan_to_num(agg_score, nan=0.0, posinf=0.0, neginf=0.0)
        agg_score /= valid_count
        better = agg_score > best_score
        second_score = np.where(better, best_score, np.maximum(second_score, agg_score))
        best_score = np.maximum(best_score, agg_score)
        best_depth[better] = d

    confidence = best_score - np.maximum(second_score, -1.0)
    valid_mask = (
        (best_score >= ncc_thresh)
        & (confidence >= confidence_margin)
        & (best_depth > 0)
    )
    depth_map = np.where(valid_mask, best_depth, 0.0).astype(np.float32)

    n_valid = int(valid_mask.sum())
    pct = 100.0 * n_valid / valid_mask.size
    if n_valid:
        vals = depth_map[valid_mask]
        logger.info(
            "  [%s] Raw depth: valid=%.1f%% median=%.2f std=%.2f "
            "range=[%.2f, %.2f] NCC=%.3f margin=%.3f landmarks=%d",
            ref_name, pct, float(np.median(vals)), float(vals.std()),
            float(vals.min()), float(vals.max()),
            float(np.median(best_score[valid_mask])),
            float(np.median(confidence[valid_mask])), n_landmarks,
        )
        at_min = 100.0 * np.mean(vals <= d_min + 0.02 * max(d_max - d_min, 1e-6))
        at_max = 100.0 * np.mean(vals >= d_max - 0.02 * max(d_max - d_min, 1e-6))
        if at_min > 25 or at_max > 25:
            logger.warning(
                "  [%s] %.1f%% near d_min and %.1f%% near d_max; "
                "depth range may still be too narrow",
                ref_name, at_min, at_max,
            )
    else:
        logger.warning("  [%s] No confident raw depth pixels (NCC/margin thresholds)", ref_name)

    # Do not synthesize large regions of geometry.  Optional fill is deliberately
    # limited to one small pass and only used when the raw map has enough support.
    if depth_cfg.get("depth_fill", False) and pct >= float(depth_cfg.get("min_valid_before_fill_pct", 15.0)):
        depth_map = fill_depth(depth_map, valid_mask, iterations=int(depth_cfg.get("fill_iterations", 1)))

    return depth_map


def fill_depth(depth_map: np.ndarray, valid_mask: np.ndarray = None, iterations: int = 1) -> np.ndarray:
    """Conservatively fill tiny holes from nearby measured depth values."""
    filled = depth_map.copy()
    if valid_mask is not None and not valid_mask.any():
        return filled
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(0, iterations)):
        dilated = cv2.dilate(filled, kernel)
        filled = np.where(filled == 0, dilated, filled)
    if filled.max() > 0:
        # Very mild edge-preserving smoothing; never rescale the depth range.
        filled = cv2.bilateralFilter(filled.astype(np.float32), 5, 1.5, 1.5)
        if valid_mask is not None:
            filled = np.where(
                (valid_mask | (depth_map > 0)), filled, 0.0
            ).astype(np.float32)
    return filled


def _camera_center(cam: Camera) -> np.ndarray:
    return -cam.R.T @ cam.t


def choose_reference_images(
    state,
    image_names: List[str],
    max_depth_images: int,
    n_neighbors: int,
    cfg: Dict = None,
) -> List[Tuple[str, List[str]]]:
    """Choose references with good landmark support and neighbours with useful baseline.

    Neighbour selection strategy:
    - Primary criterion: shared landmarks (overlap) — configurable minimum threshold.
    - Baseline: spread neighbours across the available baseline range rather than
      always maximising it.  This gives the plane-sweep both short-baseline precision
      and long-baseline disambiguation.
    - Fallback: when no candidate has enough shared landmarks, sort ALL registered
      cameras by spatial distance (camera-centre norm) and take the closest ones.
      Previously the fallback used index arithmetic on a landmark-count-sorted list,
      which selected cameras with similar landmark counts, not spatially nearby ones.
    """
    registered_names = [
        nm for nm in image_names
        if state.cameras.get(nm) is not None and state.cameras[nm].registered
    ]
    if not registered_names:
        return []

    depth_cfg = (cfg or {}).get("depth", {})
    # Configurable minimum shared landmarks; was hardcoded to 10 via walrus operator.
    min_shared = int(depth_cfg.get("min_shared_landmarks", 10))

    lm_per_cam = {nm: 0 for nm in registered_names}
    observations = {nm: set() for nm in registered_names}
    for lm_id, lm in state.landmarks.items():
        for nm in lm.observations:
            if nm in observations:
                lm_per_cam[nm] += 1
                observations[nm].add(lm_id)

    sorted_cams = sorted(registered_names, key=lambda x: lm_per_cam[x], reverse=True)
    if max_depth_images > 0:
        if len(sorted_cams) <= max_depth_images:
            refs = sorted_cams
        else:
            idx = np.linspace(0, len(sorted_cams) - 1, max_depth_images).astype(int)
            refs = [sorted_cams[i] for i in idx]
    else:
        refs = sorted_cams

    centers = {nm: _camera_center(state.cameras[nm]) for nm in registered_names}
    results = []

    for ref in refs:
        ref_center = centers[ref]
        candidates = []
        for nm in registered_names:
            if nm == ref:
                continue
            shared = len(observations[ref].intersection(observations[nm]))
            if shared < min_shared:
                continue
            baseline = float(np.linalg.norm(ref_center - centers[nm]))
            candidates.append((shared, baseline, nm))

        if not candidates:
            # Fallback: sort by SPATIAL distance (camera-centre norm), not by index in
            # a landmark-count-sorted list.  The old fallback was wrong because
            # registered_names is ordered by landmark count, not spatial proximity.
            spatial_sorted = sorted(
                [nm for nm in registered_names if nm != ref],
                key=lambda nm: float(np.linalg.norm(ref_center - centers[nm]))
            )
            fallback = spatial_sorted[:n_neighbors]
            if fallback:
                results.append((ref, fallback))
            continue

        # Select neighbours that span a useful range of baselines rather than
        # always maximising the baseline.  Strategy:
        #   1. Keep the top-overlap candidates (most shared landmarks).
        #   2. From those, sample at evenly-spaced baseline quantiles so we get
        #      at least one short-baseline and one long-baseline neighbour.
        # This improves plane-sweep stereo: short-baseline gives smooth localisation,
        # long-baseline resolves depth ambiguity.
        candidates.sort(key=lambda x: x[0], reverse=True)     # sort by shared DESC
        top_pool = candidates[: max(n_neighbors * 3, len(candidates))]
        top_pool.sort(key=lambda x: x[1])                      # sort by baseline ASC
        if len(top_pool) <= n_neighbors:
            chosen = [x[2] for x in top_pool]
        else:
            idx_spread = np.linspace(0, len(top_pool) - 1, n_neighbors).astype(int)
            chosen = [top_pool[i][2] for i in idx_spread]

        results.append((ref, chosen))
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
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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
    depth_max_dim = min(cfg.get("image", {}).get("resize_for_processing", 1600), 600)
    max_refs = int(depth_cfg.get("max_depth_images", 10))
    n_neighbors = int(depth_cfg.get("num_neighbors", 2))

    image_names = [p.name for p in images]
    logger.info("[INFO] Loading images for depth estimation (max_dim=%d)...", depth_max_dim)
    images_gray: Dict[str, np.ndarray] = {}
    img_path_dict = {p.name: p for p in images}

    for nm in image_names:
        if state.cameras.get(nm) and state.cameras[nm].registered:
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
            image_name=cam.image_name, width=dw, height=dh,
            fx=cam.fx * scale_x, fy=cam.fy * scale_y,
            cx=cam.cx * scale_x, cy=cam.cy * scale_y,
        )
        c.R, c.t, c.registered = cam.R.copy(), cam.t.copy(), cam.registered
        scaled_cameras[nm] = c

    refs_and_neighbors = choose_reference_images(state, image_names, max_refs, n_neighbors, cfg=cfg)
    logger.info(
        "[INFO] Estimating depth for %d images at %dpx "
        "(adaptive plane-sweep ZNCC)...",
        len(refs_and_neighbors), depth_max_dim,
    )

    depth_maps: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, (ref_name, nei_names) in enumerate(refs_and_neighbors):
        logger.info("  [%d/%d] Depth: %s (neighbors: %s)", i + 1, len(refs_and_neighbors), ref_name, nei_names)
        t_ref = time.time()
        depth_map = estimate_depth_for_image(
            ref_name, nei_names, scaled_cameras, images_gray, cfg, state=state
        )
        if depth_map is None or depth_map.max() <= 0:
            logger.warning("  Depth estimation failed/empty for %s", ref_name)
            continue
        valid_pct = 100.0 * np.mean(depth_map > 0)
        logger.info(
            "  -> depth range [%.2f, %.2f], valid=%.1f%% (%.1fs)",
            float(depth_map[depth_map > 0].min()),
            float(depth_map[depth_map > 0].max()),
            valid_pct, time.time() - t_ref,
        )
        depth_maps[ref_name] = depth_map
        stem = Path(ref_name).stem
        np.save(str(depth_dir / f"depth_map_{i+1:02d}_{stem}.npy"), depth_map)
        _save_depth_png(depth_map, str(depth_dir / f"depth_map_{i+1:02d}_{stem}.png"))
        _save_depth_preview(depth_map, ref_name, str(preview_dir / f"depth_preview_{i+1:02d}_{stem}.png"))

    logger.info("[INFO] Depth estimation done: %d maps in %.1fs", len(depth_maps), time.time() - t0)
    return depth_maps


def _save_depth_png(depth: np.ndarray, path: str):
    valid = depth > 0
    if not valid.any():
        return
    d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
    if d_max <= d_min:
        return
    norm = np.zeros_like(depth, dtype=np.uint16)
    norm[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    cv2.imwrite(path, norm)


def _save_depth_preview(depth: np.ndarray, ref_name: str, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        valid = depth > 0
        if not valid.any():
            return
        d_vis = depth.astype(np.float32).copy()
        d_vis[~valid] = np.nan
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = ax.imshow(d_vis, cmap="plasma", aspect="auto")
        plt.colorbar(im, ax=ax, label="Relative depth (scene units)")
        ax.set_title(f"Depth map: {ref_name}\nValid: {100*valid.mean():.1f}%")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.debug("Depth preview error: %s", exc)
