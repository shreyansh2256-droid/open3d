"""
fusion.py — Back-project depth maps into 3D, fuse, filter, and export PLY.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .camera import Camera
from .io_utils import load_image_rgb, ensure_dir

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Back-project one depth map to 3D
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth_map: np.ndarray,
    rgb_img: np.ndarray,
    cam: Camera,
    max_points: int = 50000,
) -> Optional[np.ndarray]:
    """
    Convert a depth map + RGB image into an (N, 6) array of (X, Y, Z, R, G, B).
    Uses pinhole back-projection and transforms to world coordinates.
    """
    if not cam.registered:
        return None

    H, W = depth_map.shape
    K_inv = cam.Kinv

    # Pixel grid
    us = np.arange(W)
    vs = np.arange(H)
    UU, VV = np.meshgrid(us, vs)

    valid = depth_map > 0
    u_v = UU[valid].ravel()
    v_v = VV[valid].ravel()
    d_v = depth_map[valid].ravel().astype(np.float64)

    if len(u_v) == 0:
        return None

    # Back-project: X_cam = depth * K_inv * [u, v, 1]^T
    uv1 = np.stack([u_v, v_v, np.ones_like(u_v)], axis=0).astype(np.float64)  # (3, N)
    X_cam = d_v * (K_inv @ uv1)               # (3, N)

    # Camera → World
    X_world = (cam.R.T @ (X_cam - cam.t[:, None])).T  # (N, 3)

    # Filter obviously bad points
    finite_mask = np.all(np.isfinite(X_world), axis=1)
    X_world = X_world[finite_mask]
    u_v = u_v[finite_mask]
    v_v = v_v[finite_mask]

    if len(X_world) == 0:
        return None

    # Subsample if too many
    if max_points > 0 and len(X_world) > max_points:
        idx = np.random.choice(len(X_world), max_points, replace=False)
        X_world = X_world[idx]
        u_v = u_v[idx]
        v_v = v_v[idx]

    # Get RGB colors
    rgb_h, rgb_w = rgb_img.shape[:2]
    u_clip = np.clip(u_v, 0, rgb_w - 1).astype(int)
    v_clip = np.clip(v_v, 0, rgb_h - 1).astype(int)
    colors = rgb_img[v_clip, u_clip].astype(np.float32)  # (N, 3)

    # Stack XYZ + RGB
    xyzrgb = np.concatenate([X_world.astype(np.float32), colors], axis=1)  # (N, 6)
    return xyzrgb


# ─────────────────────────────────────────────────────────────────────────────
# Fuse multiple depth maps
# ─────────────────────────────────────────────────────────────────────────────

def fuse_depth_maps(
    depth_maps: Dict[str, np.ndarray],
    cameras: Dict[str, Camera],
    images: List[Path],
    cfg: Dict,
    output_dir: str,
) -> Optional[object]:
    """
    Back-project all depth maps, fuse into a single point cloud,
    clean it, and return an Open3D point cloud.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("Open3D not installed; cannot fuse depth maps")
        return None

    fusion_cfg = cfg.get("fusion", {})
    voxel_size      = fusion_cfg.get("voxel_size", 0.05)
    stat_nb         = fusion_cfg.get("statistical_outlier_nb", 20)
    stat_std        = fusion_cfg.get("statistical_outlier_std", 2.0)
    rad_radius      = fusion_cfg.get("radius_outlier_radius", 0.1)
    rad_min_pts     = fusion_cfg.get("radius_outlier_min_points", 5)
    max_pts_per_img = fusion_cfg.get("max_points_per_depth_image", 50000)
    max_dim         = cfg.get("image", {}).get("resize_for_processing", 1600)

    # Build image path dict
    img_path_dict = {p.name: p for p in images}

    all_xyzrgb = []
    for ref_name, depth_map in depth_maps.items():
        cam = cameras.get(ref_name)
        if cam is None or not cam.registered:
            continue
        img_path = img_path_dict.get(ref_name)
        if img_path is None:
            continue
        rgb = load_image_rgb(img_path, max_dim)
        if rgb is None:
            continue

        rgb_h, rgb_w = rgb.shape[:2]
        dm_h, dm_w = depth_map.shape

        # Build a camera scaled to match the depth map dimensions
        scale_x = dm_w / cam.width
        scale_y = dm_h / cam.height
        from .camera import Camera as _Cam
        dep_cam = _Cam(
            image_name=cam.image_name,
            width=dm_w, height=dm_h,
            fx=cam.fx * scale_x, fy=cam.fy * scale_y,
            cx=cam.cx * scale_x, cy=cam.cy * scale_y,
        )
        dep_cam.R = cam.R
        dep_cam.t = cam.t
        dep_cam.registered = cam.registered

        xyzrgb = depth_to_pointcloud(depth_map, rgb, dep_cam, max_pts_per_img)
        if xyzrgb is not None:
            all_xyzrgb.append(xyzrgb)
            logger.info(f"  Fused {ref_name}: {len(xyzrgb):,} points")

    if not all_xyzrgb:
        logger.warning("[FUSION] No valid points from depth back-projection")
        return None

    combined = np.concatenate(all_xyzrgb, axis=0)
    logger.info(f"[FUSION] Raw combined points: {len(combined):,}")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(combined[:, 3:6].astype(np.float64) / 255.0)

    # ── Auto-scale parameters based on actual scene extent ───────────────────
    pts_arr = np.asarray(pcd.points)
    # Remove NaN/Inf before computing stats
    finite_mask = np.isfinite(pts_arr).all(axis=1)
    if not finite_mask.all():
        logger.warning("[FUSION] Removing %d NaN/Inf points", (~finite_mask).sum())
        pcd = pcd.select_by_index(np.where(finite_mask)[0])
        pts_arr = np.asarray(pcd.points)

    extents = pts_arr.max(axis=0) - pts_arr.min(axis=0)
    scene_scale = float(np.median(extents[extents > 0])) if extents.any() else 10.0

    # Measure actual nearest-neighbour distances on a sample
    sample_n = min(2000, len(pts_arr))
    sample_idx = np.random.choice(len(pts_arr), sample_n, replace=False)
    sample_pts = pts_arr[sample_idx]
    # Brute-force NN on sample (fast enough for 2000 pts)
    from scipy.spatial import cKDTree
    try:
        tree = cKDTree(pts_arr)
        dists, _ = tree.query(sample_pts, k=2)   # k=2: skip self
        nn_dists = dists[:, 1]
        nn_median = float(np.median(nn_dists))
        nn_p95 = float(np.percentile(nn_dists, 95))
        logger.info("[FUSION] NN-dist: median=%.4f  p95=%.4f  scene_scale=%.2f",
                    nn_median, nn_p95, scene_scale)
    except Exception:
        nn_median = scene_scale * 0.01
        nn_p95 = scene_scale * 0.05
        logger.warning("[FUSION] NN-dist computation failed; using estimates")

    # Adaptive parameters
    auto_voxel  = max(voxel_size,  nn_median * 0.5)     # half median-NN
    auto_radius = max(rad_radius,  nn_median * 8.0)     # 8× median-NN spacing
    # Safety cap: never larger than 10% of scene
    auto_radius = min(auto_radius, scene_scale * 0.10)

    logger.info("[FUSION] scene_scale=%.3f  auto_voxel=%.4f  auto_radius=%.4f",
                scene_scale, auto_voxel, auto_radius)

    # ── Voxel downsampling ────────────────────────────────────────────────────
    pcd = pcd.voxel_down_sample(auto_voxel)
    logger.info("[FUSION] After voxel downsampling: %d", len(pcd.points))

    # ── Statistical outlier removal ───────────────────────────────────────────
    if len(pcd.points) > stat_nb:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=stat_nb,
                                                  std_ratio=stat_std)
        logger.info("[FUSION] After statistical outlier removal: %d", len(pcd.points))

    # ── Radius outlier removal ─────────────────────────────────────────────────
    # Only apply if auto_radius is large enough relative to spacing
    if len(pcd.points) > rad_min_pts and auto_radius > auto_voxel * 2:
        pcd, _ = pcd.remove_radius_outlier(nb_points=rad_min_pts,
                                            radius=auto_radius)
        logger.info("[FUSION] After radius outlier removal: %d", len(pcd.points))
    else:
        logger.info("[FUSION] Radius outlier removal skipped (radius too small vs spacing)")

    return pcd


# ─────────────────────────────────────────────────────────────────────────────
# Export sparse SfM cloud
# ─────────────────────────────────────────────────────────────────────────────

def export_sparse_cloud(state, output_path: str) -> bool:
    """Export the SfM sparse point cloud as PLY."""
    try:
        import open3d as o3d
    except ImportError:
        logger.error("Open3D not available for PLY export")
        return _export_sparse_ply_manual(state, output_path)

    pts = state.get_points_array()
    cols = state.get_colors_array()

    if len(pts) == 0:
        logger.warning("No sparse points to export")
        return False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if len(cols) == len(pts):
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    ensure_dir(str(Path(output_path).parent))
    o3d.io.write_point_cloud(output_path, pcd)
    logger.info(f"[INFO] Sparse cloud saved: {output_path} ({len(pts):,} points)")
    return True


def _export_sparse_ply_manual(state, output_path: str) -> bool:
    """Fallback: write PLY manually without Open3D."""
    pts = state.get_points_array()
    cols = state.get_colors_array()
    if len(pts) == 0:
        return False
    try:
        ensure_dir(str(Path(output_path).parent))
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with open(output_path, "w") as f:
            f.write(header)
            for i, (p, c) in enumerate(zip(pts, cols)):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
        return True
    except Exception as e:
        logger.error(f"Manual PLY export error: {e}")
        return False


def export_dense_cloud(pcd, output_path: str) -> bool:
    """Export the dense/fused point cloud as PLY."""
    try:
        import open3d as o3d
        ensure_dir(str(Path(output_path).parent))
        o3d.io.write_point_cloud(output_path, pcd)
        logger.info(f"[INFO] Dense cloud saved: {output_path} ({len(pcd.points):,} points)")
        return True
    except Exception as e:
        logger.error(f"Dense PLY export error: {e}")
        return False
