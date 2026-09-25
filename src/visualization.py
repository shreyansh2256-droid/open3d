"""
visualization.py — Point cloud and camera trajectory visualization.
Uses Open3D offscreen rendering for automated screenshot generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .io_utils import ensure_dir

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Camera frustum helper
# ─────────────────────────────────────────────────────────────────────────────

def create_camera_frustum(cam, scale: float = 0.5):
    """Create an Open3D LineSet frustum for a camera."""
    try:
        import open3d as o3d
    except ImportError:
        return None

    if cam.R is None or not cam.registered:
        return None

    C = cam.camera_center()
    K = cam.K
    W, H = cam.width, cam.height
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Image corners in camera space at z = scale
    z = scale
    corners_cam = np.array([
        [(-cx)      / fx * z, (-cy)      / fy * z, z],
        [(W - cx)   / fx * z, (-cy)      / fy * z, z],
        [(W - cx)   / fx * z, (H - cy)   / fy * z, z],
        [(-cx)      / fx * z, (H - cy)   / fy * z, z],
    ])

    # Transform to world
    corners_world = (cam.R.T @ (corners_cam.T - cam.t[:, None])).T

    points = np.vstack([C, corners_world])  # 5 points
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [[1, 0.5, 0]] * len(lines)  # Orange

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


# ─────────────────────────────────────────────────────────────────────────────
# Save screenshot / render
# ─────────────────────────────────────────────────────────────────────────────

def save_pointcloud_screenshot(
    ply_path: str,
    output_path: str,
    state=None,
    cfg: Dict = None,
) -> bool:
    """
    Load a PLY file and save a screenshot using Open3D offscreen rendering.
    Falls back to matplotlib scatter plot if Open3D GUI is unavailable.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("Open3D not available for visualization")
        return _matplotlib_fallback(ply_path, output_path)

    if not Path(ply_path).exists():
        logger.error(f"PLY file not found: {ply_path}")
        return False

    vis_cfg = cfg.get("visualization", {}) if cfg else {}
    bg_color   = vis_cfg.get("background_color", [0.1, 0.1, 0.1])
    pt_size    = vis_cfg.get("point_size", 2.0)
    show_cams  = vis_cfg.get("show_cameras", True)
    cam_scale  = vis_cfg.get("camera_scale", 0.5)

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        logger.info(f"[VIS] Loaded {len(pcd.points):,} points for visualization")

        # Build geometry list
        geometries = [pcd]

        if show_cams and state is not None:
            for cam in state.cameras.values():
                if cam.registered:
                    frustum = create_camera_frustum(cam, scale=cam_scale)
                    if frustum:
                        geometries.append(frustum)

        # Try headless rendering
        ensure_dir(str(Path(output_path).parent))
        success = _offscreen_render(geometries, output_path, bg_color, pt_size)
        if success:
            logger.info(f"[VIS] Screenshot saved: {output_path}")
            return True
    except Exception as e:
        logger.warning(f"Open3D rendering error: {e}")

    # Fallback
    return _matplotlib_fallback(ply_path, output_path)


def _offscreen_render(geometries, output_path: str, bg_color, pt_size: float) -> bool:
    """Try Open3D offscreen rendering."""
    try:
        import open3d as o3d

        # Try the visualizer approach (works headless with Mesa/offscreen)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=720)
        vis.get_render_option().background_color = np.array(bg_color)
        vis.get_render_option().point_size = pt_size

        for geom in geometries:
            vis.add_geometry(geom)

        vis.poll_events()
        vis.update_renderer()

        # Auto-orient
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)
        ctr.rotate(0, -200)  # Look slightly down

        vis.capture_screen_image(output_path, do_render=True)
        vis.destroy_window()
        return True
    except Exception as e:
        logger.debug(f"Offscreen render failed: {e}")
        return False


def _matplotlib_fallback(ply_path: str, output_path: str) -> bool:
    """Fallback: 3D scatter plot using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # Read PLY manually
        pts, cols = _read_ply_simple(ply_path)
        if pts is None or len(pts) == 0:
            return False

        # Subsample for display
        N = min(50000, len(pts))
        idx = np.random.choice(len(pts), N, replace=False)
        pts_s = pts[idx]
        cols_s = cols[idx] / 255.0 if cols is not None else np.ones((N, 3)) * 0.5

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                   c=cols_s, s=0.5, alpha=0.6)
        ax.set_title(f"PIXEL-OPS Point Cloud ({len(pts):,} points)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_facecolor("0.1")
        fig.patch.set_facecolor("0.1")
        ax.tick_params(colors="white")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="0.1")
        plt.close()
        logger.info(f"[VIS] Matplotlib screenshot saved: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Matplotlib fallback failed: {e}")
        return False


def _read_ply_simple(ply_path: str):
    """Read XYZ and RGB from a PLY file using Open3D."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        return pts, cols
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Camera trajectory plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_camera_trajectory(state, output_dir: str):
    """Save a 2D camera trajectory overview plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = []
        for cam in state.cameras.values():
            if cam.registered:
                centers.append(cam.camera_center())

        if not centers:
            return

        centers = np.array(centers)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Top view: X-Z
        axes[0].scatter(centers[:, 0], centers[:, 2], c="cyan", s=30)
        axes[0].plot(centers[:, 0], centers[:, 2], "c-", alpha=0.5)
        axes[0].set_title("Camera trajectory (top view: X-Z)")
        axes[0].set_xlabel("X"); axes[0].set_ylabel("Z")
        # Side view: X-Y
        axes[1].scatter(centers[:, 0], centers[:, 1], c="lime", s=30)
        axes[1].plot(centers[:, 0], centers[:, 1], "g-", alpha=0.5)
        axes[1].set_title("Camera trajectory (side view: X-Y)")
        axes[1].set_xlabel("X"); axes[1].set_ylabel("Y")
        for ax in axes:
            ax.set_facecolor("0.15")
        fig.patch.set_facecolor("0.15")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "debug", "camera_trajectory.png")
        ensure_dir(os.path.join(output_dir, "debug"))
        plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="0.15")
        plt.close()
        logger.info(f"[VIS] Camera trajectory plot: {out_path}")
    except Exception as e:
        logger.debug(f"Camera trajectory plot error: {e}")
