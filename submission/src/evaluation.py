"""
evaluation.py — Reconstruction quality report generation.
"""

import os
import json
import platform
import subprocess
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .io_utils import safe_json_dump, ensure_dir

logger = logging.getLogger(__name__)


def detect_gpu() -> Dict:
    """Detect GPU/CUDA availability."""
    info = {"has_cuda": False, "gpu_name": "N/A", "vram_mb": 0}
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            info["has_cuda"] = True
            info["gpu_name"] = parts[0].strip()
            if len(parts) > 1:
                try:
                    info["vram_mb"] = int(parts[1].replace("MiB", "").strip())
                except Exception:
                    pass
    except Exception:
        pass
    return info


def get_library_versions() -> Dict:
    """Collect installed library versions."""
    versions = {}
    for lib in ["cv2", "numpy", "scipy", "open3d", "PIL", "matplotlib"]:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    return versions


def compute_cloud_stats(ply_path: str) -> Dict:
    """Compute bounding box, centroid, point count from PLY."""
    stats = {"point_count": 0, "bbox": None, "centroid": None}
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return stats
        stats["point_count"] = len(pts)
        stats["bbox"] = {
            "min": pts.min(axis=0).tolist(),
            "max": pts.max(axis=0).tolist(),
            "extents": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
        }
        stats["centroid"] = pts.mean(axis=0).tolist()
        # Estimate mean density (points per unit^3)
        extents = np.array(stats["bbox"]["extents"])
        vol = float(extents.prod()) if extents.prod() > 0 else 1e-6
        stats["mean_density_pts_per_unit3"] = round(len(pts) / vol, 4)
    except Exception as e:
        logger.debug(f"Cloud stats error: {e}")
    return stats


def build_reconstruction_report(
    images,
    state,
    depth_maps: Dict,
    features: Dict,
    matches: Dict,
    verified: Dict,
    cfg: Dict,
    output_dir: str,
    ba_ran: bool = False,
) -> Dict:
    """Build and save the reconstruction quality report."""
    n_input       = len(images)
    n_registered  = state.n_registered()
    n_sparse_pts  = state.n_points()

    # Feature stats
    total_kp = sum(v.get("kp_count", 0) for v in features.values())

    # Match stats
    n_pairs      = len(matches)
    n_ver_pairs  = len(verified)
    mean_ver     = float(np.mean([v.get("n_verified", 0) for v in verified.values()])) \
                   if verified else 0

    # Depth stats
    n_depth_maps = len(depth_maps)
    depth_valid_pcts = []
    for dm in depth_maps.values():
        if dm is not None:
            valid_pct = 100.0 * (dm > 0).sum() / dm.size
            depth_valid_pcts.append(valid_pct)

    # Dense point count
    dense_ply = os.path.join(output_dir, cfg.get("output", {}).get("dense_ply", "points.ply"))
    dense_stats = compute_cloud_stats(dense_ply) if Path(dense_ply).exists() else {}

    sparse_ply = os.path.join(output_dir, cfg.get("output", {}).get("sparse_ply", "sparse_points.ply"))
    sparse_stats = compute_cloud_stats(sparse_ply) if Path(sparse_ply).exists() else {}

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input": {
            "n_images": n_input,
            "input_dir": str(images[0].parent) if images else "",
        },
        "reconstruction": {
            "n_registered_cameras": n_registered,
            "n_sparse_points": n_sparse_pts,
            "bundle_adjustment_ran": ba_ran,
        },
        "features": {
            "total_keypoints": total_kp,
            "mean_per_image": round(total_kp / max(n_input, 1), 1),
        },
        "matching": {
            "n_pairs_matched": n_pairs,
            "n_pairs_verified": n_ver_pairs,
            "mean_verified_matches": round(mean_ver, 1),
        },
        "depth": {
            "n_depth_maps": n_depth_maps,
            "mean_valid_depth_pct": round(float(np.mean(depth_valid_pcts)), 2)
                                    if depth_valid_pcts else 0,
        },
        "sparse_cloud": sparse_stats,
        "dense_cloud": dense_stats,
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "libraries": get_library_versions(),
            "gpu": detect_gpu(),
        },
    }

    out_json = os.path.join(output_dir, cfg.get("output", {}).get("report_json", "reconstruction_report.json"))
    safe_json_dump(report, out_json)

    out_txt = os.path.join(output_dir, cfg.get("output", {}).get("report_txt", "reconstruction_report.txt"))
    _write_txt_report(report, out_txt)

    return report


def _write_txt_report(report: Dict, path: str):
    """Write a human-readable text reconstruction report."""
    lines = [
        "=" * 60,
        "PIXEL-OPS Photogrammetry Reconstruction Report",
        "=" * 60,
        f"Timestamp:          {report.get('timestamp', 'N/A')}",
        "",
        "─── Input ───────────────────────────────────────────────",
        f"Input images:       {report['input']['n_images']}",
        "",
        "─── Reconstruction ──────────────────────────────────────",
        f"Registered cameras: {report['reconstruction']['n_registered_cameras']}",
        f"Sparse 3D points:   {report['reconstruction']['n_sparse_points']:,}",
        f"Bundle adjustment:  {report['reconstruction']['bundle_adjustment_ran']}",
        "",
        "─── Features ────────────────────────────────────────────",
        f"Total keypoints:    {report['features']['total_keypoints']:,}",
        f"Mean per image:     {report['features']['mean_per_image']:.0f}",
        "",
        "─── Matching ────────────────────────────────────────────",
        f"Pairs matched:      {report['matching']['n_pairs_matched']}",
        f"Pairs verified:     {report['matching']['n_pairs_verified']}",
        f"Mean verified matches: {report['matching']['mean_verified_matches']:.0f}",
        "",
        "─── Depth Maps ──────────────────────────────────────────",
        f"Depth maps:         {report['depth']['n_depth_maps']}",
        f"Mean valid depth:   {report['depth']['mean_valid_depth_pct']:.1f}%",
        "",
        "─── Dense Cloud ─────────────────────────────────────────",
    ]
    dc = report.get("dense_cloud", {})
    lines.append(f"Dense points:       {dc.get('point_count', 0):,}")
    bb = dc.get("bbox")
    if bb:
        lines.append(f"Bounding box:       {[round(v,3) for v in bb['min']]} → "
                     f"{[round(v,3) for v in bb['max']]}")
        lines.append(f"Extents:            {[round(v,3) for v in bb['extents']]}")
    c = dc.get("centroid")
    if c:
        lines.append(f"Centroid:           {[round(v,3) for v in c]}")
    lines += [
        "",
        "─── Environment ─────────────────────────────────────────",
        f"Python:             {report['environment']['python']}",
        f"OS:                 {report['environment']['os']}",
        f"GPU:                {report['environment']['gpu'].get('gpu_name', 'N/A')}",
        "=" * 60,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_submission_manifest(
    images, state, depth_maps, report: Dict,
    output_dir: str, cfg: Dict,
) -> Dict:
    """Build submission_manifest.json."""
    import datetime
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "input_image_count": len(images),
        "registered_camera_count": state.n_registered(),
        "sparse_point_count": state.n_points(),
        "dense_point_count": report.get("dense_cloud", {}).get("point_count", 0),
        "depth_maps_generated": len(depth_maps),
        "software_versions": get_library_versions(),
        "gpu_info": detect_gpu(),
        "execution_status": "completed",
    }
    out_path = os.path.join(output_dir, "submission_manifest.json")
    safe_json_dump(manifest, out_path)
    logger.info(f"[INFO] Submission manifest: {out_path}")
    return manifest
