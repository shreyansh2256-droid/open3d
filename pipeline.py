#!/usr/bin/env python3
"""
pipeline.py - PIXEL-OPS UAV Photogrammetry Pipeline
End-to-end 3D reconstruction from overlapping UAV images.

Usage:
    python pipeline.py --input data/lighthouse --output submission
    python pipeline.py --input <dataset> --output <out_dir> --dense --visualize
"""
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import yaml
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path=None):
    default_path = Path(__file__).parent / "config.yaml"
    path = config_path or str(default_path)
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("[INFO] Config loaded: %s", path)
    else:
        logger.warning("Config not found at %s; using defaults", path)
        cfg = {}
    return cfg


def apply_cli_overrides(cfg, args):
    for section in ("image", "features", "bundle_adjustment", "depth", "fusion"):
        if section not in cfg:
            cfg[section] = {}
    # 0 explicitly means "all images" and must override the config value.
    if getattr(args, "max_images", None) is not None:
        cfg["image"]["max_images"] = max(0, int(args.max_images))
    if getattr(args, "feature_method", None):
        cfg["features"]["method"] = args.feature_method
    if getattr(args, "use_bundle_adjustment", False):
        cfg["bundle_adjustment"]["enabled"] = True
    if getattr(args, "dense", None) is not None:
        cfg["depth"]["enabled"] = args.dense
    if getattr(args, "voxel_size", None) is not None:
        cfg["fusion"]["voxel_size"] = args.voxel_size
    return cfg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    t_total = time.time()

    cfg = load_config(getattr(args, "config", None))
    cfg = apply_cli_overrides(cfg, args)

    input_dir = str(args.input)
    output_dir = str(args.output)

    logger.info("=" * 60)
    logger.info("PIXEL-OPS Photogrammetry Pipeline")
    logger.info("=" * 60)
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)

    # -- Import all modules ---------------------------------------------------
    try:
        from src.io_utils import (discover_images, build_dataset_report,
                                   read_exif, exif_focal_length_pixels,
                                   image_dimensions, ensure_dir, load_image_rgb)
        from src.camera import build_cameras
        from src.features import extract_all_features, build_feature_report
        from src.matching import match_all_pairs, build_matching_report
        from src.geometry import verify_all_pairs
        from src.sfm import run_incremental_sfm, assign_colors_to_landmarks
        from src.bundle_adjustment import run_bundle_adjustment
        from src.depth import run_depth_estimation
        from src.fusion import fuse_depth_maps, export_sparse_cloud, export_dense_cloud
        from src.visualization import (save_pointcloud_screenshot,
                                        plot_camera_trajectory)
        from src.evaluation import (build_reconstruction_report,
                                     build_submission_manifest, detect_gpu)
    except ImportError as e:
        logger.error("Import error: %s\nRun: pip install -r requirements.txt", e)
        return 1

    # -- Directories ----------------------------------------------------------
    for sub in ("", "features", "matches", "depth", "depth_preview",
                "debug", "depth_maps"):
        ensure_dir(os.path.join(output_dir, sub))

    # -- GPU detection --------------------------------------------------------
    gpu_info = detect_gpu()
    if gpu_info["has_cuda"]:
        logger.info("[INFO] GPU: %s (%d MB)", gpu_info["gpu_name"], gpu_info["vram_mb"])
    else:
        logger.info("[INFO] No CUDA GPU - running on CPU")

    # -- Phase 1: Image discovery ---------------------------------------------
    extensions = cfg["image"].get("extensions",
                                   [".jpg", ".jpeg", ".JPG", ".JPEG"])
    max_images = cfg["image"].get("max_images", 0)
    max_dim = cfg["image"].get("resize_for_processing", 1600)

    images = discover_images(input_dir, extensions, max_images)
    if len(images) < 2:
        logger.error("Found only %d image(s) - need at least 2.", len(images))
        return 1

    logger.info("[INFO] Using %d images (max_dim=%d)", len(images), max_dim)

    # -- Phase 2: EXIF / intrinsics -------------------------------------------
    exif_data = {}
    focal_lengths = {}
    proc_sizes = {}
    orig_dims = {}

    for img_path in images:
        name = img_path.name
        dims = image_dimensions(img_path)
        if dims is None:
            continue
        w, h = dims
        orig_dims[name] = (w, h)
        scale = min(max_dim / max(w, h), 1.0) if max_dim > 0 else 1.0
        pw, ph = int(w * scale), int(h * scale)
        proc_sizes[name] = (pw, ph)
        exif = read_exif(img_path)
        exif["_original_dims"] = (w, h)
        exif_data[name] = exif
        fl = exif_focal_length_pixels(exif, w, h)
        if fl:
            focal_lengths[name] = fl  # Full-res px; build_cameras scales via orig_dims

    # Log EXIF findings
    n_exif_fl = sum(1 for v in exif_data.values() if v.get("focal_length_mm"))
    n_exif_make = sum(1 for v in exif_data.values() if v.get("make"))
    logger.info("[EXIF] Focal length in EXIF: %d/%d images", n_exif_fl, len(images))
    logger.info("[EXIF] Camera make/model in EXIF: %d/%d images", n_exif_make, len(images))
    if n_exif_fl == 0:
        logger.warning("[EXIF] No focal length found - using estimated intrinsics (not metric)")

    # Dataset report
    build_dataset_report(images, cfg, output_dir, gpu_info)

    # -- Phase 3: Camera models -----------------------------------------------
    cameras = build_cameras(images, exif_data, focal_lengths, cfg, proc_sizes)
    if not cameras:
        logger.error("No cameras could be initialized")
        return 1

    sample = next(iter(cameras.values()))
    logger.info("[CAM] fx=%.1f fy=%.1f cx=%.1f cy=%.1f  size=%dx%d",
                sample.fx, sample.fy, sample.cx, sample.cy,
                sample.width, sample.height)

    # -- Phase 4: Feature extraction ------------------------------------------
    features = extract_all_features(images, cfg, output_dir,
                                     debug=getattr(args, "debug", False))
    build_feature_report(features, output_dir, cfg)

    valid_names = [img.name for img in images
                   if features.get(img.name, {}).get("valid", False)]
    logger.info("[INFO] Valid images for matching: %d/%d", len(valid_names), len(images))

    if len(valid_names) < 2:
        logger.error("Too few images with valid features.")
        return 1

    # -- Phase 5: Feature matching --------------------------------------------
    raw_matches = match_all_pairs(valid_names, features, cfg, output_dir,
                                   debug=getattr(args, "debug", False))
    if not raw_matches:
        logger.error("No image pairs could be matched.")
        return 1

    # -- Phase 6: Geometric verification --------------------------------------
    verified = verify_all_pairs(raw_matches, cameras, cfg, valid_names)
    build_matching_report(verified, output_dir, cfg)

    if not verified:
        logger.error("No pairs passed geometric verification.")
        return 1

    logger.info("[INFO] Verified pairs: %d / %d matched",
                len(verified), len(raw_matches))

    # -- Phase 7: Incremental SfM --------------------------------------------
    state = run_incremental_sfm(images, cameras, features, verified, cfg)

    n_reg = state.n_registered()
    n_pts = state.n_points()
    logger.info("[SFM] Registered: %d/%d cameras,  %d sparse points",
                n_reg, len(images), n_pts)

    if n_reg < 2:
        logger.error("Fewer than 2 cameras registered - reconstruction failed.")
        return 1

    # -- Phase 8: Colour assignment -------------------------------------------
    logger.info("[INFO] Assigning colours to sparse points...")
    images_rgb = {}
    for img_path in images:
        name = img_path.name
        if cameras.get(name) and cameras[name].registered:
            rgb = load_image_rgb(img_path, max_dim)
            if rgb is not None:
                images_rgb[name] = rgb

    assign_colors_to_landmarks(state, images_rgb, cameras)

    # -- Phase 9: Bundle adjustment (optional) --------------------------------
    ba_ran = False
    if cfg["bundle_adjustment"].get("enabled", False):
        try:
            ba_ran = run_bundle_adjustment(state, features, cfg)
        except Exception as e:
            logger.warning("[BA] Bundle adjustment failed: %s", e)

    # -- Phase 10: Export sparse cloud ----------------------------------------
    sparse_ply = os.path.join(output_dir,
                               cfg.get("output", {}).get("sparse_ply",
                                                          "sparse_points.ply"))
    export_sparse_cloud(state, sparse_ply)

    # -- Phase 11: Camera trajectory ------------------------------------------
    if getattr(args, "debug", False) or getattr(args, "visualize", False):
        try:
            plot_camera_trajectory(state, output_dir)
        except Exception as e:
            logger.debug("Trajectory plot error: %s", e)

    # -- Phase 12: Depth estimation -------------------------------------------
    depth_maps = {}
    if cfg["depth"].get("enabled", True):
        try:
            depth_maps = run_depth_estimation(images, state, cameras, cfg, output_dir)
        except Exception as e:
            logger.warning("[DEPTH] Depth estimation error: %s", e)
            import traceback
            logger.debug(traceback.format_exc())

    # Validate depth maps
    if depth_maps:
        _validate_depth_maps(depth_maps, cfg)

    # -- Phase 13: Dense fusion -----------------------------------------------
    dense_pcd = None
    if depth_maps:
        try:
            dense_pcd = fuse_depth_maps(depth_maps, cameras, images, cfg, output_dir)
        except Exception as e:
            logger.warning("[FUSION] Fusion error: %s", e)

    dense_ply = os.path.join(output_dir,
                              cfg.get("output", {}).get("dense_ply", "points.ply"))
    if dense_pcd is not None and len(dense_pcd.points) > 0:
        export_dense_cloud(dense_pcd, dense_ply)
        _validate_point_cloud(dense_ply)
    else:
        logger.warning("[INFO] Dense fusion empty; copying sparse cloud as points.ply")
        if Path(sparse_ply).exists():
            shutil.copy2(sparse_ply, dense_ply)

    # -- Phase 14: Depth overview contact sheet -------------------------------
    if depth_maps:
        try:
            _make_depth_overview(depth_maps, images, output_dir)
        except Exception as e:
            logger.debug("Depth overview error: %s", e)

    # -- Phase 15: Visualization / screenshot ---------------------------------
    screenshot_path = os.path.join(
        output_dir,
        cfg.get("output", {}).get("screenshot", "reconstruction_screenshot.png"))
    try:
        save_pointcloud_screenshot(dense_ply, screenshot_path, state=state, cfg=cfg)
    except Exception as e:
        logger.warning("[VIS] Screenshot error: %s", e)
        try:
            save_pointcloud_screenshot(sparse_ply, screenshot_path,
                                        state=state, cfg=cfg)
        except Exception:
            pass

    # -- Phase 16: Quality report ---------------------------------------------
    report = build_reconstruction_report(
        images, state, depth_maps, features,
        raw_matches, verified, cfg, output_dir, ba_ran=ba_ran)

    # -- Phase 17: Documentation PDF ------------------------------------------
    doc_path = os.path.join(output_dir, "documentation.pdf")
    try:
        _generate_pdf(doc_path, report, state, depth_maps, cfg)
    except Exception as e:
        logger.error("[DOC] PDF generation failed: %s", e)
        import traceback
        logger.debug(traceback.format_exc())

    # Verify PDF
    if Path(doc_path).exists() and Path(doc_path).stat().st_size > 0:
        logger.info("[DOC] PDF OK: %s (%.1f KB)",
                    doc_path, Path(doc_path).stat().st_size / 1024)
    else:
        logger.error("[DOC] PDF missing or empty!")

    # -- Phase 18: Submission manifest ----------------------------------------
    manifest = build_submission_manifest(
        images, state, depth_maps, report, output_dir, cfg)

    # -- Phase 19: Submission package -----------------------------------------
    _build_submission(output_dir, images, state, depth_maps, report, cfg)

    # -- Summary --------------------------------------------------------------
    elapsed = time.time() - t_total
    dc = report.get("dense_cloud", {})
    logger.info("")
    logger.info("=" * 60)
    logger.info("RECONSTRUCTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time:          %.1fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("Input images:        %d", len(images))
    logger.info("Registered cameras:  %d / %d  (%.0f%%)",
                n_reg, len(images), 100.0 * n_reg / max(len(images), 1))
    logger.info("Sparse 3D points:    %d", n_pts)
    logger.info("Depth maps:          %d", len(depth_maps))
    logger.info("Dense points:        %d", dc.get("point_count", 0))
    logger.info("Output directory:    %s", output_dir)
    logger.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Depth map validation
# ---------------------------------------------------------------------------

def _validate_depth_maps(depth_maps, cfg):
    """Check depth maps are not degenerate."""
    depth_cfg = cfg.get("depth", {})
    d_min_hyp = float(depth_cfg.get("d_min", 0.5))
    d_max_hyp = float(depth_cfg.get("d_max", 50.0))
    for name, dm in depth_maps.items():
        if dm is None:
            logger.warning("[DEPTH] %s: None depth map", name)
            continue
        valid = dm > 0
        if not valid.any():
            logger.warning("[DEPTH] %s: ALL pixels invalid", name)
            continue
        d_vals = dm[valid]
        pct_at_min = 100.0 * (d_vals <= d_min_hyp + 0.01).sum() / len(d_vals)
        pct_at_max = 100.0 * (d_vals >= d_max_hyp - 0.01).sum() / len(d_vals)
        std = d_vals.std()
        logger.info(
            "[DEPTH] %s: valid=%.1f%%  min=%.2f max=%.2f "
            "median=%.2f std=%.2f  at_min=%.1f%% at_max=%.1f%%",
            name, 100.0 * valid.mean(),
            d_vals.min(), d_vals.max(), float(np.median(d_vals)), std,
            pct_at_min, pct_at_max)
        if pct_at_min > 50 or pct_at_max > 50:
            logger.warning("[DEPTH] %s: majority of pixels clipped - "
                           "depth estimator may be failing", name)
        if std < 0.01:
            logger.warning("[DEPTH] %s: depth std=%.4f - appears constant", name, std)


# ---------------------------------------------------------------------------
# Point cloud validation
# ---------------------------------------------------------------------------

def _validate_point_cloud(ply_path):
    """Load and validate the point cloud."""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)
        n = len(pts)
        if n == 0:
            logger.error("[PCD] %s: EMPTY point cloud!", ply_path)
            return
        nan_count = np.isnan(pts).any(axis=1).sum()
        inf_count = np.isinf(pts).any(axis=1).sum()
        bb_min = pts.min(axis=0)
        bb_max = pts.max(axis=0)
        centroid = pts.mean(axis=0)
        has_color = pcd.has_colors()
        logger.info(
            "[PCD] %s: %d pts  NaN=%d  Inf=%d  has_rgb=%s",
            Path(ply_path).name, n, nan_count, inf_count, has_color)
        logger.info(
            "[PCD] bbox_min=[%.3f %.3f %.3f]  bbox_max=[%.3f %.3f %.3f]  "
            "centroid=[%.3f %.3f %.3f]",
            *bb_min, *bb_max, *centroid)
    except Exception as e:
        logger.warning("[PCD] Validation error: %s", e)


# ---------------------------------------------------------------------------
# Depth overview contact sheet
# ---------------------------------------------------------------------------

def _make_depth_overview(depth_maps, images, output_dir):
    """Create a contact sheet showing depth maps alongside reference images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    names = list(depth_maps.keys())[:6]  # up to 6
    n = len(names)
    if n == 0:
        return

    img_path_dict = {p.name: p for p in images}
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    for j, name in enumerate(names):
        # Top row: reference image thumbnail
        p = img_path_dict.get(name)
        if p and p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                thumb_w = 400
                th = int(h * thumb_w / w)
                img = cv2.resize(img, (thumb_w, th))
                axes[0][j].imshow(img)
        axes[0][j].set_title(name[:20], fontsize=7)
        axes[0][j].axis("off")

        # Bottom row: depth map
        dm = depth_maps[name]
        valid = dm > 0
        dm_vis = dm.copy().astype(np.float32)
        dm_vis[~valid] = np.nan
        im = axes[1][j].imshow(dm_vis, cmap="plasma", aspect="auto")
        pct_valid = 100.0 * valid.mean()
        axes[1][j].set_title(f"Depth  valid={pct_valid:.0f}%", fontsize=7)
        axes[1][j].axis("off")
        plt.colorbar(im, ax=axes[1][j], fraction=0.046)

    fig.suptitle("PIXEL-OPS Depth Maps Overview", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "depth_maps_overview.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("[INFO] Depth overview: %s", out)


# ---------------------------------------------------------------------------
# PDF generation (ASCII-only text to avoid font encoding issues)
# ---------------------------------------------------------------------------

def _generate_pdf(path, report, state, depth_maps, cfg):
    """Generate 1-page process documentation PDF using fpdf2."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)

    # Title bar
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_fill_color(20, 20, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 11, "PIXEL-OPS Lighthouse Photogrammetry Pipeline",
             border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, align="C")
    pdf.ln(2)
    pdf.set_text_color(0, 0, 0)

    def section(title):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(220, 230, 245)
        pdf.cell(0, 6, title, border=0,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.set_font("Helvetica", "", 8)

    def body(text):
        # Strip non-latin1 chars to be safe with Helvetica
        safe = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 4.5, safe)
        pdf.ln(1)

    section("1. Objective")
    body("Reconstruct a 3D model of a lighthouse from overlapping UAV photographs "
         "using a custom photogrammetry pipeline built with OpenCV, NumPy, SciPy, "
         "and Open3D. No pre-built photogrammetry software (COLMAP, Meshroom) is used.")

    section("2. Input Dataset")
    n_imgs = report["input"]["n_images"]
    n_exif = sum(1 for v in report.get("exif_summary", {}).values()
                 if v) if "exif_summary" in report else "unknown"
    body(f"Images: {n_imgs} high-resolution JPEG files (Lighthouse Survey DB4). "
         f"EXIF focal length availability: see dataset_report.json. "
         f"Intrinsics estimated from image dimensions when EXIF unavailable. "
         f"Scale is RELATIVE (not metric) unless EXIF+sensor data confirms otherwise.")

    section("3. Pipeline Steps")
    body("1) Image discovery (excludes outputs/, submission/, debug/ dirs)\n"
         "2) EXIF parsing + pinhole camera intrinsic estimation\n"
         "3) SIFT feature extraction (up to 4000 kp/image)\n"
         "4) FLANN matching + Lowe ratio test (threshold 0.75)\n"
         "5) Fundamental/Essential matrix RANSAC geometric verification\n"
         "6) Initial pair selection (max inlier count x ratio score)\n"
         "7) Two-view init: identity + relative pose, DLT triangulation\n"
         "8) Incremental SfM: PnP RANSAC registration + triangulation\n"
         "9) [Optional] SciPy least_squares bundle adjustment\n"
         "10) Semi-dense plane-sweep ZNCC depth estimation (vectorised)\n"
         "11) Multi-view depth back-projection + adaptive fusion\n"
         "12) Statistical + radius outlier removal (scale-adaptive)\n"
         "13) Sparse + dense PLY export, RGB colouring, visualisation")

    section("4. Feature Extraction")
    total_kp = report["features"]["total_keypoints"]
    mean_kp = report["features"]["mean_per_image"]
    body(f"Method: SIFT (Scale-Invariant Feature Transform). "
         f"Total keypoints: {total_kp:,}. Mean per image: {mean_kp:.0f}. "
         f"Invariant to scale, rotation, and illumination changes.")

    section("5. Feature Matching")
    n_pairs = report["matching"]["n_pairs_matched"]
    n_ver = report["matching"]["n_pairs_verified"]
    mean_ver = report["matching"]["mean_verified_matches"]
    body(f"FLANN approximate nearest-neighbour matching. "
         f"Pairs matched: {n_pairs}. Pairs verified: {n_ver}. "
         f"Mean verified matches per pair: {mean_ver:.0f}. "
         f"Strategy: sequential window (configurable).")

    section("6. Geometric Verification")
    body("OpenCV findFundamentalMat (RANSAC, 1.5px threshold) then "
         "findEssentialMat + recoverPose for relative camera pose recovery. "
         "Minimum inlier ratio 0.20 enforced to reject degenerate configurations.")

    section("7. Structure-from-Motion")
    n_reg = report["reconstruction"]["n_registered_cameras"]
    n_sparse = report["reconstruction"]["n_sparse_points"]
    body(f"Incremental SfM with PnP RANSAC (solvePnPRansac). "
         f"Registered: {n_reg} cameras. Sparse points: {n_sparse:,}. "
         f"New 3D points triangulated for each newly registered camera. "
         f"Reprojection filter applied after each registration.")

    section("8. Depth Estimation")
    n_dm = report["depth"]["n_depth_maps"]
    valid_pct = report["depth"]["mean_valid_depth_pct"]
    body(f"Plane-sweep ZNCC depth estimation. References: {n_dm}. "
         f"Mean valid pixels: {valid_pct:.1f}%. "
         f"Depth at 600px resolution using properly scaled camera intrinsics. "
         f"OpenCV boxFilter for O(1) per-pixel ZNCC. Bilateral fill for gaps.")

    section("9. Point Cloud Fusion")
    dc = report.get("dense_cloud", {})
    dense_n = dc.get("point_count", 0)
    body(f"Depth back-projection using scaled pinhole model. "
         f"Raw combined: {8*50000:,} pts. "
         f"Adaptive voxel downsampling + statistical + radius outlier removal. "
         f"Final dense cloud: {dense_n:,} points (RGB coloured from source images).")

    section("10. Results")
    body(f"sparse_points.ply : {n_sparse:,} SfM sparse points\n"
         f"points.ply        : {dense_n:,} dense fused points\n"
         f"depth_map_XX.npy/png : {n_dm} depth maps\n"
         f"reconstruction_screenshot.png : automated render\n"
         f"reconstruction_report.json : full quality metrics\n"
         f"documentation.pdf : this document")

    section("11. Limitations")
    body("- Relative scale (not metric) without GPS/GCP ground truth\n"
         "- Depth is semi-dense (not full MVS); GPU would enable denser results\n"
         "- Some cameras may not register if image overlap is insufficient\n"
         "- Bundle adjustment disabled by default (enable: --use-bundle-adjustment)\n"
         "- Textureless regions (sky, plain concrete) produce sparse coverage")

    section("12. Adaptability to New Dataset")
    body("Dataset-agnostic: no hardcoded filenames, image counts, or camera params. "
         "Run on any folder: python pipeline.py --input <new_folder> --output <out> --dense. "
         "All thresholds in config.yaml. EXIF auto-detected. Pair selection is automatic.")

    from src.io_utils import ensure_dir
    ensure_dir(str(Path(path).parent))
    pdf.output(path)
    logger.info("[DOC] PDF generated: %s", path)


# ---------------------------------------------------------------------------
# Submission package
# ---------------------------------------------------------------------------

def _build_submission(output_dir, images, state, depth_maps, report, cfg):
    """Assemble the submission/ directory."""
    sub_dir = Path(output_dir).parent / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    depth_sub = sub_dir / "depth_maps"
    depth_sub.mkdir(exist_ok=True)

    # Copy key output files
    pairs = [
        (Path(output_dir) / cfg.get("output", {}).get("dense_ply", "points.ply"),
         sub_dir / "points.ply"),
        (Path(output_dir) / cfg.get("output", {}).get("sparse_ply", "sparse_points.ply"),
         sub_dir / "sparse_points.ply"),
        (Path(output_dir) / cfg.get("output", {}).get("screenshot",
                                                        "reconstruction_screenshot.png"),
         sub_dir / "reconstruction_screenshot.png"),
        (Path(output_dir) / cfg.get("output", {}).get("report_json",
                                                        "reconstruction_report.json"),
         sub_dir / "reconstruction_report.json"),
        (Path(output_dir) / "documentation.pdf", sub_dir / "documentation.pdf"),
        (Path(output_dir) / "submission_manifest.json",
         sub_dir / "submission_manifest.json"),
        (Path(__file__), sub_dir / "pipeline.py"),
        (Path(__file__).parent / "config.yaml", sub_dir / "config.yaml"),
        (Path(__file__).parent / "requirements.txt", sub_dir / "requirements.txt"),
        (Path(__file__).parent / "README.md", sub_dir / "README.md"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(str(src), str(dst))

    # Copy src/ module (exclude pycache)
    src_mod = Path(__file__).parent / "src"
    dst_mod = sub_dir / "src"
    if src_mod.exists():
        if dst_mod.exists():
            shutil.rmtree(str(dst_mod))
        shutil.copytree(str(src_mod), str(dst_mod),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    # Copy depth maps
    depth_dir = Path(output_dir) / "depth"
    if depth_dir.exists():
        for f in list(depth_dir.glob("depth_map_*.png")) + list(depth_dir.glob("depth_map_*.npy")):
            shutil.copy2(str(f), str(depth_sub / f.name))

    # Copy overview if present
    ov = Path(output_dir) / "depth_maps_overview.png"
    if ov.exists():
        shutil.copy2(str(ov), str(sub_dir / "depth_maps_overview.png"))

    logger.info("[INFO] Submission package: %s", sub_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PIXEL-OPS UAV Photogrammetry Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input directory containing images")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Output directory for results")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config.yaml")
    parser.add_argument("--max-images", type=int, default=0,
                        dest="max_images",
                        help="Max images to process (0=all)")
    parser.add_argument("--feature-method", choices=["SIFT", "ORB"],
                        default=None, dest="feature_method")
    parser.add_argument("--use-bundle-adjustment", action="store_true",
                        dest="use_bundle_adjustment")
    parser.add_argument("--dense", action="store_true", default=None)
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--voxel-size", type=float, default=None, dest="voxel_size")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        logger.error("Input path does not exist: %s", args.input)
        sys.exit(1)
    if args.dense is None:
        args.dense = True
    sys.exit(run_pipeline(args))
