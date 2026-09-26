#!/usr/bin/env python3
"""
pipeline_round2.py — PIXEL-OPS UAV Photogrammetry Pipeline, Round 2.

Extends the Round-1 image pipeline with MOV video input:
  4 MOV videos
      ↓ intelligent frame extraction (every N seconds)
      ↓ quality filtering (adaptive Laplacian-variance per video)
      ↓ near-duplicate filtering (thumbnail correlation)
      ↓ selected frames (JPEG) + video_frames_manifest.json
      ↓ [EXISTING ROUND-1 PIPELINE: SIFT → matching → geometry → SfM →
         triangulation → depth → fusion → dense point cloud → visualization]

Usage:
    python pipeline_round2.py --video-dir "path/to/videos" --output outputs_round2
    python pipeline_round2.py --video-dir "path/to/videos" --output outputs_round2 --dense
    python pipeline_round2.py --video-dir "path/to/videos" --output outputs_round2 --test-phase 3

Phases (--test-phase N stops after phase N):
    1 — syntax/import/dependency check
    2 — verify all MOV files and decode sample frames
    3 — frame extraction only (print counts, validate manifest)
    4 — cross-video connectivity test
    5 — small reconstruction (first 30 frames)
    6 — full Round-2 reconstruction (default when --test-phase not set)
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
        logger.info("[CFG] Config loaded: %s", path)
    else:
        logger.warning("Config not found at %s; using defaults", path)
        cfg = {}
    return cfg


def apply_cli_overrides(cfg, args):
    for section in ("image", "video", "features", "bundle_adjustment", "depth", "fusion"):
        if section not in cfg:
            cfg[section] = {}
    if getattr(args, "video_dir", None):
        cfg["video"]["video_dir"] = args.video_dir
    if getattr(args, "max_images", None) is not None:
        cfg["image"]["max_images"] = max(0, int(args.max_images))
    if getattr(args, "frame_interval", None) is not None:
        cfg["video"]["frame_interval_seconds"] = float(args.frame_interval)
    if getattr(args, "max_frames_per_video", None) is not None:
        cfg["video"]["max_frames_per_video"] = int(args.max_frames_per_video)
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
# Phase 1: Import / dependency check
# ---------------------------------------------------------------------------

def phase1_imports():
    logger.info("─" * 60)
    logger.info("PHASE 1: Syntax / Import / Dependency Check")
    logger.info("─" * 60)

    errors = []
    modules = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "open3d": "open3d",
        "scipy": "scipy",
        "PIL": "pillow",
        "yaml": "pyyaml",
        "matplotlib": "matplotlib",
    }
    for mod, pkg in modules.items():
        try:
            __import__(mod)
            logger.info("  [OK] %s", mod)
        except ImportError:
            logger.error("  [MISSING] %s  (pip install %s)", mod, pkg)
            errors.append(mod)

    src_modules = [
        "src.io_utils", "src.camera", "src.features", "src.matching",
        "src.geometry", "src.sfm", "src.triangulation", "src.depth",
        "src.fusion", "src.visualization", "src.evaluation",
        "src.video_input",
    ]
    for m in src_modules:
        try:
            __import__(m)
            logger.info("  [OK] %s", m)
        except Exception as e:
            logger.error("  [FAIL] %s: %s", m, e)
            errors.append(m)

    if errors:
        logger.error("Phase 1 FAILED: %d missing/broken modules", len(errors))
        return False
    logger.info("Phase 1 PASSED")
    return True


# ---------------------------------------------------------------------------
# Phase 2: Verify all MOV files + decode sample frames
# ---------------------------------------------------------------------------

def phase2_verify_videos(videos, output_dir):
    logger.info("─" * 60)
    logger.info("PHASE 2: Verify MOV Files")
    logger.info("─" * 60)

    from src.video_input import probe_video
    import cv2

    all_ok = True
    for i, v in enumerate(videos, 1):
        if not v.exists():
            logger.error("  Video %d MISSING: %s", i, v)
            all_ok = False
            continue
        probe = probe_video(v)
        if not probe["readable"]:
            logger.error("  Video %d NOT READABLE: %s", i, v.name)
            all_ok = False
            continue
        logger.info(
            "  Video %d: %s  %dx%d  %.2f fps  %d frames  %.1fs",
            i, v.name, probe["width"], probe["height"],
            probe["fps"], probe["frame_count"], probe["duration_seconds"],
        )
        # Decode one sample frame at 10s
        cap = cv2.VideoCapture(str(v))
        cap.set(cv2.CAP_PROP_POS_MSEC, 10000)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            logger.info("    -> Sample frame decoded OK: %dx%d", frame.shape[1], frame.shape[0])
        else:
            logger.warning("    -> Sample frame decode FAILED (video may be very short)")

    if all_ok:
        logger.info("Phase 2 PASSED")
    else:
        logger.error("Phase 2 FAILED")
    return all_ok


# ---------------------------------------------------------------------------
# Phase 3: Frame extraction
# ---------------------------------------------------------------------------

def phase3_extract_frames(videos, cfg, output_dir, force=False):
    logger.info("─" * 60)
    logger.info("PHASE 3: Frame Extraction")
    logger.info("─" * 60)

    from src.video_input import prepare_video_frames

    frame_paths, manifest = prepare_video_frames(
        videos, cfg, output_dir, force=force)

    if not frame_paths:
        logger.error("Phase 3 FAILED: no frames extracted")
        return None, None

    logger.info("─" * 40)
    logger.info("Frame extraction summary:")
    logger.info("  Total source frames: %d", manifest.get("total_source_frames", 0))
    logger.info("  Total candidates:   %d", manifest.get("total_candidates", 0))
    logger.info("  Total selected:     %d", manifest.get("total_selected", 0))
    logger.info("  Total removed:      %d", manifest.get("total_removed", 0))
    logger.info("")
    for vname, vstats in manifest.get("videos", {}).items():
        logger.info(
            "  %s -> %d/%d selected (blur-removed=%d, dup-removed=%d, thr=%.1f)",
            vname,
            vstats.get("n_written", 0),
            vstats.get("n_candidates", 0),
            vstats.get("n_blur_removed", 0),
            vstats.get("n_near_duplicate_removed", 0),
            vstats.get("quality", {}).get("threshold", 0),
        )

    # Validate manifest paths
    missing = [str(p) for p in frame_paths if not p.exists()]
    if missing:
        logger.error("Phase 3: %d frame files MISSING from disk!", len(missing))
        return None, None

    logger.info("Phase 3 PASSED: %d frames on disk", len(frame_paths))
    return frame_paths, manifest


# ---------------------------------------------------------------------------
# Phase 4: Cross-video connectivity test
# ---------------------------------------------------------------------------

def phase4_connectivity(frame_paths, manifest, cfg, output_dir):
    logger.info("─" * 60)
    logger.info("PHASE 4: Cross-Video Connectivity Test")
    logger.info("─" * 60)

    from src.io_utils import discover_images, ensure_dir
    from src.camera import build_cameras
    from src.features import extract_all_features
    from src.matching import match_all_pairs
    from src.geometry import verify_all_pairs

    conn_dir = ensure_dir(os.path.join(output_dir, "connectivity"))

    # Group frames by source video
    frame_meta = {Path(e["frame_path"]).name: e
                  for e in manifest.get("frames", [])}

    videos_in_manifest = {}
    for e in manifest.get("frames", []):
        vid = e["source_video"]
        if vid not in videos_in_manifest:
            videos_in_manifest[vid] = []
        videos_in_manifest[vid].append(Path(e["frame_path"]))

    vid_names = sorted(videos_in_manifest.keys())
    n_vids = len(vid_names)
    logger.info("  Videos in manifest: %d", n_vids)

    if n_vids < 2:
        logger.warning("  Only 1 video present; cross-video connectivity not applicable")
        return {}

    # Sample up to 5 representative frames per video for connectivity test
    sample_per_vid = 5
    sampled_paths = []
    sampled_vid_map = {}   # frame_name -> vid_name
    for vid in vid_names:
        paths = videos_in_manifest[vid]
        n = len(paths)
        idxs = np.linspace(0, n - 1, min(sample_per_vid, n), dtype=int)
        for i in idxs:
            p = paths[i]
            sampled_paths.append(p)
            sampled_vid_map[p.name] = vid

    logger.info("  Sampled %d frames for connectivity test", len(sampled_paths))

    # Extract features on sample
    sample_dir = str(conn_dir)
    exif_data = {p.name: {} for p in sampled_paths}
    proc_sizes = {}
    for p in sampled_paths:
        try:
            import cv2
            img = cv2.imread(str(p))
            if img is not None:
                h, w = img.shape[:2]
                proc_sizes[p.name] = (w, h)
                exif_data[p.name]["_original_dims"] = (w, h)
        except Exception:
            proc_sizes[p.name] = (1280, 720)

    cameras = build_cameras(sampled_paths, exif_data, {}, cfg, proc_sizes)
    features = extract_all_features(sampled_paths, cfg, sample_dir, debug=False)

    valid_names = [p.name for p in sampled_paths
                   if features.get(p.name, {}).get("valid", False)]
    logger.info("  Valid feature sets: %d/%d", len(valid_names), len(sampled_paths))

    # Use exhaustive matching for the small connectivity sample
    conn_cfg = dict(cfg)
    conn_cfg["matching"] = dict(cfg.get("matching", {}))
    conn_cfg["matching"]["strategy"] = "exhaustive"

    raw_matches = match_all_pairs(valid_names, features, conn_cfg, sample_dir, debug=False)
    verified = verify_all_pairs(raw_matches, cameras, conn_cfg, valid_names)

    # Build per-video-pair connectivity
    pair_matches = {}
    for (n1, n2), info in verified.items():
        v1 = sampled_vid_map.get(n1, "?")
        v2 = sampled_vid_map.get(n2, "?")
        if v1 == v2:
            continue   # Same video, skip intra-video pairs
        key = tuple(sorted([v1, v2]))
        if key not in pair_matches:
            pair_matches[key] = 0
        pair_matches[key] += info.get("n_verified", 0)

    logger.info("")
    logger.info("  Cross-video connectivity:")
    connectivity_graph = {}
    for i in range(n_vids):
        for j in range(i + 1, n_vids):
            v1, v2 = vid_names[i], vid_names[j]
            key = tuple(sorted([v1, v2]))
            matches = pair_matches.get(key, 0)
            status = "CONNECTED" if matches >= 10 else "DISCONNECTED"
            label = f"{v1} <-> {v2}"
            logger.info("    %s: %d verified matches  [%s]", label, matches, status)
            connectivity_graph[label] = {
                "video_a": v1, "video_b": v2,
                "verified_matches": matches,
                "connected": matches >= 10,
            }

    # Save connectivity report
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_videos": n_vids,
        "sample_frames_per_video": sample_per_vid,
        "pairs": connectivity_graph,
        "recommendation": (
            "single_reconstruction"
            if all(v["connected"] for v in connectivity_graph.values())
            else "check_disconnected_pairs"
        ),
    }
    conn_json = os.path.join(str(conn_dir), "connectivity_report.json")
    with open(conn_json, "w") as f:
        json.dump(report, f, indent=2)

    conn_txt = os.path.join(str(conn_dir), "connectivity_report.txt")
    with open(conn_txt, "w") as f:
        f.write("PIXEL-OPS Round-2 Cross-Video Connectivity Report\n")
        f.write("=" * 60 + "\n\n")
        for label, v in connectivity_graph.items():
            f.write(f"{label}: {v['verified_matches']} verified matches  "
                    f"[{'CONNECTED' if v['connected'] else 'DISCONNECTED'}]\n")
        f.write("\nRecommendation: " + report["recommendation"] + "\n")

    logger.info("")
    logger.info("  Connectivity report saved to: %s", conn_dir)
    logger.info("Phase 4 DONE")
    return connectivity_graph


# ---------------------------------------------------------------------------
# Phase 5 / 6: Run photogrammetry reconstruction
# ---------------------------------------------------------------------------

def phase_reconstruct(frame_paths, cfg, output_dir, max_frames=0, label="full"):
    logger.info("─" * 60)
    logger.info("PHASE: %s Reconstruction (%d frames)", label.upper(), len(frame_paths))
    logger.info("─" * 60)

    # Limit frames for small test run
    images = frame_paths
    if max_frames > 0 and len(images) > max_frames:
        idxs = np.linspace(0, len(images) - 1, max_frames, dtype=int)
        images = [images[i] for i in idxs]
        logger.info("  Using %d frames (subsampled from %d for %s run)",
                    len(images), len(frame_paths), label)

    # Import pipeline modules
    from src.io_utils import (build_dataset_report, read_exif,
                               exif_focal_length_pixels, image_dimensions,
                               ensure_dir, load_image_rgb)
    from src.camera import build_cameras
    from src.features import extract_all_features, build_feature_report
    from src.matching import match_all_pairs, build_matching_report
    from src.geometry import verify_all_pairs
    from src.sfm import run_incremental_sfm, assign_colors_to_landmarks
    from src.bundle_adjustment import run_bundle_adjustment
    from src.depth import run_depth_estimation
    from src.fusion import fuse_depth_maps, export_sparse_cloud, export_dense_cloud
    from src.visualization import save_pointcloud_screenshot, plot_camera_trajectory
    from src.evaluation import build_reconstruction_report, build_submission_manifest, detect_gpu

    # Create output sub-directories
    subdirs = ["", "sparse", "dense", "depth", "depth_preview",
               "visualization", "reports", "features", "matches", "debug"]
    for sub in subdirs:
        ensure_dir(os.path.join(output_dir, sub))

    max_dim = cfg["image"].get("resize_for_processing", 1280)

    # GPU info
    gpu_info = detect_gpu()
    logger.info("[GPU] %s", gpu_info.get("gpu_name", "CPU-only"))

    # EXIF / intrinsics
    exif_data = {}
    focal_lengths = {}
    proc_sizes = {}
    for img_path in images:
        name = img_path.name
        dims = image_dimensions(img_path)
        if dims is None:
            continue
        w, h = dims
        scale = min(max_dim / max(w, h), 1.0) if max_dim > 0 else 1.0
        pw, ph = int(w * scale), int(h * scale)
        proc_sizes[name] = (pw, ph)
        exif = read_exif(img_path)
        exif["_original_dims"] = (w, h)
        exif_data[name] = exif
        fl = exif_focal_length_pixels(exif, w, h)
        if fl:
            focal_lengths[name] = fl

    n_exif_fl = sum(1 for v in exif_data.values() if v.get("focal_length_mm"))
    logger.info("[EXIF] Focal length in EXIF: %d/%d (video frames have none; estimated)",
                n_exif_fl, len(images))

    build_dataset_report(images, cfg, output_dir, gpu_info)

    # Camera models
    cameras = build_cameras(images, exif_data, focal_lengths, cfg, proc_sizes)
    if not cameras:
        logger.error("No cameras initialized")
        return 1

    sample = next(iter(cameras.values()))
    logger.info("[CAM] fx=%.1f fy=%.1f cx=%.1f cy=%.1f  size=%dx%d",
                sample.fx, sample.fy, sample.cx, sample.cy,
                sample.width, sample.height)

    # Feature extraction
    features = extract_all_features(images, cfg, output_dir,
                                     debug=False)
    build_feature_report(features, output_dir, cfg)

    valid_names = [img.name for img in images
                   if features.get(img.name, {}).get("valid", False)]
    logger.info("[INFO] Valid images for matching: %d/%d", len(valid_names), len(images))

    if len(valid_names) < 2:
        logger.error("Too few images with valid features.")
        return 1

    # Feature matching
    raw_matches = match_all_pairs(valid_names, features, cfg, output_dir, debug=False)
    if not raw_matches:
        logger.error("No pairs matched.")
        return 1

    # Geometric verification
    verified = verify_all_pairs(raw_matches, cameras, cfg, valid_names)
    build_matching_report(verified, output_dir, cfg)

    if not verified:
        logger.error("No pairs passed geometric verification.")
        return 1
    logger.info("[INFO] Verified pairs: %d / %d matched", len(verified), len(raw_matches))

    # Incremental SfM
    state = run_incremental_sfm(images, cameras, features, verified, cfg)
    n_reg = state.n_registered()
    n_pts = state.n_points()
    logger.info("[SFM] Registered: %d/%d cameras,  %d sparse points",
                n_reg, len(images), n_pts)

    if n_reg < 2:
        logger.error("Fewer than 2 cameras registered - reconstruction failed.")
        return 1

    # Color assignment
    logger.info("[INFO] Assigning colours to sparse points...")
    images_rgb = {}
    for img_path in images:
        name = img_path.name
        if cameras.get(name) and cameras[name].registered:
            rgb = load_image_rgb(img_path, max_dim)
            if rgb is not None:
                images_rgb[name] = rgb
    assign_colors_to_landmarks(state, images_rgb, cameras)

    # Optional bundle adjustment
    ba_ran = False
    if cfg.get("bundle_adjustment", {}).get("enabled", False):
        try:
            ba_ran = run_bundle_adjustment(state, features, cfg)
        except Exception as e:
            logger.warning("[BA] Bundle adjustment failed: %s", e)

    # Export sparse cloud
    sparse_ply = os.path.join(output_dir, cfg.get("output", {}).get(
        "sparse_ply", "sparse/sparse_points.ply"))
    ensure_dir(str(Path(sparse_ply).parent))
    export_sparse_cloud(state, sparse_ply)

    # Camera trajectory
    try:
        plot_camera_trajectory(state, output_dir)
    except Exception as e:
        logger.debug("Trajectory plot error: %s", e)

    # Depth estimation
    depth_maps = {}
    if cfg.get("depth", {}).get("enabled", True):
        try:
            depth_maps = run_depth_estimation(images, state, cameras, cfg, output_dir)
        except Exception as e:
            logger.warning("[DEPTH] Depth estimation error: %s", e)

    # Dense fusion
    dense_pcd = None
    if depth_maps:
        try:
            dense_pcd = fuse_depth_maps(depth_maps, cameras, images, cfg, output_dir)
        except Exception as e:
            logger.warning("[FUSION] Fusion error: %s", e)

    dense_ply = os.path.join(output_dir, cfg.get("output", {}).get(
        "dense_ply", "dense/point_cloud.ply"))
    ensure_dir(str(Path(dense_ply).parent))

    if dense_pcd is not None and len(dense_pcd.points) > 0:
        export_dense_cloud(dense_pcd, dense_ply)
    else:
        logger.warning("[INFO] Dense fusion empty; copying sparse cloud as point_cloud.ply")
        if Path(sparse_ply).exists():
            shutil.copy2(sparse_ply, dense_ply)

    # Visualization (3D dotted point cloud screenshot)
    screenshot_path = os.path.join(output_dir, cfg.get("output", {}).get(
        "screenshot", "visualization/reconstruction_screenshot.png"))
    ensure_dir(str(Path(screenshot_path).parent))
    try:
        save_pointcloud_screenshot(dense_ply, screenshot_path, state=state, cfg=cfg)
    except Exception as e:
        logger.warning("[VIS] Screenshot error: %s", e)
        try:
            save_pointcloud_screenshot(sparse_ply, screenshot_path, state=state, cfg=cfg)
        except Exception:
            pass

    # Quality report
    report = build_reconstruction_report(
        images, state, depth_maps, features,
        raw_matches, verified, cfg, output_dir, ba_ran=ba_ran)

    # Log summary
    dc = report.get("dense_cloud", {})
    logger.info("")
    logger.info("=" * 60)
    logger.info("RECONSTRUCTION COMPLETE (%s)", label)
    logger.info("=" * 60)
    logger.info("Input frames:        %d", len(images))
    logger.info("Registered cameras:  %d / %d  (%.0f%%)",
                n_reg, len(images), 100.0 * n_reg / max(len(images), 1))
    logger.info("Sparse 3D points:    %d", n_pts)
    logger.info("Depth maps:          %d", len(depth_maps))
    logger.info("Dense points:        %d", dc.get("point_count", 0))
    logger.info("Sparse PLY:          %s", sparse_ply)
    logger.info("Dense PLY:           %s", dense_ply)
    logger.info("Screenshot:          %s", screenshot_path)
    logger.info("=" * 60)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    t_total = time.time()

    cfg = load_config(getattr(args, "config", None))
    cfg = apply_cli_overrides(cfg, args)

    output_dir = str(args.output)
    test_phase = getattr(args, "test_phase", 6) or 6

    logger.info("=" * 60)
    logger.info("PIXEL-OPS Round-2 Photogrammetry Pipeline")
    logger.info("=" * 60)
    logger.info("Output:     %s", output_dir)
    logger.info("Test phase: %d", test_phase)

    from src.io_utils import ensure_dir
    ensure_dir(output_dir)

    # ── Phase 1 ────────────────────────────────────────────────────────────
    if not phase1_imports():
        return 1
    if test_phase <= 1:
        return 0

    # ── Discover videos ────────────────────────────────────────────────────
    from src.video_input import discover_videos

    video_dir = getattr(args, "video_dir", None) or cfg.get("video", {}).get(
        "video_dir", "")
    if not video_dir:
        logger.error("No video directory specified. Use --video-dir or set video.video_dir in config.yaml")
        return 1

    ext = cfg.get("video", {}).get("extensions", None)
    videos = discover_videos(str(video_dir), ext)
    if not videos:
        logger.error("No video files found in %s", video_dir)
        return 1
    logger.info("[VIDEO] Found %d video(s) in %s", len(videos), video_dir)

    # ── Phase 2 ────────────────────────────────────────────────────────────
    if not phase2_verify_videos(videos, output_dir):
        return 1
    if test_phase <= 2:
        return 0

    # ── Phase 3 ────────────────────────────────────────────────────────────
    force_extract = getattr(args, "force_extract", False)
    frame_paths, manifest = phase3_extract_frames(
        videos, cfg, output_dir, force=force_extract)
    if frame_paths is None:
        return 1
    if test_phase <= 3:
        return 0

    # ── Phase 4 ────────────────────────────────────────────────────────────
    connectivity = phase4_connectivity(frame_paths, manifest, cfg, output_dir)
    if test_phase <= 4:
        return 0

    # ── Phase 5 ────────────────────────────────────────────────────────────
    if test_phase == 5:
        rc = phase_reconstruct(
            frame_paths, cfg, output_dir, max_frames=30, label="small-test")
        return rc

    # ── Phase 6 (full) ─────────────────────────────────────────────────────
    rc = phase_reconstruct(frame_paths, cfg, output_dir, max_frames=0, label="full")
    elapsed = time.time() - t_total
    logger.info("Total wall time: %.1fs (%.1fmin)", elapsed, elapsed / 60)
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="PIXEL-OPS Round-2 UAV Photogrammetry Pipeline (MOV video input)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-dir", "-v", dest="video_dir", default=None,
                        help="Directory containing .MOV video files")
    parser.add_argument("--output", "-o", default="outputs_round2",
                        help="Output directory for results")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config.yaml")
    parser.add_argument("--frame-interval", type=float, default=None,
                        dest="frame_interval",
                        help="Seconds between extracted frames (default: 2.0)")
    parser.add_argument("--max-frames-per-video", type=int, default=None,
                        dest="max_frames_per_video",
                        help="Max frames to keep per video (0=all)")
    parser.add_argument("--max-images", type=int, default=None,
                        dest="max_images",
                        help="Max total frames to use for reconstruction (0=all)")
    parser.add_argument("--feature-method", choices=["SIFT", "ORB"],
                        default=None, dest="feature_method")
    parser.add_argument("--use-bundle-adjustment", action="store_true",
                        dest="use_bundle_adjustment")
    parser.add_argument("--dense", action="store_true", default=None)
    parser.add_argument("--no-dense", dest="dense", action="store_false")
    parser.add_argument("--voxel-size", type=float, default=None, dest="voxel_size")
    parser.add_argument("--force-extract", action="store_true", dest="force_extract",
                        help="Force re-extraction even if manifest cache is valid")
    parser.add_argument("--test-phase", type=int, default=None, dest="test_phase",
                        help="Stop after this phase (1-6). 6=full pipeline (default).")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dense is None:
        args.dense = True
    sys.exit(run(args))
