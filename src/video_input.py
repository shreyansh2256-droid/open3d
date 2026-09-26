"""
video_input.py - VIDEO -> SELECTED FRAMES input layer (PIXEL-OPS Round 2).

Converts .MOV/.MP4/... video files into a small set of good, non-redundant
frames plus provenance metadata, so the existing Round-1 image pipeline
(SIFT -> matching -> geometric verification -> incremental SfM -> triangulation
-> depth -> dense point cloud) runs on them unchanged.

    .MOV
      -> candidate extraction   (timestamp/frame seeking, every N seconds)
      -> quality filtering      (Laplacian variance, adaptive threshold per video)
      -> near-duplicate filter  (downscaled grey: mean-abs-diff + correlation)
      -> selected JPEG frames + video_frames_manifest.json

Nothing here re-implements SfM, matching or depth estimation.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from .io_utils import ensure_dir, safe_json_dump

logger = logging.getLogger(__name__)

# Fallback list; config.yaml `video.extensions` takes precedence.
DEFAULT_VIDEO_EXTENSIONS = [".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg"]

# Small greyscale resolution used for blur scoring / frame signatures.
QUALITY_MAX_DIM_DEFAULT = 960
SIGNATURE_SIZE = (64, 36)          # (width, height) signature of each frame


# ─────────────────────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────────────────────

def _video_config(cfg: Dict) -> Dict:
    return cfg.get("video", {}) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Discovery / probing
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(input_dir: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """Recursively find video files under input_dir."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []

    exclude_dirs = {
        "outputs", "submission", "src", "debug", "depth", "depth_preview",
        "features", "matches", "__pycache__", ".git", "node_modules",
        "venv", ".venv", "env", "video_frames", "extracted_frames",
        "outputs_round2", "outputs_final", "outputs_final2", "outputs_final3",
        "outputs_test", "outputs_test2",
    }
    ext_set = {e.lower() for e in (extensions or DEFAULT_VIDEO_EXTENSIONS)}

    videos = []
    for p in sorted(input_path.rglob("*")):
        skip = False
        for part in p.relative_to(input_path).parts[:-1]:
            if part.lower() in exclude_dirs:
                skip = True
                break
        if skip:
            continue
        if p.is_file() and p.suffix.lower() in ext_set:
            videos.append(p)

    logger.info("[VIDEO] Videos discovered: %d", len(videos))
    return videos


def probe_video(path: Path) -> Dict:
    """Read video metadata without decoding the whole stream."""
    cap = cv2.VideoCapture(str(path))
    info = {
        "video": path.name,
        "path": str(path),
        "fps": 0.0,
        "frame_count": 0,
        "duration_seconds": 0.0,
        "width": 0,
        "height": 0,
        "readable": False,
    }
    if not cap.isOpened():
        logger.warning("[VIDEO] Cannot open %s", path.name)
        return info

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not np.isfinite(fps) or fps <= 0.01:
        fps = 30.0
        logger.warning("[VIDEO] %s: unusable FPS reported; assuming %.1f", path.name, fps)

    info.update({
        "fps": round(fps, 4),
        "frame_count": n_frames,
        "duration_seconds": round(n_frames / fps, 3) if n_frames > 0 else 0.0,
        "width": w,
        "height": h,
        "readable": n_frames > 0 and w > 0,
    })

    logger.info(
        "[VIDEO] %s: %dx%d  %.2f fps  %d frames  %.1fs",
        path.name, w, h, fps, n_frames, info["duration_seconds"],
    )
    return info


def sanitize_id(text: str) -> str:
    """Make a filename-safe identifier from a video file name."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return cleaned or "video"


# ─────────────────────────────────────────────────────────────────────────────
# Small image helpers (blur score + frame signature)
# ─────────────────────────────────────────────────────────────────────────────

def _resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max_dim <= 0 or max(w, h) <= max_dim:
        return img
    scale = max_dim / float(max(w, h))
    return cv2.resize(
        img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _to_gray(img: np.ndarray, max_dim: int) -> np.ndarray:
    small = _resize_max_dim(img, max_dim)
    if small.ndim == 3:
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small


def laplacian_variance(gray: np.ndarray) -> float:
    """Classic focus measure: variance of the Laplacian response."""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_signature(gray: np.ndarray) -> np.ndarray:
    """Tiny greyscale signature used for near-duplicate comparison."""
    sig = cv2.resize(gray, SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)
    return sig.astype(np.float32)


def signature_similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> Tuple[float, float]:
    """Return (mean_abs_diff_normalised, pearson_correlation) between signatures."""
    a = sig_a.ravel().astype(np.float64)
    b = sig_b.ravel().astype(np.float64)
    mad = float(np.mean(np.abs(a - b)) / 255.0)
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    corr = float(np.dot(a_c, b_c) / denom) if denom > 1e-9 else 1.0
    return mad, corr


def is_near_duplicate(mad: float, corr: float, nd_cfg: Dict) -> bool:
    """Conservative near-duplicate decision (both metrics must agree by default)."""
    mad_thr = float(nd_cfg.get("mean_abs_diff_threshold", 0.012))
    corr_thr = float(nd_cfg.get("correlation_threshold", 0.999))
    require_both = bool(nd_cfg.get("require_both_metrics", True))
    mad_hit = mad < mad_thr
    corr_hit = corr > corr_thr
    return (mad_hit and corr_hit) if require_both else (mad_hit or corr_hit)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate extraction (timestamp based, seeking - never decodes every frame)
# ─────────────────────────────────────────────────────────────────────────────

def target_timestamps(probe: Dict, vcfg: Dict) -> List[float]:
    """Timestamps (seconds) of the candidate frames for one video."""
    interval = float(vcfg.get("frame_interval_seconds", 2.0))
    if interval <= 0:
        interval = 2.0
    start = max(0.0, float(vcfg.get("start_offset_seconds", 0.0)))
    end_trim = max(0.0, float(vcfg.get("end_trim_seconds", 0.0)))
    last = max(0.0, float(probe.get("duration_seconds", 0.0)) - end_trim)

    stamps = []
    t = start
    while t <= last + 1e-6:
        stamps.append(round(t, 4))
        t += interval
    return stamps


def extract_candidates(video_path: Path, probe: Dict, cfg: Dict) -> List[Dict]:
    """Seek to every target timestamp, decode ONE frame and score it.

    Only the small blur-score inputs (greyscale statistics + a 64x36
    signature) are retained; full frames are not kept in memory.
    """
    vcfg = _video_config(cfg)
    qcfg = vcfg.get("quality", {}) or {}
    max_dim = int(qcfg.get("max_dim", QUALITY_MAX_DIM_DEFAULT))
    fps = float(probe.get("fps", 0.0)) or 30.0
    n_frames = int(probe.get("frame_count", 0))

    stamps = target_timestamps(probe, vcfg)
    candidates: List[Dict] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("[VIDEO] Cannot open %s for extraction", video_path.name)
        return candidates

    for k, t_target in enumerate(stamps):
        idx = int(round(t_target * fps))
        if n_frames > 0:
            idx = min(idx, n_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.debug("[VIDEO] %s: seek to frame %d failed", video_path.name, idx)
            continue
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if actual < 0:
            actual = idx
        gray = _to_gray(frame, max_dim)
        candidates.append({
            "extraction_index": k,
            "source_frame": int(actual),
            "timestamp_seconds": round(actual / fps, 3),
            "laplacian_variance": float(laplacian_variance(gray)),
            "signature": frame_signature(gray),
        })

    cap.release()
    logger.info(
        "[VIDEO] %s: %d candidate frames decoded (target every %.1fs, %d timestamps)",
        video_path.name, len(candidates),
        float(vcfg.get("frame_interval_seconds", 2.0)), len(stamps),
    )
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Quality filtering (adaptive Laplacian-variance threshold, per video)
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_blur_threshold(variances: List[float], qcfg: Dict) -> Tuple[float, Dict]:
    """Per-video adaptive sharpness threshold.

    The threshold is derived from the video's own distribution
    (median - k * robust_sigma) and is then capped so that at most
    `max_removal_percent` of the candidates and never more than
    `size - min_keep_frames` can be discarded.  No single fixed constant is
    applied to all four videos.
    """
    vals = np.asarray([v for v in variances if np.isfinite(v)], dtype=np.float64)
    stats = {
        "n_candidates": int(vals.size),
        "min": float(vals.min()) if vals.size else 0.0,
        "max": float(vals.max()) if vals.size else 0.0,
        "mean": float(vals.mean()) if vals.size else 0.0,
        "median": 0.0,
        "mad": 0.0,
        "robust_sigma": 0.0,
        "threshold": 0.0,
        "n_below_threshold": 0,
        "max_removal_percent": 0.0,
    }
    if vals.size == 0:
        return 0.0, stats

    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    robust_sigma = 1.4826 * mad
    k = float(qcfg.get("mad_k", 3.0))
    min_abs = float(qcfg.get("min_absolute_variance", 6.0))
    max_removal_pct = float(qcfg.get("max_removal_percent", 35.0))
    min_keep = int(qcfg.get("min_keep_frames", 5))

    threshold = max(min_abs, median - k * robust_sigma)

    # Cap removal: never remove more than max_removal_pct% of candidates,
    # and always keep at least min_keep frames.
    allowed = int(np.floor(vals.size * max_removal_pct / 100.0))
    allowed = max(0, min(allowed, max(vals.size - min_keep, 0)))
    if allowed > 0:
        cap_value = float(np.sort(vals)[allowed])
        threshold = max(threshold, cap_value)
    elif allowed == 0:
        threshold = 0.0  # keep everything: not enough candidates to drop any

    n_below = int((vals < threshold).sum())

    stats.update({
        "median": round(median, 3),
        "mad": round(mad, 3),
        "robust_sigma": round(robust_sigma, 3),
        "threshold": round(float(threshold), 3),
        "n_below_threshold": n_below,
        "max_removal_percent": max_removal_pct,
    })
    return float(threshold), stats


# ─────────────────────────────────────────────────────────────────────────────
# Frame selection: quality filter then conservative near-duplicate filter
# ─────────────────────────────────────────────────────────────────────────────

def select_frames(candidates: List[Dict], cfg: Dict) -> Tuple[List[Dict], List[Dict], Dict]:
    """Apply quality + near-duplicate filtering to one video's candidates.

    Returns (selected, removed, stats).
    """
    vcfg = _video_config(cfg)
    qcfg = vcfg.get("quality", {}) or {}
    nd_cfg = vcfg.get("near_duplicate", {}) or {}
    quality_on = bool(qcfg.get("enabled", True))
    nd_on = bool(nd_cfg.get("enabled", True))

    removed: List[Dict] = []

    # ── Step 1: adaptive quality filter ──────────────────────────────────────
    threshold, qstats = adaptive_blur_threshold(
        [c["laplacian_variance"] for c in candidates], qcfg)

    kept: List[Dict] = []
    for cand in candidates:
        c = dict(cand)
        c["quality_threshold"] = round(threshold, 3)
        if quality_on and c["laplacian_variance"] < threshold:
            c["removal_reason"] = "low_sharpness"
            c.pop("signature", None)
            removed.append(c)
            continue
        kept.append(c)

    # ── Step 2: conservative near-duplicate filter ───────────────────────────
    selected: List[Dict] = []
    dup_mads: List[float] = []
    dup_corrs: List[float] = []

    for c in kept:
        if not selected or not nd_on:
            selected.append(c)
            continue
        mad, corr = signature_similarity(selected[-1]["signature"], c["signature"])
        c["near_duplicate_mean_abs_diff"] = round(mad, 6)
        c["near_duplicate_correlation"] = round(corr, 6)
        dup_mads.append(mad)
        dup_corrs.append(corr)
        if is_near_duplicate(mad, corr, nd_cfg):
            c["removal_reason"] = "near_duplicate"
            c.pop("signature", None)
            removed.append(c)
            continue
        selected.append(c)

    for c in selected:
        c.pop("signature", None)

    n_dup_removed = sum(1 for r in removed if r.get("removal_reason") == "near_duplicate")
    n_blur_removed = sum(1 for r in removed if r.get("removal_reason") == "low_sharpness")
    stats = {
        "quality": qstats,
        "quality_filter_enabled": quality_on,
        "near_duplicate_filter_enabled": nd_on,
        "n_blur_removed": n_blur_removed,
        "n_near_duplicate_removed": n_dup_removed,
        "near_duplicate_mad_mean": round(float(np.mean(dup_mads)), 6) if dup_mads else None,
        "near_duplicate_mad_min": round(float(np.min(dup_mads)), 6) if dup_mads else None,
        "near_duplicate_corr_mean": round(float(np.mean(dup_corrs)), 6) if dup_corrs else None,
        "near_duplicate_mad_threshold": float(nd_cfg.get("mean_abs_diff_threshold", 0.012)),
        "near_duplicate_corr_threshold": float(nd_cfg.get("correlation_threshold", 0.999)),
    }

    # Optional uniform sub-sampling for fast/small runs.
    max_per_video = int(vcfg.get("max_frames_per_video", 0) or 0)
    if max_per_video > 0 and len(selected) > max_per_video:
        idxs = np.linspace(0, len(selected) - 1, max_per_video, dtype=int)
        stats["subsampled_from"] = len(selected)
        stats["subsample_indices"] = [int(i) for i in idxs]
        selected = [selected[i] for i in idxs]

    stats["n_selected"] = len(selected)
    stats["n_candidates"] = len(candidates)
    return selected, removed, stats


# ─────────────────────────────────────────────────────────────────────────────
# Frame writing + manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_selected_frames(
    video_path: Path,
    selected: List[Dict],
    frames_dir: Path,
    cfg: Dict,
    video_id: str,
    video_index: int,
    global_start: int = 0,
) -> List[Dict]:
    """Re-seek the video and write the selected frames as JPEGs with metadata."""
    vcfg = _video_config(cfg)
    jpeg_quality = int(vcfg.get("jpeg_quality", 95))
    prefix = f"{video_id}_V{video_index}"
    entries: List[Dict] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("[VIDEO] Cannot open %s for frame writing", video_path.name)
        return entries

    for sel_index, c in enumerate(selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(c["source_frame"]))
        ok, frame = cap.read()
        if not ok or frame is None:
            logger.warning("[VIDEO] %s: could not re-read frame %d",
                           video_path.name, c["source_frame"])
            continue
        fname = (f"{prefix}_f{int(c['source_frame']):06d}"
                 f"_t{c['timestamp_seconds']:010.3f}.jpg")
        fpath = Path(frames_dir) / fname
        cv2.imwrite(str(fpath), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        h, w = frame.shape[:2]
        entries.append({
            "frame_path": str(fpath),
            "frame_name": fname,
            "source_video": video_path.name,
            "source_video_path": str(video_path),
            "source_video_id": video_id,
            "source_video_index": video_index,
            "source_frame": int(c["source_frame"]),
            "timestamp_seconds": float(c["timestamp_seconds"]),
            "extraction_index": int(c["extraction_index"]),
            "selected_index": sel_index,
            "global_index": global_start + len(entries),
            "laplacian_variance": round(float(c["laplacian_variance"]), 3),
            "quality_threshold": float(c.get("quality_threshold", 0.0)),
            "near_duplicate_mean_abs_diff": c.get("near_duplicate_mean_abs_diff"),
            "near_duplicate_correlation": c.get("near_duplicate_correlation"),
            "width": int(w),
            "height": int(h),
        })

    cap.release()
    logger.info("[VIDEO] %s: wrote %d selected frames to %s",
                video_path.name, len(entries), frames_dir)
    return entries


def _clear_stale_frames(frames_dir: Path) -> None:
    """Remove all JPEG files from the frames directory before fresh extraction."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        return
    removed = 0
    for f in frames_dir.glob("*.jpg"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    if removed:
        logger.info("[VIDEO] Cleared %d stale frames from %s", removed, frames_dir)


def _manifest_cache_key(videos: List[Path], cfg: Dict) -> Dict:
    """Deterministic description of everything that changes frame selection."""
    vcfg = _video_config(cfg)
    return {
        "videos": [{"name": v.name, "size": v.stat().st_size if v.exists() else 0}
                   for v in videos],
        "frame_interval_seconds": float(vcfg.get("frame_interval_seconds", 2.0)),
        "start_offset_seconds": float(vcfg.get("start_offset_seconds", 0.0)),
        "end_trim_seconds": float(vcfg.get("end_trim_seconds", 0.0)),
        "max_frames_per_video": int(vcfg.get("max_frames_per_video", 0) or 0),
        "quality": vcfg.get("quality", {}),
        "near_duplicate": vcfg.get("near_duplicate", {}),
    }


def load_manifest(output_dir: str, cfg: Dict) -> Optional[Dict]:
    """Load an existing frame manifest, or None."""
    path = Path(output_dir) / _video_config(cfg).get("manifest",
                                                     "video_frames_manifest.json")
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[VIDEO] Could not read manifest %s: %s", path, e)
        return None


def prepare_video_frames(
    videos: List[Path],
    cfg: Dict,
    output_dir: str,
    force: bool = False,
) -> Tuple[List[Path], Dict]:
    """Convert every video into selected frames. Returns (frame paths, manifest)."""
    if not videos:
        return [], {}

    vcfg = _video_config(cfg)
    frames_dir = ensure_dir(os.path.join(output_dir,
                                         vcfg.get("frames_dir", "video_frames")))
    manifest_path = os.path.join(output_dir, vcfg.get("manifest",
                                                      "video_frames_manifest.json"))
    cache_key = _manifest_cache_key(videos, cfg)

    # ── Reuse previously extracted frames when the settings are unchanged ────
    if not force:
        prev = load_manifest(output_dir, cfg)
        if prev and prev.get("cache_key") == cache_key:
            paths = [Path(e["frame_path"]) for e in prev.get("frames", [])]
            missing = [p for p in paths if not p.exists()]
            if paths and not missing:
                logger.info(
                    "[VIDEO] Reusing %d previously extracted frames (manifest cache hit)",
                    len(paths))
                return paths, prev
            logger.info("[VIDEO] Manifest cache stale (%d missing frames); re-extracting",
                        len(missing))

    _clear_stale_frames(frames_dir)

    t0 = time.time()
    frames: List[Dict] = []
    removed: List[Dict] = []
    per_video: Dict[str, Dict] = {}
    probes: List[Dict] = []

    for vid_index, video_path in enumerate(videos, start=1):
        probe = probe_video(video_path)
        probes.append(probe)
        if not probe["readable"]:
            logger.warning("[VIDEO] Skipping unreadable video: %s", video_path.name)
            continue

        video_id = sanitize_id(video_path.stem)[:40]
        candidates = extract_candidates(video_path, probe, cfg)
        if not candidates:
            logger.warning("[VIDEO] No candidate frames for %s", video_path.name)
            continue

        selected, removed_v, stats = select_frames(candidates, cfg)
        entries = write_selected_frames(video_path, selected, frames_dir, cfg,
                                        video_id, vid_index,
                                        global_start=len(frames))
        for rem in removed_v:
            rem = dict(rem)
            rem.pop("signature", None)
            rem["source_video"] = video_path.name
            removed.append(rem)

        stats["video"] = video_path.name
        stats["video_id"] = video_id
        stats["video_index"] = vid_index
        stats["n_written"] = len(entries)
        stats["fps"] = probe["fps"]
        stats["frame_count"] = probe["frame_count"]
        stats["duration_seconds"] = probe["duration_seconds"]
        stats["width"] = probe["width"]
        stats["height"] = probe["height"]
        if entries:
            stats["first_timestamp"] = entries[0]["timestamp_seconds"]
            stats["last_timestamp"] = entries[-1]["timestamp_seconds"]
        per_video[video_path.name] = stats
        frames.extend(entries)

        logger.info(
            "[VIDEO] %s -> %d/%d candidates selected "
            "(blur-removed=%d, near-dup-removed=%d, quality_threshold=%.2f)",
            video_path.name, len(entries), stats["n_candidates"],
            stats["n_blur_removed"], stats["n_near_duplicate_removed"],
            stats["quality"]["threshold"],
        )

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_key": cache_key,
        "frames_dir": str(frames_dir),
        "total_source_videos": len(videos),
        "total_source_frames": int(sum(p["frame_count"] for p in probes)),
        "total_candidates": int(sum(v["n_candidates"] for v in per_video.values())),
        "total_selected": len(frames),
        "total_removed": len(removed),
        "config": vcfg,
        "probes": probes,
        "videos": per_video,
        "frames": frames,
        "removed": removed,
    }
    safe_json_dump(manifest, manifest_path)
    logger.info("[VIDEO] %d frames from %d videos selected in %.1fs -> %s",
                len(frames), len(videos), time.time() - t0, manifest_path)

    paths = [Path(e["frame_path"]) for e in frames]
    return paths, manifest
