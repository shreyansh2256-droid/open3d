"""
io_utils.py — Image discovery, loading, EXIF reading, directory management.
"""

import os
import sys
import json
import logging
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image as PILImage
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Directory utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json_dump(data, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ─────────────────────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_images(input_dir: str, extensions: List[str], max_images: int = 0) -> List[Path]:
    """Recursively find all supported image files under input_dir.

    Excludes common non-dataset subdirectories like outputs/, submission/, src/, etc.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Directories to skip when scanning recursively
    EXCLUDE_DIRS = {
        "outputs", "submission", "src", "debug", "depth", "depth_preview",
        "features", "matches", "__pycache__", ".git", "node_modules",
        "venv", ".venv", "env",
    }

    ext_set = {e.lower() for e in extensions}
    images = []
    for p in sorted(input_path.rglob("*")):
        # Skip if any path component is in the excluded set
        skip = False
        for part in p.relative_to(input_path).parts[:-1]:  # parent parts only
            if part.lower() in EXCLUDE_DIRS:
                skip = True
                break
        if skip:
            continue
        if p.is_file() and p.suffix.lower() in ext_set:
            images.append(p)

    images.sort(key=lambda x: x.name)

    if max_images and max_images > 0 and len(images) > max_images:
        logger.info(f"Limiting to {max_images} images (found {len(images)})")
        # Sample uniformly across the sorted list for better coverage
        indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
        images = [images[i] for i in indices]

    logger.info(f"[INFO] Images discovered: {len(images)}")
    return images


# ─────────────────────────────────────────────────────────────────────────────
# EXIF reading
# ─────────────────────────────────────────────────────────────────────────────

def read_exif(image_path: Path) -> Dict:
    """Return a normalised EXIF dict from the image. Tries exifread first, PIL fallback."""
    result = {}
    # --- Try exifread (handles DNG / RAW TIFF tags that PIL misses) -----------
    try:
        import exifread
        with open(str(image_path), "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="EOF")
        for key, val in tags.items():
            try:
                v = val.values
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                if hasattr(v, 'num') and hasattr(v, 'den'):
                    v = float(v.num) / float(v.den) if v.den != 0 else 0.0
                result[key] = v
            except Exception:
                result[key] = str(val)
    except Exception:
        pass

    # --- Also pull PIL EXIF for normalised keys ------------------------------
    try:
        img = PILImage.open(str(image_path))
        raw_exif = img._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                result[tag] = value
    except Exception:
        pass

    # --- Normalise the keys we care about ------------------------------------
    # FocalLength (mm)
    for k in ("EXIF FocalLength", "FocalLength"):
        v = result.get(k)
        if v is not None:
            try:
                result["focal_length_mm"] = float(v)
                break
            except Exception:
                pass

    # FocalPlaneXResolution (pixels/cm or pixels/inch)
    for k in ("EXIF FocalPlaneXResolution", "FocalPlaneXResolution"):
        v = result.get(k)
        if v is not None:
            try:
                result["focal_plane_x_res"] = float(v)
                break
            except Exception:
                pass

    # FocalPlaneResolutionUnit: 2=inch, 3=cm
    for k in ("EXIF FocalPlaneResolutionUnit", "FocalPlaneResolutionUnit"):
        v = result.get(k)
        if v is not None:
            try:
                result["focal_plane_res_unit"] = int(v)
                break
            except Exception:
                pass

    # FocalLengthIn35mmFilm
    for k in ("EXIF FocalLengthIn35mmFilm", "FocalLengthIn35mmFilm"):
        v = result.get(k)
        if v is not None:
            try:
                result["focal_length_35mm"] = float(v)
                break
            except Exception:
                pass

    # Make / Model
    for k in ("Image Make", "Make"):
        v = result.get(k)
        if v and str(v).strip():
            result["make"] = str(v).strip()
            break
    for k in ("Image Model", "Model"):
        v = result.get(k)
        if v and str(v).strip():
            result["model"] = str(v).strip()
            break

    return result


def exif_focal_length_pixels(exif: Dict, width: int, height: int) -> Optional[float]:
    """
    Compute focal length in pixels using best available EXIF data.

    Priority:
    1. FocalLength + FocalPlaneXResolution -> direct pixel pitch calculation
    2. FocalLengthIn35mmFilm -> estimate via sensor diagonal
    3. FocalLength alone -> fallback using conservative sensor estimate
    """
    try:
        fl_mm = exif.get("focal_length_mm")
        fp_res = exif.get("focal_plane_x_res")   # pixels per unit
        fp_unit = exif.get("focal_plane_res_unit", 2)  # 2=inch, 3=cm

        # Method 1: FocalLength + FocalPlaneXResolution (most accurate)
        if fl_mm and fp_res and fp_res > 0:
            # Convert resolution to pixels/mm
            if fp_unit == 2:    # per inch -> per mm
                fp_res_per_mm = fp_res / 25.4
            elif fp_unit == 3:  # per cm -> per mm
                fp_res_per_mm = fp_res / 10.0
            else:               # assume per mm
                fp_res_per_mm = fp_res
            fl_px = fl_mm * fp_res_per_mm
            logger.debug("Focal length from FocalPlane: %.1f px (%.2f mm, res=%.2f/mm)",
                         fl_px, fl_mm, fp_res_per_mm)
            return float(fl_px)

        # Method 2: FocalLengthIn35mmFilm
        fl35 = exif.get("focal_length_35mm")
        if fl35 and fl35 > 0:
            diag_px = (width**2 + height**2) ** 0.5
            diag_35 = 43.267   # 35mm film frame diagonal (mm)
            fl_px = (fl35 / diag_35) * diag_px
            logger.debug("Focal length from 35mm equiv: %.1f px (fl35=%.1fmm)", fl_px, fl35)
            return float(fl_px)

        # Method 3: FocalLength alone with generic sensor estimate
        if fl_mm and fl_mm > 0:
            # senseFly albris: FocalPlaneXResolution=7142.857 px/cm => 714.29 px/mm
            # For unknown cameras, use conservative estimate.
            # Common 1" sensor: 13.2mm wide, ~6000px wide => 454 px/mm
            # 1/2.3" sensor: 6.17mm wide, ~4000px wide => 648 px/mm
            # Use image width / (fl35 / diag_35 * sensor_diag) if 35mm known
            # Fallback: assume sensor width ~6mm (common small drone sensor)
            sensor_w_mm = 6.17
            fl_px = fl_mm * (width / sensor_w_mm)
            logger.debug("Focal length from fallback sensor: %.1f px (fl=%.2f mm)", fl_px, fl_mm)
            return float(fl_px)

    except Exception as e:
        logger.debug("Focal length extraction error: %s", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image_rgb(path: Path, max_dim: int = 0) -> Optional[np.ndarray]:
    """Load image as RGB uint8 numpy array, optionally downscaling."""
    try:
        img = PILImage.open(str(path)).convert("RGB")
        if max_dim > 0:
            w, h = img.size
            scale = min(max_dim / max(w, h), 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
        return np.array(img)
    except Exception as e:
        logger.warning(f"Failed to load image {path.name}: {e}")
        return None


def load_image_gray(path: Path, max_dim: int = 0) -> Optional[np.ndarray]:
    """Load image as grayscale uint8 numpy array."""
    rgb = load_image_rgb(path, max_dim)
    if rgb is None:
        return None
    # Luminance conversion
    gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
    return gray


def image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height) without fully decoding the image."""
    try:
        img = PILImage.open(str(path))
        return img.size  # (width, height)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset report
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset_report(
    images: List[Path],
    cfg: Dict,
    output_dir: str,
    gpu_info: Dict,
) -> Dict:
    """Inspect images and write dataset_report.json."""
    report = {
        "image_count": len(images),
        "images": [],
        "gpu": gpu_info,
        "pipeline_config": cfg,
    }

    resolutions = []
    exif_count = 0
    focal_lengths = []

    for img_path in images:
        dims = image_dimensions(img_path)
        if dims is None:
            logger.warning(f"Could not read dimensions: {img_path.name}")
            continue

        w, h = dims
        exif = read_exif(img_path)
        has_exif = len(exif) > 0
        if has_exif:
            exif_count += 1

        fl_px = exif_focal_length_pixels(exif, w, h)
        if fl_px:
            focal_lengths.append(fl_px)

        resolutions.append((w, h))
        report["images"].append({
            "name": img_path.name,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 4),
            "has_exif": has_exif,
            "focal_length_px": round(fl_px, 1) if fl_px else None,
        })

    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        report["summary"] = {
            "unique_resolutions": list(set(resolutions)),
            "min_width": min(widths),
            "max_width": max(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "exif_available": exif_count,
            "exif_percent": round(100 * exif_count / len(images), 1),
            "mean_focal_length_px": round(float(np.mean(focal_lengths)), 1) if focal_lengths else None,
            "estimated_camera_model": "Pinhole",
        }

    safe_json_dump(report, os.path.join(output_dir, cfg.get("output", {}).get("dataset_report", "dataset_report.json")))
    return report
