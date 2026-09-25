# PIXEL-OPS UAV Photogrammetry Pipeline

**Hackathon**: PIXEL-OPS UAV-Based Photogrammetry — Trial Round  
**Team**: PIXEL-OPS  
**Target Dataset**: Lighthouse Survey (DB4)

---

## Overview

A complete end-to-end photogrammetry pipeline built from scratch using open-source
computer-vision libraries (OpenCV, NumPy, SciPy, Open3D). No pre-built photogrammetry
software (COLMAP, Meshroom, Metashape) is required or used.

Given a folder of overlapping UAV images, the pipeline produces:
- A **sparse 3D point cloud** (SfM reconstruction)
- **Depth maps** per reference image (semi-dense patch matching)
- A **dense 3D point cloud** (fused from depth maps)
- Quality reports, visualizations, and a PDF process document

---

## Problem Statement

Reconstruct a 3D model of a lighthouse from overlapping aerial photographs.
The pipeline must be self-contained, reproducible, and adaptable to any new
image dataset without code changes.

---

## Pipeline Architecture

```
INPUT IMAGES
    │
    ▼
Image Discovery & Validation (src/io_utils.py)
    │
    ▼
EXIF Parsing + Camera Intrinsics (src/camera.py)
    │
    ▼
SIFT Feature Extraction (src/features.py)
    │
    ▼
FLANN + Lowe Ratio Matching (src/matching.py)
    │
    ▼
Fundamental/Essential Matrix RANSAC (src/geometry.py)
    │
    ▼
Initial Pair Selection (src/triangulation.py)
    │
    ▼
Two-View Initialization + DLT Triangulation
    │
    ▼
Incremental PnP SfM (src/sfm.py)
    │
    ▼
[Optional] Bundle Adjustment (src/bundle_adjustment.py)
    │
    ▼
Sparse PLY Export
    │
    ▼
Semi-Dense NCC Depth Estimation (src/depth.py)
    │
    ▼
Multi-View Depth Fusion + Outlier Removal (src/fusion.py)
    │
    ▼
Dense PLY Export
    │
    ▼
Visualization + Screenshot (src/visualization.py)
    │
    ▼
Quality Report + Documentation (src/evaluation.py)
```

---

## Repository Structure

```
pixelops/
├── pipeline.py          ← Main entry point
├── config.yaml          ← All parameters (no magic numbers in code)
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── io_utils.py       ← Image discovery, EXIF, loading
│   ├── camera.py         ← Pinhole camera model
│   ├── features.py       ← SIFT/ORB feature extraction
│   ├── matching.py       ← FLANN/BF matching + Lowe ratio
│   ├── geometry.py       ← F/E matrix, pose recovery, reprojection
│   ├── sfm.py            ← Incremental SfM, landmarks, PnP
│   ├── triangulation.py  ← DLT triangulation, pair selection
│   ├── bundle_adjustment.py ← SciPy least_squares BA
│   ├── depth.py          ← Semi-dense NCC depth estimation
│   ├── fusion.py         ← Depth back-projection + PLY export
│   ├── visualization.py  ← Open3D screenshot + matplotlib fallback
│   └── evaluation.py     ← Quality reports, GPU detection
│
├── outputs/              ← All pipeline outputs
│   ├── sparse_points.ply
│   ├── points.ply
│   ├── reconstruction_screenshot.png
│   ├── reconstruction_report.json
│   ├── reconstruction_report.txt
│   ├── documentation.pdf
│   ├── dataset_report.json
│   ├── feature_report.json
│   ├── matching_report.json
│   ├── depth/            ← depth_map_*.png + depth_map_*.npy
│   ├── depth_preview/
│   └── debug/
│
└── submission/           ← Ready-to-submit package
    ├── pipeline.py
    ├── src/
    ├── points.ply
    ├── sparse_points.ply
    ├── depth_maps/
    ├── reconstruction_screenshot.png
    ├── reconstruction_report.json
    ├── documentation.pdf
    ├── submission_manifest.json
    └── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Required packages:**
- `opencv-python-headless` ≥ 4.8
- `numpy` ≥ 1.24
- `scipy` ≥ 1.10
- `open3d` ≥ 0.17
- `Pillow` ≥ 9.0
- `matplotlib` ≥ 3.6
- `PyYAML` ≥ 6.0
- `fpdf2` ≥ 2.7
- `exifread` ≥ 3.0
- `imageio` ≥ 2.20

---

## Usage

### Basic reconstruction (recommended for trial round):

```bash
python pipeline.py --input data/lighthouse --output submission
```

### Full quality run with visualisation:

```bash
python pipeline.py \
    --input data/lighthouse \
    --output submission \
    --dense \
    --visualize
```

### With bundle adjustment (slower, better accuracy):

```bash
python pipeline.py \
    --input data/lighthouse \
    --output submission \
    --dense \
    --use-bundle-adjustment
```

### All options:

```
  --input PATH         Input directory with images (required)
  --output PATH        Output directory (default: submission)
  --config PATH        Path to config.yaml
  --max-images N       Limit to N images (0 = all)
  --feature-method     SIFT or ORB (default: SIFT)
  --match-method       FLANN or BF (default: FLANN)
  --use-bundle-adjustment  Enable bundle adjustment
  --dense              Enable depth estimation (default: on)
  --no-dense           Disable depth estimation
  --voxel-size FLOAT   Voxel size for point cloud downsampling
  --visualize          Generate extra visualizations
  --debug              Save debug images, verbose logging
```

---

## Input Format

- A directory (flat or recursive) containing JPG/JPEG/PNG images
- Images should have significant overlap (~60-80% recommended for UAV)
- EXIF focal length is auto-detected if available; otherwise estimated from image size
- No ground control points or GPS required

---

## Output Format

| File | Description |
|------|-------------|
| `points.ply` | Primary dense point cloud (XYZ + RGB) |
| `sparse_points.ply` | Sparse SfM point cloud |
| `depth/depth_map_*.png` | 16-bit normalized depth images |
| `depth/depth_map_*.npy` | Raw float32 depth arrays |
| `reconstruction_screenshot.png` | Automated point cloud render |
| `reconstruction_report.json` | Full metrics as JSON |
| `reconstruction_report.txt` | Human-readable quality report |
| `documentation.pdf` | 1-page process document |
| `submission_manifest.json` | Submission metadata |
| `dataset_report.json` | Input image analysis |

---

## Algorithm Details

### Feature Extraction
SIFT (Scale-Invariant Feature Transform) detects up to 4000 keypoints per image
that are invariant to scale and rotation changes. Contrast threshold and edge
threshold are configurable in `config.yaml`.

### Feature Matching
FLANN (Fast Library for Approximate Nearest Neighbours) matches descriptors
between image pairs. Lowe's ratio test (threshold 0.75) rejects ambiguous
matches. Only sequential-window pairs (±10 images) are matched by default.

### Geometric Verification
OpenCV `findFundamentalMat` (RANSAC, 1.5px threshold) followed by
`findEssentialMat` and `recoverPose` recovers the relative camera pose
for each verified pair.

### Initial Pair Selection
The seed pair is chosen by maximizing a composite score (inlier count × inlier
ratio), with mild preference for pairs separated by more than a few frames.

### Incremental SfM
Starting from the seed pair, cameras are added one-by-one via PnP RANSAC
against the growing 3D landmark set. New 3D points are triangulated for
each newly registered camera.

### Depth Estimation
For each reference image, a depth hypothesis sweep (64 levels, 0.5–50 scene
units) is performed. NCC patch similarity (11×11 window) across 2 neighbour
views selects the winning depth. Invalid pixels are filled by propagation.

### Point Cloud Fusion
Valid depth pixels are back-projected using the pinhole model and transformed
to world coordinates using the camera pose. Multi-view point sets are merged,
voxel-downsampled (0.05 units), and cleaned with statistical + radius outlier
removal using Open3D.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| CPU | Any modern | Multi-core x86-64 |
| GPU | Not required | NVIDIA (speedup for Open3D) |
| Disk | 2 GB free | 10 GB for large datasets |

---

## Configuration

All parameters are in `config.yaml`. Key settings:

```yaml
image:
  max_images: 60          # Limit images for speed (0 = all)
  resize_for_processing: 1600  # Max processing dimension

features:
  sift_nfeatures: 4000    # Features per image
  method: SIFT

matching:
  ratio_threshold: 0.75   # Lowe ratio
  sequential_window: 10   # Match ± N neighbours

bundle_adjustment:
  enabled: false          # Set true for quality mode
```

---

## Validation on New Dataset

The pipeline is fully dataset-agnostic. To run on a new dataset:

```bash
python pipeline.py --input /path/to/new/images --output /path/to/output --dense
```

No code changes are needed. The pipeline automatically:
- Discovers all JPG/PNG images recursively
- Estimates camera intrinsics from EXIF or image dimensions
- Selects the best image pairs for matching
- Chooses the best initial pair for SfM
- Registers all connectable cameras
- Reports all metrics for the new dataset

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `No module named 'cv2'` | `pip install opencv-python-headless` |
| `No module named 'open3d'` | `pip install open3d` |
| `Too few features` | Increase `sift_nfeatures` in config.yaml |
| `No pairs verified` | Reduce `min_inliers` in config.yaml |
| `0 cameras registered` | Check image overlap; try `strategy: exhaustive` |
| OOM / out of memory | Reduce `max_images` or `resize_for_processing` |
| Depth maps all zeros | Try reducing `ncc_threshold` to 0.1 |

---

## Limitations

- **Scale**: Reconstruction is in relative (not metric) scale unless GPS/GCP is provided
- **Speed**: Full dense reconstruction on 60 images takes ~20–40 min on CPU
- **Bundle adjustment**: Disabled by default; enable with `--use-bundle-adjustment`
- **Textureless regions**: Sky, plain concrete walls may lack features
- **Duplicate keypoints**: NCC depth can produce false matches in repetitive textures

---

## License

MIT License — free to use, modify, and distribute.
