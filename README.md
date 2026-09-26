# PIXEL-OPS UAV Photogrammetry Pipeline — Round 2

**Hackathon**: PIXEL-OPS UAV-Based Photogrammetry — Round 2  
**Dataset**: AU Alfresco Site — 4 MOV videos, 1280×720 @ 59.94 fps  
**Approach**: Custom photogrammetry pipeline (no COLMAP, no Meshroom)

---

## Overview

A complete end-to-end UAV photogrammetry pipeline that converts **four .MOV video
files** into a **dense 3D colored dotted point cloud** using only open-source
computer-vision libraries (OpenCV, NumPy, SciPy, Open3D).

```
4 MOV videos (12,305 total source frames)
        ↓
Intelligent frame extraction (1 frame / 2 seconds → ~113 frames)
        ↓
Quality filtering (adaptive Laplacian-variance per video)
        ↓
Near-duplicate filtering (thumbnail correlation)
        ↓
Selected frames + video_frames_manifest.json
        ↓
SIFT feature extraction
        ↓
FLANN feature matching
        ↓
Fundamental/Essential matrix RANSAC (geometric verification)
        ↓
Incremental Structure-from-Motion (custom PnP RANSAC)
        ↓
Camera pose estimation + track propagation
        ↓
Multi-view triangulation (DLT)
        ↓
Sparse 3D point cloud
        ↓
Plane-sweep ZNCC depth estimation
        ↓
Multi-view depth fusion
        ↓
Dense colored 3D point cloud (point_cloud.ply)
        ↓
3D dotted point-cloud visualization + screenshot
```

---

## Round-2 Input Videos

| # | File | Resolution | FPS | Frames | Duration |
|---|------|-----------|-----|--------|----------|
| 1 | AU_Alfresco_1_Task 1 - Saturday.MOV | 1280×720 | 59.94 | 3,406 | 56.8s |
| 2 | AU_Alfresco_2_Task 1 - Saturday.MOV | 1280×720 | 59.94 | 3,542 | 59.1s |
| 3 | AU_Alfresco_3_Task 1 - Saturday.MOV | 1280×720 | 59.94 | 2,839 | 47.4s |
| 4 | AU_Alfresco_4_Task 1 - Saturday.MOV | 1280×720 | 59.94 | 2,518 | 42.0s |
| **Total** | | | | **12,305** | **~205s** |

Videos are NOT committed to the repository (too large). Point the pipeline to
their local path via `--video-dir` or `config.yaml`.

---

## Repository Structure

```
pixelops/
├── pipeline_round2.py      ← Round-2 entry point (video input)
├── pipeline.py             ← Round-1 image-only entry point (kept for reference)
├── config.yaml             ← All configurable parameters
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── src/
│   ├── video_input.py      ← MOV → selected frames (Round-2 addition)
│   ├── io_utils.py         ← Image discovery, EXIF, loading
│   ├── camera.py           ← Pinhole camera model
│   ├── features.py         ← SIFT / ORB feature extraction
│   ├── matching.py         ← FLANN matching + Lowe ratio test
│   ├── geometry.py         ← Fundamental/Essential matrix RANSAC
│   ├── sfm.py              ← Incremental SfM (PnP RANSAC, track propagation)
│   ├── triangulation.py    ← DLT triangulation, parallax filtering
│   ├── depth.py            ← Plane-sweep ZNCC depth estimation
│   ├── fusion.py           ← Multi-view depth fusion, PLY export
│   ├── visualization.py    ← Open3D point cloud visualization
│   ├── evaluation.py       ← Quality report generation
│   ├── bundle_adjustment.py← SciPy-based bundle adjustment (optional)
│   └── __init__.py
│
└── outputs_round2/         ← Generated (not committed)
    ├── video_frames/           ← Extracted JPEG frames
    ├── video_frames_manifest.json
    ├── connectivity/
    │   ├── connectivity_report.json
    │   └── connectivity_report.txt
    ├── sparse/
    │   └── sparse_points.ply
    ├── dense/
    │   └── point_cloud.ply     ← Final dense colored point cloud
    ├── depth/                  ← Per-image depth maps (.npy + .png)
    ├── visualization/
    │   └── reconstruction_screenshot.png
    └── reports/
        ├── reconstruction_report.json
        ├── reconstruction_report.txt
        └── dataset_report.json
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline (Phases 1–6)

```bash
python pipeline_round2.py \
    --video-dir "C:\path\to\2nd round" \
    --output outputs_round2 \
    --dense
```

### Phased Testing (recommended order)

```bash
# Phase 1: Syntax + import check
python pipeline_round2.py --video-dir "..." --output outputs_round2 --test-phase 1

# Phase 2: Verify all 4 MOV files + decode sample frames
python pipeline_round2.py --video-dir "..." --output outputs_round2 --test-phase 2

# Phase 3: Frame extraction only (see counts per video)
python pipeline_round2.py --video-dir "..." --output outputs_round2 --test-phase 3

# Phase 4: Cross-video connectivity test (checks if videos can form one reconstruction)
python pipeline_round2.py --video-dir "..." --output outputs_round2 --test-phase 4

# Phase 5: Small reconstruction (30 frames only, fast)
python pipeline_round2.py --video-dir "..." --output outputs_round2 --test-phase 5

# Phase 6: Full Round-2 reconstruction
python pipeline_round2.py --video-dir "..." --output outputs_round2
```

### Quick test with faster settings

```bash
python pipeline_round2.py \
    --video-dir "..." \
    --output outputs_round2_test \
    --frame-interval 5.0 \
    --max-frames-per-video 15 \
    --no-dense
```

---

## Video Frame Extraction

Video frames are extracted using timestamp-based seeking (not decoding all frames).

**Default settings** (`frame_interval_seconds = 2.0`):
- Video 1 (56.8s) → ~28 candidate frames
- Video 2 (59.1s) → ~29 candidate frames
- Video 3 (47.4s) → ~23 candidate frames
- Video 4 (42.0s) → ~21 candidate frames
- **Total candidates: ~101 frames → typically 80–100 after filtering**

Each extracted frame includes metadata:
```json
{
  "frame_path": "outputs_round2/video_frames/..._f001200_t0020.000.jpg",
  "source_video": "AU_Alfresco_1_Task 1 - Saturday.MOV",
  "source_frame": 1200,
  "timestamp_seconds": 20.019,
  "extraction_index": 10,
  "laplacian_variance": 145.3
}
```

---

## Quality Filtering

Blurry frames are removed using **adaptive Laplacian-variance** (per video):

- Threshold = max(min_abs_variance, median − k × robust_sigma)
- Capped so no more than 35% of candidates are removed per video
- Each video gets its own threshold (no single global constant)
- Default: `mad_k=3.0`, `min_absolute_variance=6.0`

Near-duplicate filtering uses 64×36 thumbnail comparison:
- Mean absolute pixel difference < 0.012 **AND** Pearson correlation > 0.999
- Conservative: both metrics must agree before a frame is discarded

---

## Structure-from-Motion

The SfM module (`src/sfm.py`) uses incremental registration:

1. **Initial pair**: Selected by max verified inlier count × match ratio
2. **Two-view initialization**: Identity + relative pose from Essential matrix; DLT triangulation
3. **Incremental registration**: PnP RANSAC (`cv2.solvePnPRansac`) for each new camera
4. **Track propagation**: After each registration, existing 3D landmarks are propagated to the new camera (not re-triangulated), preventing duplicate landmarks
5. **Triangulation**: Only genuinely new point pairs are triangulated
6. **Connectivity-ordered**: Cameras are registered in order of their 2D-3D correspondence count, maximizing early success

Key bookkeeping invariants enforced:
- One feature index → at most one landmark per image
- One landmark → at most one observation per image  
- PnP inliers are committed to feat2lm and landmark.observations immediately

---

## Dense Reconstruction

Depth estimation uses **plane-sweep ZNCC** (Zero-mean Normalized Cross-Correlation):
- Reference camera sweeps depth hypotheses between adaptive d_min/d_max
- ZNCC computed using OpenCV `boxFilter` (O(1) per pixel per hypothesis)
- Best depth = depth at maximum ZNCC score above threshold (0.35)

Dense point cloud:
- Each depth pixel back-projected to 3D using scaled pinhole model
- RGB color sampled from the reference image at the 2D pixel location
- Multi-view fusion with voxel downsampling + statistical outlier removal
- Output: `outputs_round2/dense/point_cloud.ply` (XYZRGB format)

---

## 3D Dotted Point Cloud Visualization

The final reconstruction is visualized as **individual colored 3D dots** using Open3D:

```bash
# Interactive viewer (Open3D)
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('outputs_round2/dense/point_cloud.ply')
vis = o3d.visualization.Visualizer()
vis.create_window('PIXEL-OPS Point Cloud')
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.point_size = 2.0
opt.background_color = [0.1, 0.1, 0.1]
vis.run()
vis.destroy_window()
"
```

Controls:
- **Left drag**: Rotate
- **Middle drag / scroll**: Zoom
- **Right drag**: Pan
- **R**: Reset view

A screenshot is automatically saved to `outputs_round2/visualization/reconstruction_screenshot.png`.

---

## Configuration Options

All parameters are in `config.yaml`. Key options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video.frame_interval_seconds` | `2.0` | Seconds between extracted frames |
| `video.max_frames_per_video` | `0` | Max frames per video (0=all) |
| `video.quality.mad_k` | `3.0` | Blur filter aggressiveness |
| `video.quality.max_removal_percent` | `35.0` | Cap on frames removed per video |
| `video.near_duplicate.mean_abs_diff_threshold` | `0.012` | Near-dup sensitivity |
| `image.resize_for_processing` | `1280` | Max dim for SIFT/matching |
| `features.sift_nfeatures` | `6000` | Max SIFT keypoints per image |
| `matching.sequential_window` | `25` | Matching window size |
| `depth.max_depth_images` | `30` | Max depth reference images |
| `depth.num_neighbors` | `4` | Neighboring views per reference |
| `fusion.voxel_size` | `0.05` | Point cloud voxel downsampling |
| `visualization.point_size` | `2.0` | Rendered dot size |

---

## Expected Output

After a successful full run:

```
outputs_round2/
├── video_frames/               ~80-100 extracted JPEG frames
├── video_frames_manifest.json  Per-frame provenance metadata
├── connectivity/
│   ├── connectivity_report.json
│   └── connectivity_report.txt  "Video X <-> Video Y: N verified matches"
├── sparse/
│   └── sparse_points.ply       SfM sparse cloud
├── dense/
│   └── point_cloud.ply         Dense colored point cloud (XYZRGB)
├── depth/                      Per-image depth maps
├── visualization/
│   └── reconstruction_screenshot.png
└── reports/
    └── reconstruction_report.json
```

---

## Technical Notes

- **No COLMAP, no Meshroom** — fully custom OpenCV + NumPy implementation
- **Relative scale** — without GPS / GCP ground control, scale is not metric
- **Camera intrinsics** — estimated as `1.2 × max(1280, 720) = 1536 px` focal length
  (video frames carry no EXIF focal length data)
- **GPU** — SIFT/matching/SfM all run on CPU; Open3D voxelization uses CPU
- **Bundle adjustment** — disabled by default; enable with `--use-bundle-adjustment`

---

## Requirements

See `requirements.txt`. Key dependencies:
- `opencv-python >= 4.8`
- `open3d >= 0.17`
- `numpy`, `scipy`, `Pillow`, `pyyaml`, `matplotlib`
