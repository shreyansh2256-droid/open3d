"""
Geometric diagnostics:
1. Sparse cloud extents and camera centers
2. Depth map back-projection vs sparse cloud
3. Plane-sweep normal vs UAV geometry
4. Depth scale check (d_max=50 in SfM units)
"""
import sys, numpy as np
sys.path.insert(0, '.')

import open3d as o3d
from src.camera import Camera
from src.sfm import SfMState
from pathlib import Path
import yaml, pickle, os

# ── Load config ───────────────────────────────────────────────────────────────
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# ── Load sparse cloud ─────────────────────────────────────────────────────────
sparse_pcd = o3d.io.read_point_cloud('outputs/sparse_points.ply')
sparse_pts = np.asarray(sparse_pcd.points)
print("=== SPARSE CLOUD ===")
print(f"  Points:   {len(sparse_pts):,}")
print(f"  BBox min: {sparse_pts.min(0).round(3)}")
print(f"  BBox max: {sparse_pts.max(0).round(3)}")
print(f"  Extents:  {(sparse_pts.max(0)-sparse_pts.min(0)).round(3)}")
print(f"  Centroid: {sparse_pts.mean(0).round(3)}")
print(f"  Depth Z range:  [{sparse_pts[:,2].min():.3f}, {sparse_pts[:,2].max():.3f}]")
print(f"  X range:        [{sparse_pts[:,0].min():.3f}, {sparse_pts[:,0].max():.3f}]")
print(f"  Y range:        [{sparse_pts[:,1].min():.3f}, {sparse_pts[:,1].max():.3f}]")

# ── Load one depth map and check depth scale relative to sparse ───────────────
dm_path = sorted(Path('outputs/depth').glob('*.npy'))[0]
dm = np.load(str(dm_path))
valid = dm > 0
print(f"\n=== DEPTH MAP: {dm_path.name} ===")
print(f"  Shape: {dm.shape}")
print(f"  Valid: {valid.sum():,} / {dm.size:,}  ({100*valid.mean():.1f}%)")
if valid.any():
    dv = dm[valid]
    print(f"  Range: [{dv.min():.3f}, {dv.max():.3f}]")
    print(f"  Median: {np.median(dv):.3f}  Std: {dv.std():.3f}")
    # Histogram
    bins = np.linspace(dv.min(), dv.max(), 6)
    h,_ = np.histogram(dv, bins=bins)
    for i in range(len(h)):
        pct = 100*h[i]/len(dv)
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: {h[i]:,}  ({pct:.1f}%)")

# ── Rebuild cameras from the outputs ─────────────────────────────────────────
# The pipeline stored cameras in state; we must reconstruct from what's available.
# Check reconstruction_report for camera info
import json
report = json.load(open('outputs/reconstruction_report.json'))
print(f"\n=== RECONSTRUCTION REPORT ===")
print(f"  Registered cameras: {report['reconstruction']['n_registered_cameras']}")
print(f"  Sparse points: {report['reconstruction']['n_sparse_points']}")

# ── Analyse plane-sweep geometry ──────────────────────────────────────────────
print(f"\n=== PLANE-SWEEP GEOMETRY ANALYSIS ===")
print("  Plane normal in depth.py: n = [0, 0, 1]  (hardcoded, camera Z-axis)")
print("  This sweeps planes PARALLEL to image plane (frontoparallel)")
print()
print("  Camera convention in sfm.py:")
print("    cam1 = identity: R=I, t=0   (world origin)")
print("    cam2 = R_rel, t_rel from recoverPose()")
print("    set_pose(R,t): stores R (world→cam), t (world→cam)")
print()
print("  Back-projection in fusion.py depth_to_pointcloud():")
print("    X_cam = depth * K_inv @ [u, v, 1].T       # line 53")
print("    X_world = (R.T @ (X_cam - t)).T            # line 56")
print()
print("  QUESTION: What does 'depth' mean in the depth map?")
print("  In compute_homography_plane(), depth is the Z-coordinate")
print("  in CAMERA 1's frame (frontoparallel plane distance).")
print("  This IS the Z component of X_cam, so back-projection is correct:")
print("    X_cam = d * [u-cx, v-cy, 1] / fx  (approximately)")
print("    Correct: X_cam[2] = d  (exactly, since d * K_inv * [u,v,1] has z=d)")

# Verify: K_inv @ [cx, cy, 1] should give [0,0,1] (principal ray)
# Let's use sample camera intrinsics from the report
print(f"\n=== VERIFY DEPTH SCALE vs SPARSE CLOUD ===")
print("  Sparse cloud Z range:", sparse_pts[:,2].min().round(3), "to", sparse_pts[:,2].max().round(3))
print("  Sparse cloud |extent|:", np.linalg.norm(sparse_pts.max(0)-sparse_pts.min(0)).round(3))
print()
print("  Camera 1 is at world origin (identity pose)")
print("  Camera centers should be within sparse cloud bbox")
print("  d_min=0.5, d_max=50.0 in config")
print()

# Estimate expected depth range:
# If sparse cloud has extent ~10 units (from trajectory plot X in [-6,6], Z in [-0.5,10])
# and cameras are orbiting radius ~6-7 units from center
# then depth to scene should be ~5-15 units
print("  From camera trajectory: orbital radius ~6 units, scene Z-depth ~0-10")
print("  Expected depth to scene: ~3-15 units")
print("  d_max=50.0 is way TOO LARGE relative to actual scene scale")
print("  Many pixels will get d=50 (or near it) which projects to far-away garbage points")

# ── Analyze back-projection of a sample ──────────────────────────────────────
print(f"\n=== BACK-PROJECTION SAMPLE CHECK ===")
print("  Checking if K_inv formula gives correct camera coords:")
# Use nominal intrinsics
fx = fy = 1281.6
cx, cy = 800.0, 600.0
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
Kinv = np.linalg.inv(K)

# At depth d, principal point (cx, cy) should map to (0, 0, d) in camera frame
for d in [5.0, 20.0, 50.0]:
    ray = Kinv @ np.array([cx, cy, 1.0])
    X_cam = d * ray
    print(f"  d={d}: X_cam = {X_cam.round(4)}  (expect [0, 0, {d}])")

# Corner pixel at depth d
for d in [10.0]:
    for u,v in [(0,0),(1600,1200),(800,600)]:
        ray = Kinv @ np.array([float(u), float(v), 1.0])
        X_cam = d * ray
        print(f"  d={d} pix=({u},{v}): X_cam = {X_cam.round(3)}")

# ── At depth=600px image, intrinsics are scaled ────────────────────────────────
print(f"\n=== DEPTH MAP RESOLUTION INTRINSIC SCALING ===")
# Image loaded at max_dim=600: 7152x5368 → scale=600/7152=0.0839... wait
# Actually process size is 1600px, then depth uses min(1600,600)=600
# At 600px: scale = 600/max(1600,1200) = 600/1600 = 0.375
# So depth image is 1600*0.375=600 x 1200*0.375=450
depth_scale = 600.0 / 1600.0
fx_dep = fx * depth_scale
cx_dep = cx * depth_scale
cy_dep = 600.0 * depth_scale
print(f"  Process size: 1600x1200  →  depth size: 600x450")
print(f"  Depth scale: {depth_scale:.4f}")
print(f"  fx_dep = {fx} * {depth_scale:.4f} = {fx_dep:.2f}")
print(f"  cx_dep = {cx} * {depth_scale:.4f} = {cx_dep:.2f}")
print(f"  cy_dep = {cy} * {depth_scale:.4f} = {cy_dep:.2f}")
print(f"  (In fusion.py dep_cam is built with scale_x=dm_w/cam.width)")
print(f"  dm_w={int(1600*depth_scale)}, cam.width=1600 → scale_x={depth_scale:.4f}  ✓")

# ── The REAL problem: plane normal is [0,0,1] in CAMERA-1 frame ───────────────
print(f"\n=== ROOT CAUSE ANALYSIS ===")
print("""
PROBLEM 1 — HOMOGRAPHY PLANE NORMAL:
  depth.py compute_homography_plane() uses normal=[0,0,1] (camera Z axis).
  This sweeps planes PARALLEL to the image plane of camera 1.
  For a UAV orbiting a lighthouse, camera 1 is looking roughly DOWNWARD.
  Its Z axis points DOWN toward the ground.
  The plane-sweep therefore sweeps horizontal planes at heights 0.5 to 50.0.
  This is actually CORRECT for overhead cameras.

PROBLEM 2 — DEPTH RANGE vs SCENE SCALE:
  d_min=0.5, d_max=50.0 are in SfM world units.
  SfM world scale: cameras orbit at radius ~6-7 units (from trajectory plot).
  Scene depth (ground below cameras) is ~0-10 units.
  d_max=50.0 is 5-8x the actual scene depth.
  Pixels at the image periphery where the homography warp falls outside
  the neighbour image get ZNCC≈0, which may beat the threshold.
  Result: 50.0 is assigned to large peripheral regions.

PROBLEM 3 — DEPTH MAP IS NOISY / RANDOM-LOOKING:
  The ZNCC is averaged across only 2 neighbours.
  Each pixel independently picks its "best" depth.
  With only 2 views, the depth estimate is very noisy.
  No cross-check (left-right consistency) is applied.
  No sub-pixel refinement.
  The fill_depth() then propagates noisy values everywhere.
  This creates a "salt-and-pepper" depth map with large depth variance.

PROBLEM 4 — d_max=50 POINTS LAND FAR FROM SCENE:
  Back-projecting d=50 for a camera looking at a scene ~6 units away
  places those points 50/6 ≈ 8x farther than the actual scene.
  The sparse cloud has extent ~15 units on each axis.
  Dense cloud has extent ~108 units (7x larger than sparse).
  This confirms that d_max=50 points are dominating the dense cloud.
""")
