"""
camera.py — Pinhole camera model, intrinsic/extrinsic management.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Camera:
    """Pinhole camera model: K, R, t."""
    image_name: str
    width: int
    height: int
    # Intrinsics
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    # Extrinsics (world-to-camera)
    R: Optional[np.ndarray] = field(default=None)  # (3,3)
    t: Optional[np.ndarray] = field(default=None)  # (3,)
    # State
    registered: bool = False
    scale: float = 1.0   # Scale factor if image was resized

    def __post_init__(self):
        if self.cx == 0:
            self.cx = self.width / 2.0
        if self.cy == 0:
            self.cy = self.height / 2.0

    @property
    def K(self) -> np.ndarray:
        """3×3 intrinsic matrix."""
        return np.array([
            [self.fx,  0.0,     self.cx],
            [0.0,      self.fy, self.cy],
            [0.0,      0.0,     1.0   ],
        ], dtype=np.float64)

    @property
    def Kinv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    @property
    def P(self) -> np.ndarray:
        """3×4 projection matrix P = K [R | t]."""
        if self.R is None or self.t is None:
            raise ValueError(f"Camera {self.image_name} has no pose.")
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt

    def set_identity_pose(self):
        """Set this camera as the reference (R=I, t=0)."""
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros(3, dtype=np.float64)
        self.registered = True

    def set_pose(self, R: np.ndarray, t: np.ndarray):
        self.R = R.astype(np.float64)
        self.t = t.astype(np.float64).ravel()
        self.registered = True

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """Project (N,3) world points → (N,2) image coords."""
        if self.R is None or self.t is None:
            raise ValueError("Camera has no pose")
        X_cam = (self.R @ X_world.T).T + self.t  # (N,3)
        # Filter behind camera
        valid = X_cam[:, 2] > 1e-6
        uvw = (self.K @ X_cam.T).T          # (N,3)
        uv = np.full((len(X_world), 2), np.nan)
        uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
        return uv

    def camera_center(self) -> np.ndarray:
        """World position of camera centre C = -R^T t."""
        return -(self.R.T @ self.t)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "R": self.R.tolist() if self.R is not None else None,
            "t": self.t.tolist() if self.t is not None else None,
            "registered": self.registered,
        }


def estimate_intrinsics(
    width: int,
    height: int,
    focal_length_px: Optional[float] = None,
    focal_ratio: float = 1.2,
) -> tuple:
    """Return (fx, fy, cx, cy) for a camera.

    If focal_length_px is given, use it directly.
    Otherwise estimate as focal_ratio × max(width, height).
    """
    if focal_length_px and focal_length_px > 0:
        fx = fy = focal_length_px
    else:
        fx = fy = focal_ratio * max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def build_cameras(
    images,                # List[Path]
    exif_data: Dict,       # image_name → exif dict
    focal_lengths: Dict,   # image_name → focal_length_px (may be None)
    cfg: Dict,
    proc_sizes: Dict,      # image_name → (proc_w, proc_h)
) -> Dict[str, Camera]:
    """Construct Camera objects for all images."""
    camera_cfg = cfg.get("camera", {})
    focal_ratio = camera_cfg.get("focal_length_ratio", 1.2)
    focal_override = camera_cfg.get("focal_length_override", None)

    cameras = {}
    for img_path in images:
        name = img_path.name
        proc_w, proc_h = proc_sizes.get(name, (0, 0))
        if proc_w == 0:
            continue
        fl_px = focal_override or focal_lengths.get(name)
        # Adjust focal length for processing scale
        orig_dims = exif_data.get(name, {}).get("_original_dims", None)
        if fl_px and orig_dims:
            orig_w, orig_h = orig_dims
            scale = proc_w / orig_w
            fl_px = fl_px * scale

        fx, fy, cx, cy = estimate_intrinsics(proc_w, proc_h, fl_px, focal_ratio)
        cam = Camera(
            image_name=name,
            width=proc_w,
            height=proc_h,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        cameras[name] = cam
    return cameras
