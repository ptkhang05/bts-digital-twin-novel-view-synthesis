from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _as_float_array(values: Iterable[float], expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def qvec_to_rotmat(qvec: Iterable[float]) -> np.ndarray:
    """Convert a COLMAP WXYZ quaternion to a 3x3 rotation matrix."""
    q = _as_float_array(qvec, (4,), "qvec")
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("qvec must not be zero")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def opencv_w2c_to_nerfstudio_c2w(qvec: Iterable[float], tvec: Iterable[float]) -> np.ndarray:
    """Convert COLMAP/OpenCV world-to-camera pose to Nerfstudio/OpenGL camera-to-world.

    COLMAP stores x_cam = R * x_world + t in an OpenCV camera frame
    (x right, y down, z forward). Nerfstudio camera frames use OpenGL axes
    (x right, y up, z backward), so the camera-frame Y and Z axes are flipped.
    """
    rot_w2c = qvec_to_rotmat(qvec)
    trans_w2c = _as_float_array(tvec, (3,), "tvec")
    c2w_opencv = np.eye(4, dtype=np.float64)
    c2w_opencv[:3, :3] = rot_w2c.T
    c2w_opencv[:3, 3] = -rot_w2c.T @ trans_w2c
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_opencv @ opencv_to_opengl


def opencv_camera_center_to_nerfstudio_c2w(qvec: Iterable[float], camera_center: Iterable[float]) -> np.ndarray:
    """Convert OpenCV camera rotation plus world-space camera center to Nerfstudio c2w."""
    rot_w2c = qvec_to_rotmat(qvec)
    center = _as_float_array(camera_center, (3,), "camera_center")
    c2w_opencv = np.eye(4, dtype=np.float64)
    c2w_opencv[:3, :3] = rot_w2c.T
    c2w_opencv[:3, 3] = center
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w_opencv @ opencv_to_opengl


def validate_transform_matrix(matrix: Iterable[Iterable[float]], name: str = "transform_matrix") -> np.ndarray:
    array = _as_float_array(matrix, (4, 4), name)
    if not np.allclose(array[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    return array


def compute_vertical_fov_degrees(height: int, fy: float) -> float:
    if height <= 0:
        raise ValueError("height must be positive")
    if fy <= 0:
        raise ValueError("fy must be positive")
    return math.degrees(2.0 * math.atan(float(height) / (2.0 * float(fy))))
