import math

import numpy as np

from bts_nvs.camera import (
    compute_vertical_fov_degrees,
    opencv_camera_center_to_nerfstudio_c2w,
    opencv_w2c_to_nerfstudio_c2w,
    qvec_to_rotmat,
)


def test_identity_colmap_pose_flips_to_nerfstudio_camera_axes():
    c2w = opencv_w2c_to_nerfstudio_c2w(
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([0.0, 0.0, 0.0]),
    )

    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    np.testing.assert_allclose(c2w, expected, atol=1e-8)


def test_colmap_translation_is_converted_to_camera_center():
    c2w = opencv_w2c_to_nerfstudio_c2w(
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([0.0, 0.0, -2.0]),
    )

    np.testing.assert_allclose(c2w[:3, 3], np.array([0.0, 0.0, 2.0]), atol=1e-8)


def test_camera_center_pose_keeps_translation_and_flips_axes():
    c2w = opencv_camera_center_to_nerfstudio_c2w(
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        camera_center=np.array([1.0, 2.0, 3.0]),
    )

    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(c2w, expected, atol=1e-8)


def test_qvec_to_rotmat_normalizes_input_quaternion():
    rot = qvec_to_rotmat(np.array([2.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rot, np.eye(3), atol=1e-8)


def test_vertical_fov_uses_image_height_and_focal_length():
    fov = compute_vertical_fov_degrees(height=480, fy=480)

    assert math.isclose(fov, 53.13010235415598, rel_tol=1e-12)
