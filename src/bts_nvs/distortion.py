from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bts_nvs.exceptions import DataValidationError


@dataclass(frozen=True)
class RectifiedCalibration:
    raw_width: int
    raw_height: int
    raw_fx: float
    raw_fy: float
    raw_cx: float
    raw_cy: float
    distortion: tuple[float, float, float, float, float, float]
    render_width: int
    render_height: int
    render_fx: float
    render_fy: float
    render_cx: float
    render_cy: float


def rectify_intrinsics(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> RectifiedCalibration:
    """Match Nerfstudio's perspective-image rectification and ROI crop."""
    if width <= 0 or height <= 0:
        raise DataValidationError("Camera width and height must be positive")
    if fx <= 0 or fy <= 0:
        raise DataValidationError("Camera focal lengths must be positive")
    if k4 != 0.0:
        raise DataValidationError("Perspective rectification does not support a non-zero k4 coefficient")

    cv2 = _load_cv2()
    raw_k = _opencv_camera_matrix(fx=fx, fy=fy, cx=cx, cy=cy)
    distortion = (float(k1), float(k2), float(k3), float(k4), float(p1), float(p2))
    cv_distortion = _opencv_distortion(distortion)
    if np.any(cv_distortion):
        render_k, roi = cv2.getOptimalNewCameraMatrix(raw_k, cv_distortion, (width, height), 0)
    else:
        render_k = raw_k.copy()
        roi = (0, 0, width, height)

    x, y, render_width, render_height = (int(value) for value in roi)
    if render_width <= 0 or render_height <= 0:
        raise DataValidationError("Camera distortion produced an empty rectified image")
    render_k[0, 2] -= x
    render_k[1, 2] -= y

    return RectifiedCalibration(
        raw_width=width,
        raw_height=height,
        raw_fx=float(fx),
        raw_fy=float(fy),
        raw_cx=float(cx),
        raw_cy=float(cy),
        distortion=distortion,
        render_width=render_width,
        render_height=render_height,
        render_fx=float(render_k[0, 0]),
        render_fy=float(render_k[1, 1]),
        render_cx=float(render_k[0, 2] + 0.5),
        render_cy=float(render_k[1, 2] + 0.5),
    )


def redistort_image(image: np.ndarray, calibration: RectifiedCalibration) -> np.ndarray:
    """Map a Nerfstudio-rectified render back to the raw contest camera image."""
    expected_shape = (calibration.render_height, calibration.render_width)
    if image.ndim not in (2, 3) or image.shape[:2] != expected_shape:
        raise DataValidationError(
            f"Rectified image shape must start with {expected_shape}, got {image.shape}"
        )

    if not any(calibration.distortion):
        return image.copy()

    cv2 = _load_cv2()
    raw_k = _opencv_camera_matrix(
        fx=calibration.raw_fx,
        fy=calibration.raw_fy,
        cx=calibration.raw_cx,
        cy=calibration.raw_cy,
    )
    render_k = _opencv_camera_matrix(
        fx=calibration.render_fx,
        fy=calibration.render_fy,
        cx=calibration.render_cx,
        cy=calibration.render_cy,
    )
    yy, xx = np.mgrid[: calibration.raw_height, : calibration.raw_width]
    raw_pixels = np.stack((xx, yy), axis=-1).astype(np.float32).reshape(-1, 1, 2)
    rectified_pixels = cv2.undistortPoints(
        raw_pixels,
        raw_k,
        _opencv_distortion(calibration.distortion),
        P=render_k,
    ).reshape(calibration.raw_height, calibration.raw_width, 2)
    return cv2.remap(
        image,
        rectified_pixels[..., 0],
        rectified_pixels[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _opencv_camera_matrix(*, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx - 0.5], [0.0, fy, cy - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _opencv_distortion(values: tuple[float, float, float, float, float, float]) -> np.ndarray:
    k1, k2, k3, k4, p1, p2 = values
    return np.array([k1, k2, p1, p2, k3, k4, 0.0, 0.0], dtype=np.float64)


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise DataValidationError(
            "Lens-distortion rendering requires OpenCV. Run this command in the Nerfstudio environment."
        ) from exc
    return cv2
