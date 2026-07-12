import numpy as np

from bts_nvs.distortion import rectify_intrinsics, redistort_image


def test_rectify_intrinsics_keeps_pinhole_calibration_unchanged():
    calibration = rectify_intrinsics(
        width=16,
        height=12,
        fx=10.0,
        fy=11.0,
        cx=8.0,
        cy=6.0,
        k1=0.0,
    )

    assert calibration.render_width == 16
    assert calibration.render_height == 12
    assert calibration.render_fx == 10.0
    assert calibration.render_fy == 11.0
    assert calibration.render_cx == 8.0
    assert calibration.render_cy == 6.0


def test_rectify_intrinsics_matches_nerfstudio_crop_for_radial_camera():
    calibration = rectify_intrinsics(
        width=1320,
        height=989,
        fx=925.4770547129467,
        fy=925.4770547129467,
        cx=660.0,
        cy=494.5,
        k1=-0.114793940674488,
    )

    assert calibration.render_width == 1319
    assert calibration.render_height == 988
    assert calibration.render_fx != calibration.render_fy


def test_redistort_image_restores_raw_dimensions_and_smooth_content():
    calibration = rectify_intrinsics(
        width=64,
        height=48,
        fx=45.0,
        fy=45.0,
        cx=32.0,
        cy=24.0,
        k1=-0.12,
    )
    yy, xx = np.mgrid[: calibration.render_height, : calibration.render_width]
    undistorted = np.stack(
        [
            xx / max(calibration.render_width - 1, 1),
            yy / max(calibration.render_height - 1, 1),
            np.full_like(xx, 0.5, dtype=np.float64),
        ],
        axis=-1,
    ).astype(np.float32)

    distorted = redistort_image(undistorted, calibration)

    assert distorted.shape == (48, 64, 3)
    assert np.isfinite(distorted).all()
    assert float(distorted[24, 32, 2]) == 0.5
