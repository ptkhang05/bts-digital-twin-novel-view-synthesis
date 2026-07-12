import json
from pathlib import Path

import numpy as np
import pytest

from bts_nvs.distortion import redistort_image
from bts_nvs.exceptions import DataValidationError
from bts_nvs.render_exact import load_exact_targets


def test_load_exact_targets_uses_rectified_intrinsics_and_exact_names(tmp_path: Path):
    targets = {
        "camera_model": "OPENCV",
        "fl_x": 925.4770547129467,
        "fl_y": 925.4770547129467,
        "cx": 660.0,
        "cy": 494.5,
        "w": 1320,
        "h": 989,
        "k1": -0.114793940674488,
        "frames": [
            {
                "file_path": "test/images/HNI0131_target.JPG",
                "transform_matrix": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 3],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    path = tmp_path / "target_cameras.json"
    path.write_text(json.dumps(targets), encoding="utf-8")

    exact_targets = load_exact_targets(path)

    assert len(exact_targets) == 1
    assert exact_targets[0].name == "HNI0131_target.JPG"
    calibration = exact_targets[0].calibration
    assert calibration.render_width == 1320
    assert calibration.render_height == 989
    white_render = np.ones((calibration.render_height, calibration.render_width, 3), dtype=np.float32)
    restored = redistort_image(white_render, calibration)
    assert float(restored.min()) == 1.0


def test_load_exact_targets_rejects_fisheye_model(tmp_path: Path):
    targets = {
        "camera_model": "OPENCV_FISHEYE",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": "target.png",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    path = tmp_path / "target_cameras.json"
    path.write_text(json.dumps(targets), encoding="utf-8")

    with pytest.raises(DataValidationError, match="perspective"):
        load_exact_targets(path)
