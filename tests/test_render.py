import json
from pathlib import Path

import pytest

from bts_nvs.exceptions import DataValidationError
from bts_nvs.render import build_camera_path, build_render_command, rendered_image_directory


def test_build_camera_path_converts_target_transforms_to_nerfstudio_camera_path(tmp_path: Path):
    targets = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": "target_000.png",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    target_file = tmp_path / "target_cameras.json"
    target_file.write_text(json.dumps(targets), encoding="utf-8")

    camera_path, names = build_camera_path(target_file)

    assert names == ["target_000.png"]
    assert camera_path["render_width"] == 16
    assert camera_path["render_height"] == 12
    assert camera_path["camera_path"][0]["camera_to_world"] == targets["frames"][0]["transform_matrix"]
    assert camera_path["camera_path"][0]["fov"] > 0


def test_build_camera_path_strict_contest_rejects_too_few_targets(tmp_path: Path):
    targets = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": "target_000.png",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    target_file = tmp_path / "target_cameras.json"
    target_file.write_text(json.dumps(targets), encoding="utf-8")

    with pytest.raises(DataValidationError, match="40-70"):
        build_camera_path(target_file, strict_contest=True)


def test_build_render_command_requests_lossless_image_sequence(tmp_path: Path):
    command = build_render_command(
        checkpoint=tmp_path / "config.yml",
        camera_path_file=tmp_path / "camera_path.json",
        output_path=tmp_path / "renders" / "targets",
    )

    assert command[:2] == ["ns-render", "camera-path"]
    assert "--output-format" in command
    assert command[command.index("--output-format") + 1] == "images"
    assert "--image-format" in command
    assert command[command.index("--image-format") + 1] == "png"


def test_rendered_image_directory_matches_nerfstudio_image_output_rule(tmp_path: Path):
    output_path = tmp_path / "submission" / "targets"

    assert rendered_image_directory(output_path) == output_path
