import json
from pathlib import Path

import pytest
from PIL import Image

from bts_nvs.exceptions import DataValidationError
from bts_nvs.render import (
    _rename_rendered_images,
    build_camera_path,
    build_exact_render_command,
    build_render_command,
    render_targets,
    rendered_image_directory,
    targets_have_lens_distortion,
)


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


def test_build_camera_path_preserves_exact_non_png_target_names(tmp_path: Path):
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
                "file_path": "test/images/HCM0249_0042_V.JPG",
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

    _, names = build_camera_path(target_file)

    assert names == ["HCM0249_0042_V.JPG"]


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


def test_build_exact_render_command_uses_current_nerfstudio_python(tmp_path: Path):
    command = build_exact_render_command(
        checkpoint=tmp_path / "config.yml",
        targets=tmp_path / "target_cameras.json",
        output=tmp_path / "submission",
    )

    assert command[1:3] == ["-m", "bts_nvs.render_exact"]
    assert command[command.index("--checkpoint") + 1] == str(tmp_path / "config.yml")
    assert command[command.index("--targets") + 1] == str(tmp_path / "target_cameras.json")


def test_render_targets_auto_distortion_routes_nonzero_model_to_exact_renderer(tmp_path: Path):
    targets = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "k1": -0.1,
        "frames": [
            {
                "file_path": "target.JPG",
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

    command = render_targets(
        checkpoint=tmp_path / "config.yml",
        targets=target_file,
        output=tmp_path / "submission",
        dry_run=True,
    )

    assert command[1:3] == ["-m", "bts_nvs.render_exact"]


def test_targets_have_lens_distortion_is_false_for_pinhole_targets(tmp_path: Path):
    targets = {
        "camera_model": "SIMPLE_PINHOLE",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": "target.jpg",
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

    assert targets_have_lens_distortion(target_file) is False
    command = render_targets(
        checkpoint=tmp_path / "config.yml",
        targets=target_file,
        output=tmp_path / "submission",
        dry_run=True,
    )
    assert command[:2] == ["ns-render", "camera-path"]


def test_render_targets_failure_preserves_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "camera_model": "SIMPLE_PINHOLE",
        "fl_x": 10.0,
        "fl_y": 10.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": "target.jpg",
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
    output = tmp_path / "candidate"
    output.mkdir()
    previous = output / "target.jpg"
    previous.write_bytes(b"previous-good-render")

    def fail_command(_command: list[str]) -> None:
        raise RuntimeError("simulated renderer failure")

    monkeypatch.setattr("bts_nvs.render.run_external_command", fail_command)

    with pytest.raises(RuntimeError, match="simulated renderer failure"):
        render_targets(tmp_path / "config.yml", target_file, output)

    assert previous.read_bytes() == b"previous-good-render"
    assert not list(tmp_path.glob(".candidate.*"))


def test_rendered_image_directory_matches_nerfstudio_image_output_rule(tmp_path: Path):
    output_path = tmp_path / "submission" / "targets"

    assert rendered_image_directory(output_path) == output_path


def test_rename_rendered_images_encodes_jpeg_when_target_name_is_jpg(tmp_path: Path):
    render_dir = tmp_path / "renders"
    output_dir = tmp_path / "submission"
    render_dir.mkdir()
    output_dir.mkdir()
    Image.new("RGB", (4, 3), color=(200, 10, 30)).save(render_dir / "00000.png")

    _rename_rendered_images(render_dir, output_dir, ["HCM0249_0042_V.JPG"])

    output = output_dir / "HCM0249_0042_V.JPG"
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (4, 3)
