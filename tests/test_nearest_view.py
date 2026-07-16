import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import bts_nvs.nearest_view as nearest_view_module
import bts_nvs.path_safety as path_safety_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.nearest_view import _read_target_poses, render_nearest_dataset


def _write_nearest_scene(scene: Path, test_pose_name: str = "test_poses.csv") -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "near_red.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "near_blue.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 near_red.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 near_blue.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / test_pose_name).write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def _write_temporal_scene(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "DJI_20250101000001_0001_V.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "DJI_20250101000003_0003_V.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 DJI_20250101000001_0001_V.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 DJI_20250101000003_0003_V.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "DJI_20250101000002_0002_V.png,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def _write_temporal_gap_scene(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "DJI_20250101000001_0001_V.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "DJI_20250101000004_0004_V.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 DJI_20250101000001_0001_V.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 DJI_20250101000004_0004_V.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "DJI_20250101000002_0002_V.png,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def test_render_nearest_dataset_writes_exact_target_name_by_default(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")

    result = render_nearest_dataset(root=root, output=tmp_path / "submission")

    output = tmp_path / "submission" / "scene_a" / "target.JPG"
    assert result.scene_count == 1
    assert result.image_count == 1
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (2, 2)


def test_render_nearest_dataset_rejects_noncanonical_singular_test_pose_csv_name(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a", test_pose_name="test_pose.csv")

    with pytest.raises(DataValidationError, match="test/test_poses.csv"):
        render_nearest_dataset(root=root, output=tmp_path / "submission")


def test_render_nearest_dataset_can_force_png_stem_names(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", name_policy="png", image_format="png")

    output = tmp_path / "submission" / "scene_a" / "target.png"
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
        assert image.getpixel((0, 0)) == (255, 0, 0)


def test_render_nearest_dataset_atomically_replaces_previous_output(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")
    output = tmp_path / "submission"
    output.mkdir()
    (output / "stale.txt").write_text("old", encoding="utf-8")

    render_nearest_dataset(root=root, output=output)

    assert not (output / "stale.txt").exists()
    assert (output / "scene_a" / "target.JPG").is_file()


def test_render_nearest_dataset_failure_preserves_entire_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")
    _write_nearest_scene(root / "scene_b")
    output = tmp_path / "submission"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")
    real_writer = nearest_view_module._write_selected_prediction
    calls = 0

    def fail_on_second_prediction(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated nearest-view failure")
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(nearest_view_module, "_write_selected_prediction", fail_on_second_prediction)

    with pytest.raises(RuntimeError, match="simulated nearest-view failure"):
        render_nearest_dataset(root=root, output=output)

    assert [path.relative_to(output).as_posix() for path in output.rglob("*")] == ["previous.txt"]
    assert sentinel.read_text(encoding="utf-8") == "last-known-good"
    assert not list(tmp_path.glob(".submission.*"))


def test_render_nearest_dataset_promotion_failure_restores_previous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")
    output = tmp_path / "submission"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")
    real_replace = os.replace
    failed = False

    def fail_staging_promotion(source, destination):
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if not failed and destination_path == output and source_path.name.startswith(".submission."):
            failed = True
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(path_safety_module.os, "replace", fail_staging_promotion)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        render_nearest_dataset(root=root, output=output)

    assert sentinel.read_text(encoding="utf-8") == "last-known-good"
    assert [path.relative_to(output).as_posix() for path in output.rglob("*")] == ["previous.txt"]
    assert not list(tmp_path.glob(".submission.*"))


def test_render_nearest_dataset_defaults_to_jpeg_quality_95(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_nearest_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "default")
    render_nearest_dataset(root=root, output=tmp_path / "explicit", jpeg_quality=95)

    default_jpeg = tmp_path / "default" / "scene_a" / "target.JPG"
    explicit_jpeg = tmp_path / "explicit" / "scene_a" / "target.JPG"
    assert default_jpeg.read_bytes() == explicit_jpeg.read_bytes()


def test_read_target_poses_converts_colmap_world_to_camera_translation_to_camera_center(tmp_path: Path):
    poses = tmp_path / "test_poses.csv"
    poses.write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,0,0,0,1,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )

    target = _read_target_poses(poses)[0]

    np.testing.assert_allclose(target.camera_center, np.asarray([1.0, 2.0, -3.0]))


def test_render_nearest_dataset_temporal_blend_uses_bracketing_frame_indices(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_temporal_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", selection_mode="temporal-blend")

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
        red, green, blue = image.getpixel((0, 0))
        assert 120 <= red <= 135
        assert green == 0
        assert 120 <= blue <= 135


def test_render_nearest_dataset_temporal_blend_defaults_to_linear_frame_weight(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_temporal_gap_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", selection_mode="temporal-blend")

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    with Image.open(output) as image:
        red, green, blue = image.getpixel((0, 0))
        assert 165 <= red <= 175
        assert green == 0
        assert 80 <= blue <= 90


def test_render_nearest_dataset_temporal_blend_can_use_midpoint_weight(tmp_path: Path):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_temporal_gap_scene(root / "scene_a")

    render_nearest_dataset(
        root=root,
        output=tmp_path / "submission",
        selection_mode="temporal-blend",
        blend_weight_policy="midpoint",
    )

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    with Image.open(output) as image:
        red, green, blue = image.getpixel((0, 0))
        assert 120 <= red <= 135
        assert green == 0
        assert 120 <= blue <= 135
