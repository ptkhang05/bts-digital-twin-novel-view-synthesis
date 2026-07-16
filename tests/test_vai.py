from pathlib import Path

import pytest

from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import discover_vai_scenes, find_test_poses_csv, is_vai_scene


def _write_scene_layout(scene: Path) -> None:
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)


def test_discover_vai_scenes_uses_generic_current_dataset_layout(tmp_path: Path):
    _write_scene_layout(tmp_path / "bonsai")
    _write_scene_layout(tmp_path / "HCM0421")
    (tmp_path / "notes").mkdir()

    scenes = discover_vai_scenes(tmp_path)

    assert [scene.name for scene in scenes] == ["HCM0421", "bonsai"]
    assert all(is_vai_scene(scene) for scene in scenes)


def test_discover_vai_scenes_rejects_root_without_any_scene(tmp_path: Path):
    with pytest.raises(DataValidationError, match="No VAI scenes"):
        discover_vai_scenes(tmp_path)


def test_find_test_poses_csv_requires_canonical_current_dataset_filename(tmp_path: Path):
    scene = tmp_path / "scene"
    (scene / "test").mkdir(parents=True)
    (scene / "test" / "test_pose.csv").write_text("legacy prose example", encoding="utf-8")

    with pytest.raises(DataValidationError, match="test/test_poses.csv"):
        find_test_poses_csv(scene)
