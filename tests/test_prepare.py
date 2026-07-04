import json
from pathlib import Path

import pytest
from PIL import Image

from bts_nvs.exceptions import DataValidationError
from bts_nvs.prepare import prepare_scene


def _write_minimal_scene(scene: Path, image_name: str = "frame.png") -> None:
    (scene / "images").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(100, 120, 140)).save(scene / "images" / image_name)
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_target_cameras(scene: Path, count: int = 2) -> None:
    frames = []
    for index in range(count):
        frames.append(
            {
                "file_path": f"target_{index:03d}.png",
                "transform_matrix": [
                    [1, 0, 0, index],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": frames,
    }
    (scene / "target_cameras.json").write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_scene_copies_images_and_writes_nerfstudio_transforms(tmp_path: Path):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)

    result = prepare_scene(scene=scene, output=output)

    assert result.transforms_path == output / "transforms.json"
    assert (output / "images" / "frame.png").exists()
    transforms = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert transforms["frames"][0]["file_path"] == "images/frame.png"
    assert metadata["image_count"] == 1


def test_prepare_scene_copies_target_cameras_when_present(tmp_path: Path):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)
    _write_target_cameras(scene, count=2)

    result = prepare_scene(scene=scene, output=output)

    targets = json.loads((output / "target_cameras.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert result.target_count == 2
    assert len(targets["frames"]) == 2
    assert metadata["target_count"] == 2


def test_prepare_scene_rejects_missing_frame_image(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene, image_name="missing.png")
    (scene / "images" / "missing.png").unlink()

    with pytest.raises(DataValidationError, match="missing.png"):
        prepare_scene(scene=scene, output=tmp_path / "processed")


def test_prepare_scene_strict_contest_rejects_too_few_phase1_training_images(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene)

    with pytest.raises(DataValidationError, match="100-300"):
        prepare_scene(scene=scene, output=tmp_path / "processed", strict_contest=True)


def test_prepare_scene_supports_vai_phase1_layout_and_filters_sparse_images(tmp_path: Path):
    scene = tmp_path / "official_scene"
    sparse = scene / "train" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "train" / "images" / "keep.png")

    (sparse / "cameras.txt").write_text(
        "# Camera list\n"
        "1 PINHOLE 16 12 10 11 8 6\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 -2 1 keep.png\n"
        "\n"
        "2 1 0 0 0 0 0 -3 1 missing.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )

    result = prepare_scene(scene=scene, output=tmp_path / "processed")

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    targets = json.loads((tmp_path / "processed" / "target_cameras.json").read_text(encoding="utf-8"))
    assert result.image_count == 1
    assert transforms["frames"][0]["file_path"] == "images/keep.png"
    assert [frame["file_path"] for frame in transforms["frames"]] == ["images/keep.png"]
    assert result.target_count == 1
    assert targets["frames"][0]["file_path"] == "target.JPG"
    assert targets["frames"][0]["transform_matrix"][0][3] == -1.0


def test_prepare_scene_can_write_filename_holdout_split(tmp_path: Path):
    scene = tmp_path / "raw"
    (scene / "images").mkdir(parents=True)
    frames = []
    for index in range(4):
        name = f"frame_{index}.png"
        Image.new("RGB", (16, 12), color=(index, index, index)).save(scene / "images" / name)
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": [
                    [1, 0, 0, index],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": frames,
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")

    prepare_scene(scene=scene, output=tmp_path / "processed", holdout_interval=2)

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert transforms["train_filenames"] == ["images/frame_0.png", "images/frame_2.png"]
    assert transforms["val_filenames"] == ["images/frame_1.png", "images/frame_3.png"]
    assert transforms["test_filenames"] == ["images/frame_1.png", "images/frame_3.png"]
