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


def test_prepare_scene_rejects_missing_frame_image(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene, image_name="missing.png")
    (scene / "images" / "missing.png").unlink()

    with pytest.raises(DataValidationError, match="missing.png"):
        prepare_scene(scene=scene, output=tmp_path / "processed")


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
