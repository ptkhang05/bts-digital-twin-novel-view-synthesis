import zipfile
from pathlib import Path

import pytest
from PIL import Image

from bts_nvs.exceptions import DataValidationError
from bts_nvs.validate_submission import validate_submission


def _write_vai_scene(root: Path, scene_name: str = "scene_a", test_pose_name: str = "test_poses.csv") -> Path:
    scene = root / scene_name
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(scene / "train" / "images" / "train.png")
    (scene / "test" / test_pose_name).write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target_001.JPG,1,0,0,0,0,0,0,10,10,4,3,8,6\n"
        "target_002.JPG,1,0,0,0,0,0,0,10,10,4,3,4,3\n",
        encoding="utf-8",
    )
    return scene


def test_validate_submission_accepts_folder_with_exact_names_and_sizes(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root)
    submission = tmp_path / "submission" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(submission / "target_001.JPG")
    Image.new("RGB", (4, 3), color=(30, 20, 10)).save(submission / "target_002.JPG")

    result = validate_submission(data_root=data_root, submission=submission.parent)

    assert result.valid
    assert result.scene_count == 1
    assert result.image_count == 2


def test_validate_submission_accepts_singular_test_pose_csv_name(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root, test_pose_name="test_pose.csv")
    submission = tmp_path / "submission" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(submission / "target_001.JPG")
    Image.new("RGB", (4, 3), color=(30, 20, 10)).save(submission / "target_002.JPG")

    result = validate_submission(data_root=data_root, submission=submission.parent)

    assert result.valid
    assert result.image_count == 2


def test_validate_submission_rejects_missing_output_image(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root)
    submission = tmp_path / "submission" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(submission / "target_001.JPG")

    with pytest.raises(DataValidationError, match="Missing output image: scene_a/target_002.JPG"):
        validate_submission(data_root=data_root, submission=submission.parent).raise_for_errors()


def test_validate_submission_rejects_wrong_image_size(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root)
    submission = tmp_path / "submission" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (7, 6), color=(10, 20, 30)).save(submission / "target_001.JPG")
    Image.new("RGB", (4, 3), color=(30, 20, 10)).save(submission / "target_002.JPG")

    with pytest.raises(DataValidationError, match=r"Image size mismatch for scene_a/target_001.JPG"):
        validate_submission(data_root=data_root, submission=submission.parent).raise_for_errors()


def test_validate_submission_accepts_zip_without_wrapper_folder(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root)
    source = tmp_path / "source" / "scene_a"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(source / "target_001.JPG")
    Image.new("RGB", (4, 3), color=(30, 20, 10)).save(source / "target_002.JPG")
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for image_path in source.iterdir():
            archive.write(image_path, f"scene_a/{image_path.name}")

    result = validate_submission(data_root=data_root, submission=zip_path)

    assert result.valid
    assert result.image_count == 2


def test_validate_submission_rejects_extra_zip_wrapper_folder(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root)
    zip_path = tmp_path / "submission.zip"
    image_path = tmp_path / "target_001.JPG"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(image_path, "submission/scene_a/target_001.JPG")

    with pytest.raises(DataValidationError, match="Missing scene folder: scene_a"):
        validate_submission(data_root=data_root, submission=zip_path).raise_for_errors()
