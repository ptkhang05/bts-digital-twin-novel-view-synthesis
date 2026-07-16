import hashlib
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.package as package_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.package import create_submission_zip


def _write_dataset_and_outputs(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    scene = data_root / "scene_a"
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(scene / "train" / "images" / "train.JPG", format="JPEG")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target_001.JPG,1,0,0,0,0,0,0,10,10,4,3,8,6\n"
        "target_002.jpg,1,0,0,0,0,0,0,10,10,2,2,4,3\n",
        encoding="utf-8",
    )
    submission = tmp_path / "outputs" / "candidate" / "rendered" / "scene_a"
    submission.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(submission / "target_001.JPG", format="JPEG")
    Image.new("RGB", (4, 3), color=(30, 20, 10)).save(submission / "target_002.jpg", format="JPEG")
    return data_root, submission.parent


def test_create_submission_zip_validates_exact_dataset_and_returns_sha256(tmp_path: Path):
    data_root, submission = _write_dataset_and_outputs(tmp_path)
    output = tmp_path / "submission.zip"

    result = create_submission_zip(data_root=data_root, submission=submission, output=output)

    assert result.zip_path == output
    assert result.scene_count == 1
    assert result.image_count == 2
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["scene_a/target_001.JPG", "scene_a/target_002.jpg"]


def test_create_submission_zip_rejects_non_exact_output_before_replacing_zip(tmp_path: Path):
    data_root, submission = _write_dataset_and_outputs(tmp_path)
    (submission / "scene_a" / "target_002.jpg").unlink()
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous valid artifact")

    with pytest.raises(DataValidationError, match="Missing output image"):
        create_submission_zip(data_root=data_root, submission=submission, output=output)

    assert output.read_bytes() == b"previous valid artifact"


def test_create_submission_zip_interruption_preserves_previous_zip_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root, submission = _write_dataset_and_outputs(tmp_path)
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous valid artifact")
    staging = tmp_path / "outputs" / ".staging"

    def interrupt(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(package_module, "_write_zip_archive", interrupt)

    with pytest.raises(OSError, match="simulated interruption"):
        create_submission_zip(
            data_root=data_root,
            submission=submission,
            output=output,
            staging_dir=staging,
        )

    assert output.read_bytes() == b"previous valid artifact"
    assert list(staging.iterdir()) == []


def test_create_submission_zip_rejects_output_inside_submission_tree(tmp_path: Path):
    data_root, submission = _write_dataset_and_outputs(tmp_path)

    with pytest.raises(DataValidationError, match="outside the submission directory"):
        create_submission_zip(
            data_root=data_root,
            submission=submission,
            output=submission / "submission.zip",
        )
