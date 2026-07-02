import zipfile
from pathlib import Path

import pytest
from PIL import Image

from bts_nvs.exceptions import DataValidationError
from bts_nvs.package import create_submission_zip


def _write_scene_outputs(root: Path, scene_id: str, count: int) -> None:
    scene_dir = root / scene_id
    scene_dir.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(scene_dir / f"target_{index:03d}.png")


def test_create_submission_zip_keeps_scene_directories_and_pngs(tmp_path: Path):
    submission = tmp_path / "submission"
    _write_scene_outputs(submission, "scene_001", 40)
    _write_scene_outputs(submission, "scene_002", 40)

    result = create_submission_zip(submission, tmp_path / "submission.zip")

    assert result.scene_count == 2
    assert result.image_count == 80
    with zipfile.ZipFile(result.zip_path) as archive:
        names = archive.namelist()
    assert "scene_001/target_000.png" in names
    assert "scene_002/target_039.png" in names


def test_create_submission_zip_strict_contest_accepts_phase1_upper_range(tmp_path: Path):
    submission = tmp_path / "submission"
    _write_scene_outputs(submission, "scene_001", 65)

    result = create_submission_zip(submission, tmp_path / "submission.zip")

    assert result.image_count == 65


def test_create_submission_zip_strict_contest_rejects_scene_with_too_few_phase1_targets(tmp_path: Path):
    submission = tmp_path / "submission"
    _write_scene_outputs(submission, "scene_001", 39)

    with pytest.raises(DataValidationError, match="40-70"):
        create_submission_zip(submission, tmp_path / "submission.zip")
