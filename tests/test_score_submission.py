import math
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.score_submission as score_submission_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.score_submission import score_submission


def _write_scene(root: Path, scene_name: str, color: tuple[int, int, int]) -> Path:
    scene = root / scene_name
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)
    (scene / "test" / "images").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(scene / "train" / "images" / "train.png")
    Image.new("RGB", (4, 4), color=color).save(scene / "test" / "images" / "target_001.JPG")
    return scene


def test_score_submission_reports_aggregate_and_per_scene_metrics_for_folder(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_scene(data_root, "scene_a", (0, 0, 0))
    _write_scene(data_root, "scene_b", (255, 255, 255))
    submission = tmp_path / "submission"
    (submission / "scene_a").mkdir(parents=True)
    (submission / "scene_b").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(submission / "scene_a" / "target_001.png")
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(submission / "scene_b" / "target_001.png")

    result = score_submission(data_root=data_root, submission=submission, match_by_stem=True)

    assert result["aggregate"]["count"] == 2
    assert result["aggregate"]["mae"] == 0.0
    assert math.isinf(result["aggregate"]["psnr"])
    assert [scene["scene"] for scene in result["scenes"]] == ["scene_a", "scene_b"]


def test_score_submission_accepts_zip_submission(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_scene(data_root, "scene_a", (10, 20, 30))
    zip_path = tmp_path / "submission.zip"
    image_path = tmp_path / "target_001.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(image_path, "scene_a/target_001.png")

    result = score_submission(data_root=data_root, submission=zip_path, match_by_stem=True)

    assert result["aggregate"]["count"] == 1
    assert result["scenes"][0]["scene"] == "scene_a"


@pytest.mark.parametrize("unexpected_name", [None, "target_extra.JPG"])
def test_score_submission_rejects_missing_or_extra_images(tmp_path: Path, unexpected_name: str | None):
    data_root = tmp_path / "data"
    scene = _write_scene(data_root, "scene_a", (10, 20, 30))
    Image.new("RGB", (4, 4), color=(40, 50, 60)).save(scene / "test" / "images" / "target_002.JPG")
    submission = tmp_path / "submission"
    (submission / "scene_a").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(submission / "scene_a" / "target_001.png")
    if unexpected_name is not None:
        Image.new("RGB", (4, 4), color=(40, 50, 60)).save(submission / "scene_a" / "target_002.png")
        Image.new("RGB", (4, 4), color=(70, 80, 90)).save(submission / "scene_a" / unexpected_name)

    with pytest.raises(DataValidationError, match="exactly match"):
        score_submission(data_root=data_root, submission=submission, match_by_stem=True)


def test_score_submission_uses_equal_scene_mean_for_final_score(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "data"
    _write_scene(data_root, "scene_a", (10, 20, 30))
    scene_b = _write_scene(data_root, "scene_b", (40, 50, 60))
    for index in (2, 3):
        Image.new("RGB", (4, 4), color=(40, 50, 60)).save(
            scene_b / "test" / "images" / f"target_{index:03d}.JPG"
        )

    submission = tmp_path / "submission"
    for scene_name, count in (("scene_a", 1), ("scene_b", 3)):
        scene_dir = submission / scene_name
        scene_dir.mkdir(parents=True)
        for index in range(1, count + 1):
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(scene_dir / f"target_{index:03d}.png")

    def fake_evaluate(pred_dir, _gt_dir, **_kwargs):
        if Path(pred_dir).name == "scene_a":
            return {"count": 1, "mae": 1.0, "mse": 1.0, "psnr": 10.0, "ssim": 0.2, "lpips": 0.9,
                    "psnr_norm": 0.2, "score": 0.2}
        return {"count": 3, "mae": 3.0, "mse": 9.0, "psnr": 30.0, "ssim": 0.8, "lpips": 0.1,
                "psnr_norm": 0.6, "score": 0.8}

    monkeypatch.setattr(score_submission_module, "evaluate_directories", fake_evaluate)

    result = score_submission(data_root=data_root, submission=submission, match_by_stem=True)

    assert result["aggregate"]["count"] == 4
    assert result["aggregate"]["score"] == pytest.approx(0.5)
    assert result["aggregate"]["mae"] == pytest.approx(2.0)
    assert result["scoring"]["scene_aggregation"] == "equal_mean"


def test_score_submission_labels_default_psnr_max_as_local_proxy(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_scene(data_root, "scene_a", (10, 20, 30))
    submission = tmp_path / "submission"
    (submission / "scene_a").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(submission / "scene_a" / "target_001.png")

    result = score_submission(data_root=data_root, submission=submission, match_by_stem=True)

    assert result["scoring"] == {
        "label": "local_proxy_not_official_btc_score",
        "psnr_max": 50.0,
        "psnr_max_is_official": False,
        "scene_aggregation": "equal_mean",
    }
