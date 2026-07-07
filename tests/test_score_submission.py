import math
import zipfile
from pathlib import Path

from PIL import Image

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
