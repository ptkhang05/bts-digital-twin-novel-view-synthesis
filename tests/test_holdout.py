import json
import os
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.path_safety as path_safety
from bts_nvs.exceptions import DataValidationError
from bts_nvs.holdout import prepare_holdout
from bts_nvs.score_submission import score_submission


def _write_vai_scene(scene: Path, image_names: tuple[str, ...]) -> None:
    image_dir = scene / "train" / "images"
    sparse_dir = scene / "train" / "sparse" / "0"
    test_dir = scene / "test"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    image_lines = ["# Image list"]
    for image_id, name in enumerate(image_names, start=1):
        Image.new("RGB", (16, 12), color=(image_id * 20, 40, 60)).save(image_dir / name)
        image_lines.extend((f"{image_id} 1 0 0 0 0 0 -2 1 {name}", ""))

    (sparse_dir / "cameras.txt").write_text(
        "# Camera list\n1 SIMPLE_RADIAL 16 12 10 8 6 -0.1\n",
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("", encoding="utf-8")
    (sparse_dir / "points3D.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
        encoding="ascii",
    )
    (test_dir / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,0,0,-2,10,10,8,6,16,12\n",
        encoding="utf-8",
    )


def test_prepare_holdout_writes_deterministic_targets_and_scoreable_ground_truth(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root / "scene_b", ("z.png", "a.png", "c.png", "b.png"))
    _write_vai_scene(data_root / "scene_a", ("04.png", "01.png", "03.png", "02.png"))
    processed_root = tmp_path / "processed"
    ground_truth_root = tmp_path / "ground-truth"

    results = prepare_holdout(
        data_root=data_root,
        processed_root=processed_root,
        ground_truth_root=ground_truth_root,
        interval=2,
        copy_mode="hardlink",
    )

    assert [result.scene_id for result in results] == ["scene_a", "scene_b"]
    transforms = json.loads((processed_root / "scene_b" / "transforms.json").read_text(encoding="utf-8"))
    targets_path = processed_root / "scene_b" / "holdout_cameras.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    metadata = json.loads((processed_root / "scene_b" / "metadata.json").read_text(encoding="utf-8"))

    assert transforms["train_filenames"] == ["images/a.png", "images/c.png"]
    assert transforms["val_filenames"] == ["images/b.png", "images/z.png"]
    assert transforms["test_filenames"] == ["images/b.png", "images/z.png"]
    assert [frame["file_path"] for frame in targets["frames"]] == ["b.png", "z.png"]
    assert targets["k1"] == -0.1
    assert metadata["holdout_cameras"] == "holdout_cameras.json"
    assert metadata["holdout_interval"] == 2
    assert metadata["holdout_target_count"] == 2
    assert str(tmp_path) not in targets_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in json.dumps(metadata)

    expected_names = ["b.png", "z.png"]
    gt_images = ground_truth_root / "scene_b" / "test" / "images"
    assert sorted(path.name for path in gt_images.iterdir()) == expected_names
    for name in expected_names:
        assert os.path.samefile(processed_root / "scene_b" / "images" / name, gt_images / name)

    predictions = tmp_path / "predictions"
    for scene_result in results:
        source_dir = ground_truth_root / scene_result.scene_id / "test" / "images"
        destination_dir = predictions / scene_result.scene_id
        destination_dir.mkdir(parents=True)
        for source in source_dir.iterdir():
            Image.open(source).save(destination_dir / source.name)

    score = score_submission(data_root=ground_truth_root, submission=predictions)
    assert score["aggregate"]["count"] == 4
    assert [scene["scene"] for scene in score["scenes"]] == ["scene_a", "scene_b"]


def test_prepare_holdout_rejects_output_root_that_overlaps_raw_data(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root / "scene_a", ("01.png", "02.png"))

    with pytest.raises(DataValidationError, match="must not overlap"):
        prepare_holdout(
            data_root=data_root,
            processed_root=data_root,
            ground_truth_root=tmp_path / "ground-truth",
            interval=2,
        )


def test_prepare_holdout_rejects_link_like_content_before_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = tmp_path / "data"
    _write_vai_scene(data_root / "scene_a", ("01.png", "02.png"))
    processed_root = tmp_path / "processed"
    unsafe_entry = processed_root / "unsafe-junction"
    unsafe_entry.mkdir(parents=True)
    monkeypatch.setattr(path_safety, "is_link_or_junction", lambda path: Path(path) == unsafe_entry)

    with pytest.raises(DataValidationError, match="unsafe symlink or junction"):
        prepare_holdout(
            data_root=data_root,
            processed_root=processed_root,
            ground_truth_root=tmp_path / "ground-truth",
            interval=2,
            overwrite=True,
        )

    assert unsafe_entry.is_dir()
