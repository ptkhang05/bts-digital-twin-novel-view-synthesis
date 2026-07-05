from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bts_nvs.camera import opencv_w2c_to_nerfstudio_c2w
from bts_nvs.exceptions import DataValidationError

TEST_POSE_COLUMNS = (
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
)

TEST_POSE_FILENAMES = ("test_poses.csv", "test_pose.csv")


def is_vai_phase1_scene(scene: Path) -> bool:
    return (scene / "train" / "images").is_dir() and (scene / "train" / "sparse" / "0").is_dir()


def discover_vai_phase1_scenes(root: Path) -> list[Path]:
    if is_vai_phase1_scene(root):
        return [root]
    if not root.is_dir():
        raise DataValidationError(f"Dataset root does not exist: {root}")
    scenes = [path for path in sorted(root.iterdir()) if path.is_dir() and is_vai_phase1_scene(path)]
    if not scenes:
        raise DataValidationError(f"No VAI phase1 scenes found under {root}")
    return scenes


def find_test_poses_csv(scene: Path) -> Path:
    test_dir = scene / "test"
    for filename in TEST_POSE_FILENAMES:
        candidate = test_dir / filename
        if candidate.exists():
            return candidate
    expected = ", ".join(f"test/{filename}" for filename in TEST_POSE_FILENAMES)
    raise DataValidationError(f"VAI scene is missing target pose CSV. Expected one of: {expected} under {scene}")


def train_image_names(scene: Path) -> set[str]:
    image_dir = scene / "train" / "images"
    if not image_dir.is_dir():
        raise DataValidationError(f"VAI scene is missing train/images: {scene}")
    names = {path.name for path in image_dir.iterdir() if path.is_file()}
    if not names:
        raise DataValidationError(f"VAI scene contains no train images: {image_dir}")
    return names


def test_poses_csv_to_transforms(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataValidationError(f"target pose CSV does not exist: {path}")
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in TEST_POSE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise DataValidationError(f"{path} is missing required columns: {', '.join(missing)}")
        for row_index, row in enumerate(reader, start=2):
            frames.append(_test_pose_row_to_frame(row, row_index=row_index))
    if not frames:
        raise DataValidationError(f"{path} contains no target poses")
    return {"camera_model": "OPENCV", "frames": frames}


def _test_pose_row_to_frame(row: dict[str, str], row_index: int) -> dict[str, Any]:
    image_name = row["image_name"].strip()
    if not image_name:
        raise DataValidationError(f"Missing image_name in test_poses.csv row {row_index}")
    qvec = [_float(row, key, row_index) for key in ("qw", "qx", "qy", "qz")]
    tvec = [_float(row, key, row_index) for key in ("tx", "ty", "tz")]
    return {
        "file_path": image_name,
        "transform_matrix": opencv_w2c_to_nerfstudio_c2w(qvec, tvec).tolist(),
        "fl_x": _float(row, "fx", row_index),
        "fl_y": _float(row, "fy", row_index),
        "cx": _float(row, "cx", row_index),
        "cy": _float(row, "cy", row_index),
        "w": _int(row, "width", row_index),
        "h": _int(row, "height", row_index),
    }


def _float(row: dict[str, str], key: str, row_index: int) -> float:
    try:
        return float(row[key])
    except ValueError as exc:
        raise DataValidationError(f"Invalid float for {key} in test_poses.csv row {row_index}: {row[key]}") from exc


def _int(row: dict[str, str], key: str, row_index: int) -> int:
    try:
        value = int(float(row[key]))
    except ValueError as exc:
        raise DataValidationError(f"Invalid integer for {key} in test_poses.csv row {row_index}: {row[key]}") from exc
    if value <= 0:
        raise DataValidationError(f"{key} must be positive in test_poses.csv row {row_index}")
    return value
