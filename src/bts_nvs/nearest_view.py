from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bts_nvs.camera import qvec_to_rotmat
from bts_nvs.colmap import read_colmap_model
from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import TEST_POSE_COLUMNS, discover_vai_phase1_scenes, train_image_names


@dataclass(frozen=True)
class NearestViewSubmission:
    output_dir: Path
    scene_count: int
    image_count: int


@dataclass(frozen=True)
class TargetPose:
    image_name: str
    camera_center: np.ndarray
    width: int
    height: int


def render_nearest_dataset(root: Path | str, output: Path | str) -> NearestViewSubmission:
    root_path = Path(root)
    output_path = Path(output)
    scene_count = 0
    image_count = 0
    for scene in discover_vai_phase1_scenes(root_path):
        image_count += render_nearest_scene(scene, output_path / scene.name)
        scene_count += 1
    return NearestViewSubmission(output_path, scene_count=scene_count, image_count=image_count)


def render_nearest_scene(scene: Path | str, output: Path | str) -> int:
    scene_path = Path(scene)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_png in output_path.glob("*.png"):
        stale_png.unlink()

    train_centers = _read_train_camera_centers(scene_path)
    targets = _read_target_poses(scene_path / "test" / "test_poses.csv")
    seen_outputs: set[str] = set()
    for target in targets:
        output_name = _target_png_name(target.image_name)
        if output_name in seen_outputs:
            raise DataValidationError(f"Duplicate target output name after PNG conversion: {output_name}")
        seen_outputs.add(output_name)
        source_image = _nearest_train_image(train_centers, target.camera_center)
        _write_resized_png(source_image, output_path / output_name, width=target.width, height=target.height)
    return len(targets)


def _read_train_camera_centers(scene: Path) -> dict[Path, np.ndarray]:
    model = read_colmap_model(scene / "train" / "sparse" / "0")
    available_names = train_image_names(scene)
    image_dir = scene / "train" / "images"
    centers: dict[Path, np.ndarray] = {}
    for image in model.images.values():
        image_name = Path(image.name).name
        if image_name not in available_names:
            continue
        rotation = qvec_to_rotmat(image.qvec)
        tvec = np.asarray(image.tvec, dtype=float)
        center = -rotation.T @ tvec
        centers[image_dir / image_name] = center
    if not centers:
        raise DataValidationError(f"No COLMAP train image poses matched files in {image_dir}")
    return centers


def _read_target_poses(path: Path) -> list[TargetPose]:
    if not path.exists():
        raise DataValidationError(f"test_poses.csv does not exist: {path}")
    targets: list[TargetPose] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in TEST_POSE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise DataValidationError(f"{path} is missing required columns: {', '.join(missing)}")
        for row_index, row in enumerate(reader, start=2):
            image_name = row["image_name"].strip()
            if not image_name:
                raise DataValidationError(f"Missing image_name in test_poses.csv row {row_index}")
            width = _positive_int(row["width"], "width", row_index)
            height = _positive_int(row["height"], "height", row_index)
            targets.append(
                TargetPose(
                    image_name=image_name,
                    camera_center=np.asarray(
                        [
                            _float(row["tx"], "tx", row_index),
                            _float(row["ty"], "ty", row_index),
                            _float(row["tz"], "tz", row_index),
                        ],
                        dtype=float,
                    ),
                    width=width,
                    height=height,
                )
            )
    if not targets:
        raise DataValidationError(f"{path} contains no target poses")
    return targets


def _nearest_train_image(train_centers: dict[Path, np.ndarray], target_center: np.ndarray) -> Path:
    return min(train_centers, key=lambda image_path: float(np.linalg.norm(train_centers[image_path] - target_center)))


def _target_png_name(image_name: str) -> str:
    name = Path(image_name).name
    if Path(name).suffix.lower() == ".png":
        return name
    return f"{Path(name).stem}.png"


def _write_resized_png(source: Path, destination: Path, width: int, height: int) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        rgb.save(destination)


def _float(value: str, key: str, row_index: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise DataValidationError(f"Invalid float for {key} in test_poses.csv row {row_index}: {value}") from exc


def _positive_int(value: str, key: str, row_index: int) -> int:
    try:
        parsed = int(float(value))
    except ValueError as exc:
        raise DataValidationError(f"Invalid integer for {key} in test_poses.csv row {row_index}: {value}") from exc
    if parsed <= 0:
        raise DataValidationError(f"{key} must be positive in test_poses.csv row {row_index}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a low-cost nearest-training-view PNG submission for VAI phase1 scenes."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, e.g. VAI_NVS_DATA/phase1/private_set1.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory containing scene_id/*.png files.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = render_nearest_dataset(root=args.root, output=args.out)
    print(f"Wrote {result.output_dir} with {result.image_count} PNGs from {result.scene_count} scenes")


if __name__ == "__main__":
    main()
