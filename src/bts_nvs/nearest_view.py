from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from bts_nvs.camera import qvec_to_rotmat
from bts_nvs.colmap import read_colmap_model
from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import TEST_POSE_COLUMNS, discover_vai_scenes, find_test_poses_csv, train_image_names

JPEG_SUFFIXES = {".jpg", ".jpeg"}
IMAGE_FORMATS = {"auto", "jpeg", "png"}
NAME_POLICIES = {"exact", "png"}
SELECTION_MODES = {"nearest-pose", "temporal-nearest", "temporal-blend"}
BLEND_WEIGHT_POLICIES = {"linear", "midpoint", "gamma0.5"}
FRAME_INDEX_PATTERN = re.compile(r"_(\d+)_V\.[^.]+$", re.IGNORECASE)


@dataclass(frozen=True)
class NearestViewSubmission:
    output_dir: Path
    scene_count: int
    image_count: int


@dataclass(frozen=True)
class TargetPose:
    image_name: str
    camera_center: np.ndarray
    frame_index: int | None
    width: int
    height: int


@dataclass(frozen=True)
class TrainView:
    image_path: Path
    camera_center: np.ndarray
    frame_index: int | None


def render_nearest_dataset(
    root: Path | str,
    output: Path | str,
    name_policy: str = "exact",
    image_format: str = "auto",
    jpeg_quality: int = 92,
    selection_mode: str = "temporal-blend",
    blend_weight_policy: str = "linear",
) -> NearestViewSubmission:
    root_path = Path(root)
    output_path = Path(output)
    scene_count = 0
    image_count = 0
    for scene in discover_vai_scenes(root_path):
        image_count += render_nearest_scene(
            scene,
            output_path / scene.name,
            name_policy=name_policy,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            selection_mode=selection_mode,
            blend_weight_policy=blend_weight_policy,
        )
        scene_count += 1
    return NearestViewSubmission(output_path, scene_count=scene_count, image_count=image_count)


def render_nearest_scene(
    scene: Path | str,
    output: Path | str,
    name_policy: str = "exact",
    image_format: str = "auto",
    jpeg_quality: int = 92,
    selection_mode: str = "temporal-blend",
    blend_weight_policy: str = "linear",
) -> int:
    _validate_output_options(
        name_policy=name_policy,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        selection_mode=selection_mode,
        blend_weight_policy=blend_weight_policy,
    )
    scene_path = Path(scene)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_image in output_path.iterdir():
        if stale_image.is_file() and stale_image.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            stale_image.unlink()

    train_views = _read_train_views(scene_path)
    targets = _read_target_poses(find_test_poses_csv(scene_path))
    seen_outputs: set[str] = set()
    for target in targets:
        output_name = _target_output_name(target.image_name, name_policy=name_policy)
        if output_name in seen_outputs:
            raise DataValidationError(f"Duplicate target output name after policy conversion: {output_name}")
        seen_outputs.add(output_name)
        _write_selected_prediction(
            train_views,
            target,
            output_path / output_name,
            selection_mode=selection_mode,
            blend_weight_policy=blend_weight_policy,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
    return len(targets)


def _read_train_views(scene: Path) -> list[TrainView]:
    model = read_colmap_model(scene / "train" / "sparse" / "0")
    available_names = train_image_names(scene)
    image_dir = scene / "train" / "images"
    views: list[TrainView] = []
    for image in model.images.values():
        image_name = Path(image.name).name
        if image_name not in available_names:
            continue
        rotation = qvec_to_rotmat(image.qvec)
        tvec = np.asarray(image.tvec, dtype=float)
        center = -rotation.T @ tvec
        views.append(
            TrainView(
                image_path=image_dir / image_name,
                camera_center=center,
                frame_index=_extract_frame_index(image_name),
            )
        )
    if not views:
        raise DataValidationError(f"No COLMAP train image poses matched files in {image_dir}")
    return sorted(views, key=lambda view: (view.frame_index is None, view.frame_index or 0, view.image_path.name))


def _read_target_poses(path: Path) -> list[TargetPose]:
    if not path.exists():
        raise DataValidationError(f"target pose CSV does not exist: {path}")
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
            qvec = np.asarray(
                [_float(row[key], key, row_index) for key in ("qw", "qx", "qy", "qz")],
                dtype=float,
            )
            tvec = np.asarray(
                [_float(row[key], key, row_index) for key in ("tx", "ty", "tz")],
                dtype=float,
            )
            rotation = qvec_to_rotmat(qvec)
            targets.append(
                TargetPose(
                    image_name=image_name,
                    camera_center=-rotation.T @ tvec,
                    frame_index=_extract_frame_index(image_name),
                    width=width,
                    height=height,
                )
            )
    if not targets:
        raise DataValidationError(f"{path} contains no target poses")
    return targets


def _write_selected_prediction(
    train_views: list[TrainView],
    target: TargetPose,
    destination: Path,
    selection_mode: str,
    blend_weight_policy: str,
    image_format: str,
    jpeg_quality: int,
) -> None:
    if selection_mode == "temporal-blend":
        low_view, high_view = _bracketing_train_views(train_views, target)
        if low_view is not None and high_view is not None and low_view.image_path != high_view.image_path:
            _write_blended_images(
                low_view,
                high_view,
                target,
                destination,
                blend_weight_policy=blend_weight_policy,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
            )
            return

    source_view = _select_train_view(train_views, target, selection_mode=selection_mode)
    _write_resized_image(
        source_view.image_path,
        destination,
        width=target.width,
        height=target.height,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def _select_train_view(train_views: list[TrainView], target: TargetPose, selection_mode: str) -> TrainView:
    if selection_mode in {"temporal-nearest", "temporal-blend"} and target.frame_index is not None:
        indexed_views = [view for view in train_views if view.frame_index is not None]
        if indexed_views:
            return min(
                indexed_views,
                key=lambda view: (
                    abs((view.frame_index or 0) - target.frame_index),
                    0 if (view.frame_index or 0) >= target.frame_index else 1,
                    view.image_path.name,
                ),
            )
    return _nearest_pose_train_view(train_views, target.camera_center)


def _bracketing_train_views(
    train_views: list[TrainView],
    target: TargetPose,
) -> tuple[TrainView | None, TrainView | None]:
    if target.frame_index is None:
        return None, None
    indexed_views = [view for view in train_views if view.frame_index is not None]
    if not indexed_views:
        return None, None
    low_candidates = [view for view in indexed_views if (view.frame_index or 0) <= target.frame_index]
    high_candidates = [view for view in indexed_views if (view.frame_index or 0) >= target.frame_index]
    low_view = (
        max(low_candidates, key=lambda view: (view.frame_index or 0, view.image_path.name))
        if low_candidates
        else None
    )
    high_view = (
        min(high_candidates, key=lambda view: (view.frame_index or 0, view.image_path.name))
        if high_candidates
        else None
    )
    if low_view is None:
        low_view = high_view
    if high_view is None:
        high_view = low_view
    return low_view, high_view


def _nearest_pose_train_view(train_views: list[TrainView], target_center: np.ndarray) -> TrainView:
    return min(train_views, key=lambda view: float(np.linalg.norm(view.camera_center - target_center)))


def _extract_frame_index(image_name: str) -> int | None:
    name = Path(image_name).name
    match = FRAME_INDEX_PATTERN.search(name)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", Path(name).stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _target_output_name(image_name: str, name_policy: str) -> str:
    name = Path(image_name).name
    if name_policy == "exact":
        return name
    if Path(name).suffix.lower() == ".png":
        return name
    return f"{Path(name).stem}.png"


def _write_resized_image(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    image_format: str,
    jpeg_quality: int,
) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        resolved_format = _resolve_image_format(destination, image_format=image_format)
        if resolved_format == "jpeg":
            rgb.save(destination, format="JPEG", quality=jpeg_quality, optimize=True, progressive=False)
        elif resolved_format == "png":
            rgb.save(destination, format="PNG", optimize=True, compress_level=9)
        else:
            raise DataValidationError(f"Unsupported output image format: {image_format}")


def _write_blended_images(
    low_view: TrainView,
    high_view: TrainView,
    target: TargetPose,
    destination: Path,
    blend_weight_policy: str,
    image_format: str,
    jpeg_quality: int,
) -> None:
    if low_view.frame_index is None or high_view.frame_index is None or target.frame_index is None:
        raise DataValidationError("Temporal blending requires frame indices for low, high, and target images")
    denominator = high_view.frame_index - low_view.frame_index
    linear_alpha = 0.0 if denominator == 0 else (target.frame_index - low_view.frame_index) / denominator
    alpha = _resolve_blend_alpha(linear_alpha, blend_weight_policy=blend_weight_policy)
    with Image.open(low_view.image_path) as low_image, Image.open(high_view.image_path) as high_image:
        low_rgb = low_image.convert("RGB")
        high_rgb = high_image.convert("RGB")
        target_size = (target.width, target.height)
        if low_rgb.size != target_size:
            low_rgb = low_rgb.resize(target_size, Image.Resampling.LANCZOS)
        if high_rgb.size != target_size:
            high_rgb = high_rgb.resize(target_size, Image.Resampling.LANCZOS)
        blended = Image.blend(low_rgb, high_rgb, alpha=alpha)
        resolved_format = _resolve_image_format(destination, image_format=image_format)
        if resolved_format == "jpeg":
            blended.save(destination, format="JPEG", quality=jpeg_quality, optimize=True, progressive=False)
        elif resolved_format == "png":
            blended.save(destination, format="PNG", optimize=True, compress_level=9)
        else:
            raise DataValidationError(f"Unsupported output image format: {image_format}")


def _resolve_blend_alpha(linear_alpha: float, blend_weight_policy: str) -> float:
    linear_alpha = max(0.0, min(float(linear_alpha), 1.0))
    if blend_weight_policy == "linear":
        return linear_alpha
    if blend_weight_policy == "midpoint":
        return 0.5
    if blend_weight_policy == "gamma0.5":
        low_weight = max(1.0 - linear_alpha, 1e-6) ** 0.5
        high_weight = max(linear_alpha, 1e-6) ** 0.5
        return high_weight / (low_weight + high_weight)
    raise DataValidationError(f"Unsupported blend weight policy: {blend_weight_policy}")


def _resolve_image_format(destination: Path, image_format: str) -> str:
    if image_format != "auto":
        return image_format
    if destination.suffix.lower() in JPEG_SUFFIXES:
        return "jpeg"
    return "png"


def _validate_output_options(
    name_policy: str,
    image_format: str,
    jpeg_quality: int,
    selection_mode: str,
    blend_weight_policy: str,
) -> None:
    if name_policy not in NAME_POLICIES:
        raise DataValidationError(f"Unsupported name policy: {name_policy}")
    if image_format not in IMAGE_FORMATS:
        raise DataValidationError(f"Unsupported image format: {image_format}")
    if selection_mode not in SELECTION_MODES:
        raise DataValidationError(f"Unsupported selection mode: {selection_mode}")
    if blend_weight_policy not in BLEND_WEIGHT_POLICIES:
        raise DataValidationError(f"Unsupported blend weight policy: {blend_weight_policy}")
    if jpeg_quality < 1 or jpeg_quality > 100:
        raise DataValidationError("jpeg_quality must be between 1 and 100")


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
        description="Create a low-cost nearest-training-view image submission for VAI scenes."
    )
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, e.g. VAI_NVS_DATA_ROUND2.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory containing scene_id/* image files.")
    parser.add_argument(
        "--name-policy",
        choices=tuple(sorted(NAME_POLICIES)),
        default="exact",
        help="Use exact image_name from test_poses.csv, or force PNG-style stem names.",
    )
    parser.add_argument(
        "--image-format",
        choices=tuple(sorted(IMAGE_FORMATS)),
        default="auto",
        help="Output encoding. auto writes JPEG for .jpg/.jpeg names and PNG otherwise.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality for JPEG outputs.")
    parser.add_argument(
        "--selection-mode",
        choices=tuple(sorted(SELECTION_MODES)),
        default="temporal-blend",
        help="How to choose train images: pose nearest, temporal nearest, or temporal interpolation.",
    )
    parser.add_argument(
        "--blend-weight-policy",
        choices=tuple(sorted(BLEND_WEIGHT_POLICIES)),
        default="linear",
        help="Weighting used by temporal-blend. linear is the best private-set fallback observed so far.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = render_nearest_dataset(
        root=args.root,
        output=args.out,
        name_policy=args.name_policy,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        selection_mode=args.selection_mode,
        blend_weight_policy=args.blend_weight_policy,
    )
    print(f"Wrote {result.output_dir} with {result.image_count} images from {result.scene_count} scenes")


if __name__ == "__main__":
    main()
