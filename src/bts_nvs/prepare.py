from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.colmap import colmap_model_to_nerfstudio, find_colmap_sparse_dir, read_colmap_model, write_ascii_ply
from bts_nvs.contest import DEFAULT_CONTEST_PHASE, validate_target_view_count, validate_training_image_count
from bts_nvs.exceptions import DataValidationError
from bts_nvs.schema import (
    load_json,
    normalized_image_relpath,
    resolve_frame_image,
    validate_transforms,
    write_json,
)
from bts_nvs.vai import find_test_poses_csv, is_vai_phase1_scene, test_poses_csv_to_transforms, train_image_names


@dataclass(frozen=True)
class PreparedScene:
    output_dir: Path
    transforms_path: Path
    metadata_path: Path
    image_count: int
    source_format: str
    target_cameras_path: Path | None = None
    target_count: int | None = None
    point_count: int = 0


def prepare_scene(
    scene: Path | str,
    output: Path | str,
    copy_mode: str = "copy",
    overwrite: bool = False,
    holdout_interval: int = 0,
    strict_contest: bool = False,
    contest_phase: str = DEFAULT_CONTEST_PHASE,
) -> PreparedScene:
    scene_path = Path(scene)
    output_path = Path(output)
    if not scene_path.exists():
        raise DataValidationError(f"Scene directory does not exist: {scene_path}")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise DataValidationError(f"Output directory is not empty: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_format, transforms, point_count = _load_scene_transforms(scene_path, output_path)
    transforms = validate_transforms(transforms, scene=scene_path)
    if strict_contest:
        validate_training_image_count(len(transforms["frames"]), phase=contest_phase)
    _copy_images(scene_path, output_path, transforms, copy_mode=copy_mode)
    if holdout_interval:
        _apply_holdout_split(transforms, holdout_interval=holdout_interval)
    transforms = validate_transforms(transforms, scene=output_path)
    target_cameras_path, target_count = _copy_target_cameras(
        scene_path,
        output_path,
        strict_contest=strict_contest,
        contest_phase=contest_phase,
    )

    transforms_path = output_path / "transforms.json"
    metadata_path = output_path / "metadata.json"
    write_json(transforms_path, transforms)
    metadata = {
        "source_scene": str(scene_path.resolve()),
        "source_format": source_format,
        "camera_convention": "nerfstudio-opengl-c2w",
        "image_count": len(transforms["frames"]),
        "point_count": point_count,
    }
    if target_count is not None:
        metadata["target_count"] = target_count
        metadata["target_cameras"] = "target_cameras.json"
    write_json(metadata_path, metadata)
    return PreparedScene(
        output_dir=output_path,
        transforms_path=transforms_path,
        metadata_path=metadata_path,
        image_count=len(transforms["frames"]),
        source_format=source_format,
        target_cameras_path=target_cameras_path,
        target_count=target_count,
        point_count=point_count,
    )


def _load_scene_transforms(scene: Path, output: Path) -> tuple[str, dict, int]:
    if is_vai_phase1_scene(scene):
        model = read_colmap_model(scene / "train" / "sparse" / "0")
        transforms, points = colmap_model_to_nerfstudio(scene, model, image_names=train_image_names(scene))
        if points:
            ply_path = output / "sparse_pc.ply"
            point_count = write_ascii_ply(points, ply_path)
            transforms["ply_file_path"] = "sparse_pc.ply"
        else:
            point_count = 0
        return "vai_phase1", transforms, point_count

    for filename in ("train_cameras.json", "transforms.json"):
        path = scene / filename
        if path.exists():
            return filename, load_json(path), 0

    sparse_dir = find_colmap_sparse_dir(scene)
    if sparse_dir is not None:
        model = read_colmap_model(sparse_dir)
        transforms, points = colmap_model_to_nerfstudio(scene, model)
        if points:
            ply_path = output / "sparse_pc.ply"
            point_count = write_ascii_ply(points, ply_path)
            transforms["ply_file_path"] = "sparse_pc.ply"
        else:
            point_count = 0
        return "colmap", transforms, point_count

    raise DataValidationError(
        "Could not detect scene input. Expected train_cameras.json, transforms.json, or COLMAP sparse files."
    )


def _copy_images(scene: Path, output: Path, transforms: dict, copy_mode: str) -> None:
    seen: set[str] = set()
    for frame in transforms["frames"]:
        source = resolve_frame_image(scene, frame["file_path"])
        relpath = normalized_image_relpath(frame["file_path"])
        if relpath.as_posix() in seen:
            raise DataValidationError(f"Duplicate output image path: {relpath.as_posix()}")
        seen.add(relpath.as_posix())
        destination = output / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "copy":
            shutil.copy2(source, destination)
        elif copy_mode == "hardlink":
            if destination.exists():
                destination.unlink()
            destination.hardlink_to(source)
        elif copy_mode == "symlink":
            if destination.exists():
                destination.unlink()
            destination.symlink_to(source.resolve())
        else:
            raise DataValidationError(f"Unsupported copy mode: {copy_mode}")
        frame["file_path"] = relpath.as_posix()


def _copy_target_cameras(
    scene: Path,
    output: Path,
    strict_contest: bool,
    contest_phase: str,
) -> tuple[Path | None, int | None]:
    target_path = scene / "target_cameras.json"
    if target_path.exists():
        targets = validate_transforms(load_json(target_path))
    elif is_vai_phase1_scene(scene):
        targets = validate_transforms(test_poses_csv_to_transforms(find_test_poses_csv(scene)))
    else:
        return None, None
    target_count = len(targets["frames"])
    if strict_contest:
        validate_target_view_count(target_count, phase=contest_phase)
    output_path = output / "target_cameras.json"
    write_json(output_path, targets)
    return output_path, target_count


def _apply_holdout_split(transforms: dict, holdout_interval: int) -> None:
    if holdout_interval <= 1:
        raise DataValidationError("holdout_interval must be greater than 1")
    train_filenames: list[str] = []
    eval_filenames: list[str] = []
    for index, frame in enumerate(transforms["frames"]):
        filename = frame["file_path"]
        if (index + 1) % holdout_interval == 0:
            eval_filenames.append(filename)
        else:
            train_filenames.append(filename)
    if not eval_filenames:
        raise DataValidationError("holdout_interval did not select any eval frames")
    if not train_filenames:
        raise DataValidationError("holdout_interval selected every frame for eval")
    transforms["train_filenames"] = train_filenames
    transforms["val_filenames"] = eval_filenames
    transforms["test_filenames"] = eval_filenames


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a BTS NVS scene for Nerfstudio.")
    parser.add_argument("--scene", type=Path, required=True, help="Raw scene directory.")
    parser.add_argument("--out", type=Path, required=True, help="Processed scene output directory.")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument(
        "--holdout-interval",
        type=int,
        default=0,
        help="Optional filename split: every Nth frame becomes val/test, the rest train.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    parser.add_argument(
        "--strict-contest",
        action="store_true",
        help="Enforce the selected Viettel AI Race rule set.",
    )
    parser.add_argument(
        "--contest-phase",
        default=DEFAULT_CONTEST_PHASE,
        help="Contest rule set for --strict-contest. Known values: phase1, overview.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_scene(
        args.scene,
        args.out,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        holdout_interval=args.holdout_interval,
        strict_contest=args.strict_contest,
        contest_phase=args.contest_phase,
    )
    print(f"Wrote {result.transforms_path} with {result.image_count} frames")


if __name__ == "__main__":
    main()
