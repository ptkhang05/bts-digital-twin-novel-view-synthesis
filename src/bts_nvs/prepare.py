from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from bts_nvs.colmap import colmap_model_to_nerfstudio, find_colmap_sparse_dir, read_colmap_model, write_ascii_ply
from bts_nvs.exceptions import DataValidationError
from bts_nvs.path_safety import assert_paths_do_not_overlap, assert_tree_has_no_links
from bts_nvs.schema import (
    DISTORTION_KEYS,
    load_json,
    normalized_image_relpath,
    resolve_frame_image,
    validate_transforms,
    write_json,
)
from bts_nvs.vai import find_test_poses_csv, is_vai_scene, test_poses_csv_to_transforms, train_image_names

COPY_MODES = ("copy", "hardlink")
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


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


def validate_preparation_provenance(
    dataset_id: str | None,
    dataset_manifest_sha256: str | None,
) -> tuple[str | None, str | None]:
    """Validate an optional dataset identity pair and normalize its digest."""

    if dataset_id is None and dataset_manifest_sha256 is None:
        return None, None
    if dataset_id is None or dataset_manifest_sha256 is None:
        raise DataValidationError(
            "Preparation provenance requires both dataset_id and dataset manifest SHA-256"
        )
    if _ID_PATTERN.fullmatch(dataset_id) is None:
        raise DataValidationError(
            "dataset_id must be a relative identifier using only letters, digits, '.', '_' or '-'"
        )
    if _SHA256_PATTERN.fullmatch(dataset_manifest_sha256) is None:
        raise DataValidationError("Dataset manifest SHA-256 must contain exactly 64 hexadecimal characters")
    return dataset_id, dataset_manifest_sha256.lower()


def prepare_scene(
    scene: Path | str,
    output: Path | str,
    copy_mode: str = "copy",
    overwrite: bool = False,
    holdout_interval: int = 0,
    dataset_id: str | None = None,
    dataset_manifest_sha256: str | None = None,
) -> PreparedScene:
    scene_path = Path(scene)
    output_path = Path(output)
    dataset_id, dataset_manifest_sha256 = validate_preparation_provenance(
        dataset_id,
        dataset_manifest_sha256,
    )
    _validate_relative_metadata_path(scene_path.name, "source_scene")
    if not scene_path.exists():
        raise DataValidationError(f"Scene directory does not exist: {scene_path}")
    if copy_mode not in COPY_MODES:
        raise DataValidationError(f"Unsupported copy mode: {copy_mode}")
    assert_paths_do_not_overlap(
        scene_path,
        output_path,
        first_label="Scene directory",
        second_label="Prepared output directory",
    )
    assert_tree_has_no_links(scene_path, "Scene directory")
    if output_path.exists() and not output_path.is_dir():
        raise DataValidationError(f"Output path is not a directory: {output_path}")
    if output_path.exists():
        assert_tree_has_no_links(output_path, "Prepared output directory")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise DataValidationError(f"Output directory is not empty: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_format, transforms, point_count = _load_scene_transforms(scene_path, output_path)
    transforms = validate_transforms(transforms, scene=scene_path)
    _copy_images(scene_path, output_path, transforms, copy_mode=copy_mode)
    if holdout_interval:
        _apply_holdout_split(transforms, holdout_interval=holdout_interval)
    transforms = validate_transforms(transforms, scene=output_path)
    target_cameras_path, target_count = _copy_target_cameras(
        scene_path,
        output_path,
        training_transforms=transforms,
    )

    transforms_path = output_path / "transforms.json"
    metadata_path = output_path / "metadata.json"
    write_json(transforms_path, transforms)
    metadata = {
        "source_scene": scene_path.name,
        "source_format": source_format,
        "camera_convention": "nerfstudio-opengl-c2w",
        "image_count": len(transforms["frames"]),
        "point_count": point_count,
        "provenance_status": "verified" if dataset_id is not None else "partial",
        "transforms": "transforms.json",
    }
    if dataset_id is not None and dataset_manifest_sha256 is not None:
        metadata["dataset_id"] = dataset_id
        metadata["dataset_manifest_sha256"] = dataset_manifest_sha256
    if (output_path / "sparse_pc.ply").is_file():
        metadata["sparse_point_cloud"] = "sparse_pc.ply"
    if target_count is not None:
        metadata["target_count"] = target_count
        metadata["target_cameras"] = "target_cameras.json"
    for field in ("source_scene", "transforms", "sparse_point_cloud", "target_cameras"):
        if field in metadata:
            _validate_relative_metadata_path(metadata[field], field)
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
    if is_vai_scene(scene):
        sparse_dir = scene / "train" / "sparse" / "0"
        model = read_colmap_model(sparse_dir)
        transforms, points = colmap_model_to_nerfstudio(scene, model, image_names=train_image_names(scene))
        point_count = _prepare_sparse_point_cloud(
            sparse_dir,
            output,
            points,
            vai_scene_name=scene.name,
        )
        if (output / "sparse_pc.ply").exists():
            transforms["ply_file_path"] = "sparse_pc.ply"
        return "vai", transforms, point_count

    for filename in ("train_cameras.json", "transforms.json"):
        path = scene / filename
        if path.exists():
            return filename, load_json(path), 0

    sparse_dir = find_colmap_sparse_dir(scene)
    if sparse_dir is not None:
        model = read_colmap_model(sparse_dir)
        transforms, points = colmap_model_to_nerfstudio(scene, model)
        point_count = _prepare_sparse_point_cloud(sparse_dir, output, points)
        if (output / "sparse_pc.ply").exists():
            transforms["ply_file_path"] = "sparse_pc.ply"
        return "colmap", transforms, point_count

    raise DataValidationError(
        "Could not detect scene input. Expected train_cameras.json, transforms.json, or COLMAP sparse files."
    )


def _prepare_sparse_point_cloud(
    sparse_dir: Path,
    output: Path,
    points: list,
    *,
    vai_scene_name: str | None = None,
) -> int:
    destination = output / "sparse_pc.ply"
    source = sparse_dir / "points3D.ply"
    if source.is_file():
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
        return len(points) or _ply_vertex_count(source)
    if points and vai_scene_name is not None and vai_scene_name != "chair":
        raise DataValidationError(
            f"VAI scene {vai_scene_name!r} is missing points3D.ply; only scene 'chair' may synthesize it from BIN"
        )
    if points:
        return write_ascii_ply(points, destination)
    return 0


def _validate_relative_metadata_path(value: object, field: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise DataValidationError(f"Metadata field {field} must be a non-empty relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise DataValidationError(f"Metadata field {field} must be a relative path: {value!r}")


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for _ in range(100):
            raw = handle.readline()
            if not raw:
                break
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise DataValidationError(f"Invalid PLY header in {path}") from exc
            if line.startswith("element vertex "):
                try:
                    return int(line.split()[-1])
                except ValueError as exc:
                    raise DataValidationError(f"Invalid PLY vertex count in {path}: {line}") from exc
            if line == "end_header":
                break
    raise DataValidationError(f"PLY header does not declare an element vertex count: {path}")


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
        else:
            raise DataValidationError(f"Unsupported copy mode: {copy_mode}")
        frame["file_path"] = relpath.as_posix()


def _copy_target_cameras(
    scene: Path,
    output: Path,
    training_transforms: dict,
) -> tuple[Path | None, int | None]:
    target_path = scene / "target_cameras.json"
    if target_path.exists():
        targets = validate_transforms(load_json(target_path))
    elif is_vai_scene(scene):
        targets = validate_transforms(test_poses_csv_to_transforms(find_test_poses_csv(scene)))
        for key in DISTORTION_KEYS:
            if key in training_transforms:
                targets[key] = training_transforms[key]
    else:
        return None, None
    target_count = len(targets["frames"])
    output_path = output / "target_cameras.json"
    write_json(output_path, targets)
    return output_path, target_count


def _apply_holdout_split(transforms: dict, holdout_interval: int) -> None:
    if holdout_interval <= 1:
        raise DataValidationError("holdout_interval must be greater than 1")
    train_filenames: list[str] = []
    eval_filenames: list[str] = []
    filenames = sorted(frame["file_path"] for frame in transforms["frames"])
    for index, filename in enumerate(filenames):
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
    parser.add_argument("--copy-mode", choices=COPY_MODES, default="copy")
    parser.add_argument("--dataset-id", help="Logical dataset ID stored in metadata.json.")
    parser.add_argument(
        "--manifest-sha256",
        help="Verified dataset manifest overall SHA-256 stored in metadata.json.",
    )
    parser.add_argument(
        "--holdout-interval",
        type=int,
        default=0,
        help="Optional filename split: every Nth frame becomes val/test, the rest train.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_scene(
        args.scene,
        args.out,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        holdout_interval=args.holdout_interval,
        dataset_id=args.dataset_id,
        dataset_manifest_sha256=args.manifest_sha256,
    )
    print(f"Wrote {result.transforms_path} with {result.image_count} frames")


if __name__ == "__main__":
    main()
