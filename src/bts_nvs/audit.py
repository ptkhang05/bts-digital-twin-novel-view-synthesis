from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from bts_nvs.colmap import camera_to_nerfstudio_intrinsics, read_colmap_model
from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import TEST_POSE_COLUMNS, discover_vai_scenes, find_test_poses_csv

_JPEG_SUFFIXES = {".jpg", ".jpeg"}
_FLOAT_TOLERANCE = 1e-6
_QUATERNION_NORM_TOLERANCE = 1e-4
_HASH_CHUNK_SIZE = 1024 * 1024
_CHECKED_MANIFEST_FIELDS = (
    "manifest_version",
    "digest_algorithm",
    "overall_sha256",
    "file_count",
    "total_bytes",
    "scene_count",
    "train_image_count",
    "target_image_count",
    "scenes",
    "files",
)


def audit_dataset(data_root: Path | str) -> dict[str, Any]:
    """Validate a VAI dataset and return a deterministic content manifest."""

    root = Path(data_root)
    _require_safe_root(root)
    scene_summaries = [_audit_scene(scene) for scene in discover_vai_scenes(root)]
    file_entries = [_hash_file(root, path) for path in _iter_regular_files(root)]

    overall = hashlib.sha256()
    for entry in file_entries:
        overall.update(f'{entry["path"]}\0{entry["size"]}\0{entry["sha256"]}\n'.encode())

    return {
        "manifest_version": 1,
        "dataset_id": root.name,
        "digest_algorithm": "sha256(path\\0size\\0sha256\\n), sorted by POSIX path",
        "overall_sha256": overall.hexdigest(),
        "file_count": len(file_entries),
        "total_bytes": sum(entry["size"] for entry in file_entries),
        "scene_count": len(scene_summaries),
        "train_image_count": sum(scene["train_image_count"] for scene in scene_summaries),
        "target_image_count": sum(scene["target_image_count"] for scene in scene_summaries),
        "scenes": scene_summaries,
        "files": file_entries,
    }


def write_audit_manifest(data_root: Path | str, manifest_path: Path | str) -> dict[str, Any]:
    root = Path(data_root)
    destination = Path(manifest_path)
    _require_safe_root(root)
    _require_manifest_outside_root(root, destination)

    manifest = audit_dataset(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest


def check_audit_manifest(data_root: Path | str, manifest_path: Path | str) -> dict[str, Any]:
    """Audit current files and compare them with a manifest without rewriting it."""

    root = Path(data_root)
    expected_path = Path(manifest_path)
    _require_safe_root(root)
    _require_manifest_outside_root(root, expected_path)
    try:
        with expected_path.open("r", encoding="utf-8") as handle:
            expected = json.load(handle)
    except FileNotFoundError as exc:
        raise DataValidationError(f"Audit manifest does not exist: {expected_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Audit manifest is not readable JSON: {expected_path} ({exc})") from exc
    if not isinstance(expected, dict):
        raise DataValidationError(f"Audit manifest must contain a JSON object: {expected_path}")

    actual = audit_dataset(root)
    mismatches = [field for field in _CHECKED_MANIFEST_FIELDS if expected.get(field) != actual.get(field)]
    if mismatches:
        raise DataValidationError(f"Audit manifest mismatch for {', '.join(mismatches)}: {expected_path}")
    return actual


def _audit_scene(scene: Path) -> dict[str, Any]:
    image_dir = scene / "train" / "images"
    sparse_dir = scene / "train" / "sparse" / "0"
    csv_path = find_test_poses_csv(scene)
    _reject_link(image_dir, "train image directory")
    _reject_link(sparse_dir, "COLMAP sparse directory")
    _reject_link(csv_path, "target pose CSV")

    image_paths = _flat_regular_files(image_dir)
    if not image_paths:
        raise DataValidationError(f"VAI scene contains no train images: {image_dir}")
    train_dimensions: dict[str, tuple[int, int]] = {}
    casefolded_train_names: set[str] = set()
    for path in image_paths:
        _validate_basename(path.name, context=f"train image in {scene.name}")
        if path.suffix.lower() not in _JPEG_SUFFIXES:
            raise DataValidationError(f"Train image must be JPEG: {path}")
        folded = path.name.casefold()
        if folded in casefolded_train_names:
            raise DataValidationError(f"Train image names collide by case in {scene}: {path.name}")
        casefolded_train_names.add(folded)
        train_dimensions[path.name] = _read_jpeg_dimensions(path)

    model = read_colmap_model(sparse_dir)
    registered_by_name: dict[str, Any] = {}
    registered_casefolded: set[str] = set()
    for image in model.images.values():
        name = _colmap_basename(image.name, scene)
        if name in registered_by_name or name.casefold() in registered_casefolded:
            raise DataValidationError(f"COLMAP image names collide by basename/case in {scene}: {name}")
        registered_by_name[name] = image
        registered_casefolded.add(name.casefold())
        if image.camera_id not in model.cameras:
            raise DataValidationError(f"COLMAP image {name} references missing camera {image.camera_id}")
        _require_normalized_quaternion(image.qvec, f"COLMAP image {scene.name}/{name}")
        _require_finite(image.tvec, f"COLMAP translation {scene.name}/{name}")

    missing_train = sorted(set(train_dimensions) - set(registered_by_name))
    if missing_train:
        raise DataValidationError(
            f"Train images are not registered in COLMAP for {scene.name}: {', '.join(missing_train)}"
        )
    for name, dimensions in train_dimensions.items():
        image = registered_by_name[name]
        camera = model.cameras[image.camera_id]
        if dimensions != (camera.width, camera.height):
            raise DataValidationError(
                f"Train image dimensions disagree with COLMAP for {scene.name}/{name}: "
                f"file={dimensions}, COLMAP={(camera.width, camera.height)}"
            )

    targets = _read_target_rows(csv_path)
    train_names = set(train_dimensions)
    for target in targets:
        name = target["image_name"]
        if name in train_names or name.casefold() in casefolded_train_names:
            raise DataValidationError(f"Target image overlaps a train image in {scene.name}: {name}")
        if name not in registered_by_name:
            raise DataValidationError(f"Target image is not registered in COLMAP for {scene.name}: {name}")
        registered = registered_by_name[name]
        camera = model.cameras[registered.camera_id]
        _validate_target_against_colmap(scene.name, target, registered, camera)

    target_names = {target["image_name"] for target in targets}
    extra_registered = set(registered_by_name) - train_names - target_names
    return {
        "scene_id": scene.name,
        "train_image_count": len(train_dimensions),
        "target_image_count": len(targets),
        "colmap_registered_image_count": len(registered_by_name),
        "colmap_extra_registered_image_count": len(extra_registered),
        "camera_models": sorted({camera.model for camera in model.cameras.values()}),
    }


def _read_target_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    folded_names: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TEST_POSE_COLUMNS:
            raise DataValidationError(
                f"{csv_path} must contain exactly these columns in order: {', '.join(TEST_POSE_COLUMNS)}"
            )
        for row_index, raw in enumerate(reader, start=2):
            name = raw["image_name"]
            _validate_basename(name, context=f"unsafe image_name in {csv_path} row {row_index}")
            if Path(name).suffix.lower() not in _JPEG_SUFFIXES:
                raise DataValidationError(
                    f"Target image_name must end in .jpg or .jpeg in {csv_path} row {row_index}: {name}"
                )
            if name in names or name.casefold() in folded_names:
                raise DataValidationError(f"Duplicate target image_name in {csv_path}: {name}")
            names.add(name)
            folded_names.add(name.casefold())

            values = {key: _finite_float(raw[key], key, csv_path, row_index) for key in TEST_POSE_COLUMNS[1:12]}
            values["width"] = _positive_integer(raw["width"], "width", csv_path, row_index)
            values["height"] = _positive_integer(raw["height"], "height", csv_path, row_index)
            if values["fx"] <= 0 or values["fy"] <= 0:
                raise DataValidationError(f"fx and fy must be positive in {csv_path} row {row_index}")
            _require_normalized_quaternion(
                tuple(values[key] for key in ("qw", "qx", "qy", "qz")),
                f"target pose {csv_path} row {row_index}",
            )
            rows.append({"image_name": name, **values})
    if not rows:
        raise DataValidationError(f"{csv_path} contains no target poses")
    return rows


def _validate_target_against_colmap(scene_name: str, target: dict[str, Any], registered: Any, camera: Any) -> None:
    name = target["image_name"]
    target_qvec = tuple(target[key] for key in ("qw", "qx", "qy", "qz"))
    target_tvec = tuple(target[key] for key in ("tx", "ty", "tz"))
    q_matches = _all_close(target_qvec, registered.qvec) or _all_close(
        target_qvec,
        (-value for value in registered.qvec),
    )
    if not q_matches or not _all_close(target_tvec, registered.tvec):
        raise DataValidationError(f"Target pose does not match COLMAP for {scene_name}/{name}")

    intrinsics = camera_to_nerfstudio_intrinsics(camera)
    expected = {
        "fx": float(intrinsics["fl_x"]),
        "fy": float(intrinsics["fl_y"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
    }
    if any(
        not math.isclose(target[key], value, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE)
        for key, value in expected.items()
    ):
        raise DataValidationError(f"Target intrinsics do not match COLMAP for {scene_name}/{name}")
    if (target["width"], target["height"]) != (camera.width, camera.height):
        raise DataValidationError(f"Target dimensions do not match COLMAP for {scene_name}/{name}")


def _require_safe_root(root: Path) -> None:
    if not root.exists() or not root.is_dir():
        raise DataValidationError(f"Dataset root does not exist or is not a directory: {root}")
    _reject_link(root, "dataset root")


def _require_manifest_outside_root(root: Path, manifest_path: Path) -> None:
    root_resolved = root.resolve(strict=True)
    manifest_resolved = manifest_path.resolve(strict=False)
    if manifest_resolved == root_resolved or manifest_resolved.is_relative_to(root_resolved):
        raise DataValidationError("Audit manifest must be outside the data root")


def _iter_regular_files(root: Path) -> Iterable[Path]:
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for directory_name in directory_names:
            _reject_link(current_path / directory_name, "dataset directory")
        for file_name in file_names:
            path = current_path / file_name
            _reject_link(path, "dataset file")
            if not path.is_file():
                raise DataValidationError(f"Dataset entry is not a regular file: {path}")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root.resolve(strict=True)):
                raise DataValidationError(f"Dataset file escapes the data root: {path}")
            yield path


def _flat_regular_files(directory: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        _reject_link(path, "dataset entry")
        if not path.is_file():
            raise DataValidationError(f"Nested or non-file entry is not allowed in {directory}: {path.name}")
        paths.append(path)
    return paths


def _hash_file(root: Path, path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    after = path.stat()
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state:
        raise DataValidationError(f"Dataset file changed while it was being hashed: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "size": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _read_jpeg_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            orientation = image.getexif().get(274, 1)
            if orientation != 1:
                raise DataValidationError(
                    f"Train image EXIF Orientation must be absent or 1: {path} (got {orientation})"
                )
            image.verify()
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise DataValidationError(f"Train image is not a readable JPEG: {path}")
            if image.mode != "RGB":
                raise DataValidationError(f"Train image must be RGB JPEG: {path} (got {image.mode})")
            return image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise DataValidationError(f"Train image is not a readable JPEG: {path} ({exc})") from exc


def _validate_basename(name: str, *, context: str) -> None:
    if (
        not name
        or name != name.strip()
        or "\x00" in name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or PureWindowsPath(name).drive
    ):
        raise DataValidationError(f"{context}: {name!r} is not a safe basename")


def _colmap_basename(name: str, scene: Path) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise DataValidationError(f"Unsafe COLMAP image path in {scene}: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataValidationError(f"Unsafe COLMAP image path in {scene}: {name!r}")
    _validate_basename(path.name, context=f"unsafe COLMAP basename in {scene}")
    return path.name


def _finite_float(value: str | None, column: str, path: Path, row_index: int) -> float:
    try:
        parsed = float(value) if value is not None else math.nan
    except ValueError as exc:
        raise DataValidationError(f"Invalid float for {column} in {path} row {row_index}: {value}") from exc
    if not math.isfinite(parsed):
        raise DataValidationError(f"Non-finite value for {column} in {path} row {row_index}: {value}")
    return parsed


def _positive_integer(value: str | None, column: str, path: Path, row_index: int) -> int:
    parsed = _finite_float(value, column, path, row_index)
    if not parsed.is_integer() or parsed <= 0:
        raise DataValidationError(f"{column} must be a positive integer in {path} row {row_index}: {value}")
    return int(parsed)


def _require_normalized_quaternion(values: Iterable[float], context: str) -> None:
    vector = tuple(values)
    _require_finite(vector, context)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_QUATERNION_NORM_TOLERANCE):
        raise DataValidationError(f"Quaternion is not normalized for {context}: norm={norm}")


def _require_finite(values: Iterable[float], context: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise DataValidationError(f"Non-finite numeric value in {context}")


def _all_close(left: Iterable[float], right: Iterable[float]) -> bool:
    return all(
        math.isclose(a, b, rel_tol=_FLOAT_TOLERANCE, abs_tol=_FLOAT_TOLERANCE)
        for a, b in zip(tuple(left), tuple(right), strict=True)
    )


def _reject_link(path: Path, label: str) -> None:
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or (callable(is_junction) and is_junction()):
        raise DataValidationError(f"{label} must not be a symlink or junction: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit VAI data and write a deterministic SHA-256 manifest.")
    parser.add_argument("--data-root", type=Path, required=True, help="VAI dataset root containing scene folders.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Write deterministic manifest JSON outside the data root.")
    mode.add_argument("--check-manifest", type=Path, help="Read-only verification against an existing manifest JSON.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.check_manifest is not None:
        manifest = check_audit_manifest(args.data_root, args.check_manifest)
        action = "Verified"
    else:
        manifest = write_audit_manifest(args.data_root, args.manifest)
        action = "Audited"
    print(
        f'{action} {manifest["scene_count"]} scenes, {manifest["train_image_count"]} train images, '
        f'{manifest["target_image_count"]} targets; sha256={manifest["overall_sha256"]}'
    )


if __name__ == "__main__":
    main()
