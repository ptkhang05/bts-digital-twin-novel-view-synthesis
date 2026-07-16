from __future__ import annotations

import argparse
import copy
import shutil
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.exceptions import DataValidationError
from bts_nvs.path_safety import (
    assert_output_separate_from,
    assert_path_has_no_links,
    assert_paths_do_not_overlap,
    assert_tree_has_no_links,
)
from bts_nvs.prepare import prepare_scene
from bts_nvs.schema import load_json, validate_transforms, write_json
from bts_nvs.vai import discover_vai_scenes

HOLDOUT_CAMERAS_FILENAME = "holdout_cameras.json"


@dataclass(frozen=True)
class PreparedHoldoutScene:
    scene_id: str
    processed_dir: Path
    cameras_path: Path
    ground_truth_images_dir: Path
    target_count: int


def prepare_holdout(
    *,
    data_root: Path | str,
    processed_root: Path | str,
    ground_truth_root: Path | str,
    interval: int = 10,
    copy_mode: str = "hardlink",
    overwrite: bool = False,
) -> list[PreparedHoldoutScene]:
    """Prepare deterministic filename holdouts plus a strict scoring ground-truth tree."""

    _validate_interval(interval)
    if copy_mode not in {"copy", "hardlink"}:
        raise DataValidationError("Holdout copy_mode must be 'copy' or 'hardlink'")

    data_path, processed_path, ground_truth_path = _validate_output_roots(
        Path(data_root),
        Path(processed_root),
        Path(ground_truth_root),
    )
    scenes = discover_vai_scenes(data_path)
    _initialize_output_root(processed_path, overwrite=overwrite)
    _initialize_output_root(ground_truth_path, overwrite=overwrite)

    results: list[PreparedHoldoutScene] = []
    for scene in scenes:
        processed_scene = processed_path / scene.name
        prepare_scene(scene=scene, output=processed_scene, copy_mode=copy_mode)
        transforms_path = processed_scene / "transforms.json"
        transforms = validate_transforms(load_json(transforms_path), scene=processed_scene)
        train_filenames, holdout_filenames = _filename_split(transforms, interval=interval)
        transforms["train_filenames"] = train_filenames
        transforms["val_filenames"] = holdout_filenames
        transforms["test_filenames"] = holdout_filenames
        write_json(transforms_path, transforms)

        cameras_path = processed_scene / HOLDOUT_CAMERAS_FILENAME
        write_json(cameras_path, _holdout_cameras(transforms, holdout_filenames))
        ground_truth_images = ground_truth_path / scene.name / "test" / "images"
        _copy_ground_truth_images(
            processed_scene=processed_scene,
            filenames=holdout_filenames,
            destination=ground_truth_images,
            copy_mode=copy_mode,
        )
        _update_metadata(processed_scene / "metadata.json", interval=interval, target_count=len(holdout_filenames))
        results.append(
            PreparedHoldoutScene(
                scene_id=scene.name,
                processed_dir=processed_scene,
                cameras_path=cameras_path,
                ground_truth_images_dir=ground_truth_images,
                target_count=len(holdout_filenames),
            )
        )
    return results


def _validate_interval(interval: int) -> None:
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 1:
        raise DataValidationError("Holdout interval must be an integer greater than 1")


def _validate_output_roots(
    data_root: Path,
    processed_root: Path,
    ground_truth_root: Path,
) -> tuple[Path, Path, Path]:
    data_absolute = assert_tree_has_no_links(data_root, "Raw dataset root")
    processed_absolute = assert_output_separate_from(
        processed_root,
        ((data_absolute, "raw dataset root"),),
        output_label="Processed holdout root",
    )
    ground_truth_absolute = assert_output_separate_from(
        ground_truth_root,
        ((data_absolute, "raw dataset root"),),
        output_label="Holdout ground-truth root",
    )
    assert_paths_do_not_overlap(
        processed_absolute,
        ground_truth_absolute,
        first_label="Processed holdout root",
        second_label="Holdout ground-truth root",
    )
    return data_absolute, processed_absolute, ground_truth_absolute


def _initialize_output_root(path: Path, *, overwrite: bool) -> None:
    path = assert_path_has_no_links(path, "Holdout output root")
    if path.exists() and not path.is_dir():
        raise DataValidationError(f"Output root exists and is not a directory: {path}")
    if path.exists():
        assert_tree_has_no_links(path, "Holdout output root")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise DataValidationError(f"Output root is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _filename_split(transforms: dict, *, interval: int) -> tuple[list[str], list[str]]:
    filenames = sorted(str(frame["file_path"]) for frame in transforms["frames"])
    if len(filenames) != len(set(filenames)):
        raise DataValidationError("Training transforms contain duplicate file paths")
    holdout = [filename for index, filename in enumerate(filenames) if (index + 1) % interval == 0]
    training = [filename for index, filename in enumerate(filenames) if (index + 1) % interval != 0]
    if not holdout:
        raise DataValidationError("Holdout interval did not select any evaluation frames")
    if not training:
        raise DataValidationError("Holdout interval selected every frame for evaluation")
    return training, holdout


def _holdout_cameras(transforms: dict, filenames: list[str]) -> dict:
    frames_by_path = {str(frame["file_path"]): frame for frame in transforms["frames"]}
    target_frames = []
    target_names: set[str] = set()
    for filename in filenames:
        frame = copy.deepcopy(frames_by_path[filename])
        target_name = Path(filename).name
        if target_name in target_names:
            raise DataValidationError(f"Holdout target basename is duplicated: {target_name}")
        target_names.add(target_name)
        frame["file_path"] = target_name
        target_frames.append(frame)

    excluded = {"frames", "train_filenames", "val_filenames", "test_filenames", "ply_file_path"}
    targets = {key: copy.deepcopy(value) for key, value in transforms.items() if key not in excluded}
    targets["frames"] = target_frames
    return validate_transforms(targets)


def _copy_ground_truth_images(
    *,
    processed_scene: Path,
    filenames: list[str],
    destination: Path,
    copy_mode: str,
) -> None:
    destination.mkdir(parents=True)
    for filename in filenames:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise DataValidationError(f"Unsafe holdout frame path: {filename}")
        source = processed_scene / relative
        if not source.is_file() or source.is_symlink():
            raise DataValidationError(f"Holdout source image is missing or unsafe: {filename}")
        output = destination / relative.name
        if copy_mode == "hardlink":
            output.hardlink_to(source)
        else:
            shutil.copy2(source, output)


def _update_metadata(path: Path, *, interval: int, target_count: int) -> None:
    metadata = load_json(path)
    metadata.update(
        {
            "holdout_cameras": HOLDOUT_CAMERAS_FILENAME,
            "holdout_interval": interval,
            "holdout_target_count": target_count,
        }
    )
    write_json(path, metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare deterministic VAI holdouts for render-and-score evaluation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--ground-truth-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = prepare_holdout(
        data_root=args.data_root,
        processed_root=args.processed_root,
        ground_truth_root=args.ground_truth_root,
        interval=args.interval,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
    )
    print(f"Prepared {sum(result.target_count for result in results)} holdout frames across {len(results)} scenes")


if __name__ == "__main__":
    main()
