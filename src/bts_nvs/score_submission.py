from __future__ import annotations

import argparse
import json
import math
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from bts_nvs.evaluate import DEFAULT_PSNR_MAX, IMAGE_SUFFIXES, evaluate_directories
from bts_nvs.exceptions import DataValidationError

LOCAL_PROXY_SCORE_LABEL = "local_proxy_not_official_btc_score"
MAX_ZIP_MEMBER_COUNT = 10_000
MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
ZIP_COPY_CHUNK_BYTES = 1024 * 1024
WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


@dataclass(frozen=True)
class _ImageSpec:
    path: Path
    size: tuple[int, int]


@dataclass(frozen=True)
class _EvaluationScene:
    name: str
    ground_truth_dir: Path
    images_by_key: dict[str, _ImageSpec]


def score_submission(
    data_root: Path | str,
    submission: Path | str,
    match_by_stem: bool = False,
    psnr_max: float = DEFAULT_PSNR_MAX,
) -> dict[str, Any]:
    _validate_psnr_max(psnr_max)
    evaluation_scenes = _load_evaluation_scenes(Path(data_root), match_by_stem=match_by_stem)
    submission_path = Path(submission)
    if submission_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="bts_nvs_submission_") as temp_dir:
            expected_count = sum(len(scene.images_by_key) for scene in evaluation_scenes)
            _extract_zip_submission(submission_path, Path(temp_dir), expected_member_count=expected_count)
            return _score_submission_folder(
                evaluation_scenes=evaluation_scenes,
                submission_dir=Path(temp_dir),
                match_by_stem=match_by_stem,
                psnr_max=psnr_max,
            )
    return _score_submission_folder(
        evaluation_scenes=evaluation_scenes,
        submission_dir=submission_path,
        match_by_stem=match_by_stem,
        psnr_max=psnr_max,
    )


def _score_submission_folder(
    evaluation_scenes: list[_EvaluationScene],
    submission_dir: Path,
    match_by_stem: bool,
    psnr_max: float,
) -> dict[str, Any]:
    prediction_dirs = _validate_submission_folder(
        submission_dir,
        evaluation_scenes=evaluation_scenes,
        match_by_stem=match_by_stem,
    )

    scene_results: list[dict[str, Any]] = []
    for scene in evaluation_scenes:
        metrics = evaluate_directories(
            prediction_dirs[scene.name],
            scene.ground_truth_dir,
            match_by_stem=match_by_stem,
            psnr_max=psnr_max,
        )
        scene_results.append({"scene": scene.name, **metrics})

    return {
        "aggregate": _aggregate_metrics(scene_results),
        "scenes": scene_results,
        "scoring": {
            "label": LOCAL_PROXY_SCORE_LABEL,
            "psnr_max": float(psnr_max),
            "psnr_max_is_official": False,
            "scene_aggregation": "equal_mean",
        },
    }


def _aggregate_metrics(scene_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not scene_results:
        raise DataValidationError("No scene metrics were computed")

    total_count = sum(int(scene["count"]) for scene in scene_results)
    if total_count <= 0:
        raise DataValidationError("No image pairs were scored")

    aggregate: dict[str, Any] = {"count": total_count}
    metric_keys = set.intersection(*(set(scene) for scene in scene_results)) - {"scene", "count"}
    for key in sorted(metric_keys):
        values = [scene[key] for scene in scene_results]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            aggregate[key] = float(sum(float(value) for value in values) / len(values))
    return aggregate


def _validate_psnr_max(psnr_max: float) -> None:
    if isinstance(psnr_max, bool) or not isinstance(psnr_max, (int, float)):
        raise DataValidationError("psnr_max must be a finite positive number")
    if not math.isfinite(float(psnr_max)) or float(psnr_max) <= 0:
        raise DataValidationError("psnr_max must be a finite positive number")


def _load_evaluation_scenes(data_root: Path, *, match_by_stem: bool) -> list[_EvaluationScene]:
    scene_dirs = _discover_evaluation_scene_dirs(data_root)
    scenes = []
    for scene_dir in scene_dirs:
        images_dir = _evaluation_images_dir(scene_dir)
        scenes.append(
            _EvaluationScene(
                name=scene_dir.name,
                ground_truth_dir=images_dir,
                images_by_key=_load_image_specs(
                    images_dir,
                    match_by_stem=match_by_stem,
                    label=f"ground-truth scene {scene_dir.name}",
                ),
            )
        )
    return scenes


def _discover_evaluation_scene_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        raise DataValidationError(f"Evaluation data root does not exist: {root}")
    if _is_unsafe_link(root):
        raise DataValidationError(f"Evaluation data root must not be a symlink or junction: {root}")
    if _has_evaluation_images(root):
        return [root]

    scenes: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if _is_unsafe_link(entry):
            raise DataValidationError(f"Evaluation root entry must not be a symlink or junction: {entry.name}")
        if entry.is_file():
            raise DataValidationError(f"Unexpected root file in evaluation data: {entry.name}")
        if not entry.is_dir():
            raise DataValidationError(f"Unsupported root entry in evaluation data: {entry.name}")
        if not _has_evaluation_images(entry):
            raise DataValidationError(f"Evaluation scene must contain test/images: {entry}")
        scenes.append(entry)
    if not scenes:
        raise DataValidationError(f"No evaluation scenes with test/images found under {root}")
    return scenes


def _has_evaluation_images(scene: Path) -> bool:
    return (scene / "test" / "images").is_dir()


def _evaluation_images_dir(scene: Path) -> Path:
    test_dir = scene / "test"
    images_dir = test_dir / "images"
    for path, label in ((scene, "scene"), (test_dir, "test directory"), (images_dir, "image directory")):
        if _is_unsafe_link(path):
            raise DataValidationError(f"Evaluation {label} must not be a symlink or junction: {path}")
    if not images_dir.is_dir():
        raise DataValidationError(f"Evaluation scene has no test/images directory: {scene}")
    return images_dir


def _validate_submission_folder(
    submission_dir: Path,
    *,
    evaluation_scenes: list[_EvaluationScene],
    match_by_stem: bool,
) -> dict[str, Path]:
    if not submission_dir.exists() or not submission_dir.is_dir():
        raise DataValidationError(f"Submission folder does not exist: {submission_dir}")
    if _is_unsafe_link(submission_dir):
        raise DataValidationError(f"Submission folder must not be a symlink or junction: {submission_dir}")

    actual_scenes: dict[str, Path] = {}
    for entry in sorted(submission_dir.iterdir(), key=lambda path: path.name):
        if _is_unsafe_link(entry):
            raise DataValidationError(f"Submission root entry must not be a symlink or junction: {entry.name}")
        if entry.is_file():
            raise DataValidationError(f"Unexpected root file in submission folder: {entry.name}")
        if not entry.is_dir():
            raise DataValidationError(f"Unsupported root entry in submission folder: {entry.name}")
        actual_scenes[entry.name] = entry

    expected_names = {scene.name for scene in evaluation_scenes}
    if set(actual_scenes) != expected_names:
        missing = sorted(expected_names - set(actual_scenes))
        extra = sorted(set(actual_scenes) - expected_names)
        raise DataValidationError(
            "Submission scene directories must exactly match the dataset; "
            f"missing={missing}, extra={extra}"
        )

    for scene in evaluation_scenes:
        predictions = _load_image_specs(
            actual_scenes[scene.name],
            match_by_stem=match_by_stem,
            label=f"prediction scene {scene.name}",
        )
        if set(predictions) != set(scene.images_by_key):
            missing = sorted(set(scene.images_by_key) - set(predictions))
            extra = sorted(set(predictions) - set(scene.images_by_key))
            raise DataValidationError(
                f"Prediction images for {scene.name} must exactly match ground truth; "
                f"missing={missing}, extra={extra}"
            )
        for key, expected in scene.images_by_key.items():
            actual = predictions[key]
            if actual.size != expected.size:
                raise DataValidationError(
                    f"Prediction image resolution mismatch for {scene.name}/{actual.path.name}: "
                    f"expected {expected.size[0]}x{expected.size[1]}, got {actual.size[0]}x{actual.size[1]}"
                )
    return actual_scenes


def _load_image_specs(directory: Path, *, match_by_stem: bool, label: str) -> dict[str, _ImageSpec]:
    if not directory.is_dir() or _is_unsafe_link(directory):
        raise DataValidationError(f"{label} must be a regular directory: {directory}")
    images: dict[str, _ImageSpec] = {}
    casefolded_names: set[str] = set()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if _is_unsafe_link(path):
            raise DataValidationError(f"{label} entry must not be a symlink or junction: {path.name}")
        if path.is_dir():
            raise DataValidationError(f"Nested folders are not allowed in {label}: {path.name}")
        if not path.is_file():
            raise DataValidationError(f"Unsupported entry in {label}: {path.name}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise DataValidationError(f"Unexpected non-image file in {label}: {path.name}")
        folded_name = path.name.casefold()
        if folded_name in casefolded_names:
            raise DataValidationError(f"Case-colliding image name in {label}: {path.name}")
        casefolded_names.add(folded_name)
        key = path.stem if match_by_stem else path.name
        if key in images:
            unit = "stem" if match_by_stem else "name"
            raise DataValidationError(f"Duplicate {label} image {unit}: {key}")
        images[key] = _ImageSpec(path=path, size=_validate_image_payload(path, label=label))
    if not images:
        raise DataValidationError(f"{label} contains no images: {directory}")
    return images


def _validate_image_payload(path: Path, *, label: str) -> tuple[int, int]:
    expected_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    try:
        with Image.open(path) as image:
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
            loaded_format = image.format
            loaded_mode = image.mode
            loaded_size = image.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise DataValidationError(f"Unreadable image payload in {label}: {path.name}") from exc
    if image_format != expected_format or loaded_format != expected_format:
        raise DataValidationError(
            f"Image payload format does not match extension in {label}: {path.name} is {image_format}"
        )
    if image_mode != "RGB" or loaded_mode != "RGB":
        raise DataValidationError(f"Image must use RGB mode in {label}: {path.name}")
    if image_size != loaded_size or image_size[0] <= 0 or image_size[1] <= 0:
        raise DataValidationError(f"Image has invalid or unstable resolution in {label}: {path.name}")
    return image_size


def _is_unsafe_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def _extract_zip_submission(zip_path: Path, output_dir: Path, *, expected_member_count: int) -> None:
    if not zip_path.exists() or not zip_path.is_file():
        raise DataValidationError(f"Submission ZIP does not exist: {zip_path}")
    if _is_unsafe_link(zip_path):
        raise DataValidationError(f"Submission ZIP must not be a symlink or junction: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_MEMBER_COUNT:
                raise DataValidationError(
                    f"ZIP member count exceeds limit: {len(infos)} > {MAX_ZIP_MEMBER_COUNT}"
                )
            seen: set[str] = set()
            seen_casefolded: set[str] = set()
            total_size = 0
            extracted_count = 0
            for info in infos:
                member = info.filename
                if info.is_dir():
                    raise DataValidationError(f"ZIP directory entries are not allowed: {member}")
                parts = _validated_zip_parts(member)
                path = PurePosixPath(*parts)
                normalized = path.as_posix()
                if normalized in seen:
                    raise DataValidationError(f"Duplicate ZIP member path: {normalized}")
                if normalized.casefold() in seen_casefolded:
                    raise DataValidationError(f"Case-colliding ZIP member path: {normalized}")
                seen.add(normalized)
                seen_casefolded.add(normalized.casefold())
                if info.flag_bits & 0x1:
                    raise DataValidationError(f"Encrypted ZIP members are not supported: {member}")
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(unix_mode):
                    raise DataValidationError(f"Symlink ZIP member is not allowed: {member}")
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in {0, stat.S_IFREG}:
                    raise DataValidationError(f"Non-regular ZIP member is not allowed: {member}")
                if Path(parts[1]).suffix.lower() not in IMAGE_SUFFIXES:
                    raise DataValidationError(f"Unsupported image suffix in ZIP member: {member}")
                if info.file_size > MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES:
                    raise DataValidationError(
                        f"ZIP member size exceeds limit for {member}: "
                        f"{info.file_size} > {MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES}"
                    )
                total_size += info.file_size
                if total_size > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
                    raise DataValidationError(
                        "ZIP total uncompressed size exceeds limit: "
                        f"{total_size} > {MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES}"
                    )
                _stream_zip_member(archive, info, output_dir / parts[0] / parts[1])
                extracted_count += 1
            if extracted_count != expected_member_count:
                raise DataValidationError(
                    f"ZIP member count must match ground truth: expected {expected_member_count}, got {extracted_count}"
                )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataValidationError(f"Submission is not a readable ZIP file: {zip_path}") from exc


def _validated_zip_parts(member: str) -> tuple[str, str]:
    if not member or "\\" in member or "\x00" in member or member.startswith("/"):
        raise DataValidationError(f"Unsafe ZIP member path: {member}")
    path = PurePosixPath(member)
    parts = path.parts
    if path.as_posix() != member or path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise DataValidationError(f"Unsafe ZIP member path: {member}")
    if len(parts) != 2:
        raise DataValidationError(f"ZIP member must be exactly scene/image, got: {member}")
    for part in parts:
        windows_path = PureWindowsPath(part)
        has_forbidden_character = any(character in '<>"|?*:' or ord(character) < 32 for character in part)
        basename = part.split(".", 1)[0].upper()
        if (
            windows_path.drive
            or windows_path.name != part
            or basename in WINDOWS_RESERVED_BASENAMES
            or part.endswith((" ", "."))
            or has_forbidden_character
        ):
            raise DataValidationError(f"Unsafe ZIP member path: {member}")
    return parts[0], parts[1]


def _stream_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with archive.open(info) as source, destination.open("xb") as output:
            while chunk := source.read(ZIP_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES:
                    raise DataValidationError(f"ZIP member size exceeds limit while reading: {info.filename}")
                output.write(chunk)
    except (EOFError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        destination.unlink(missing_ok=True)
        raise DataValidationError(f"ZIP CRC/read failure for member: {info.filename}") from exc
    if written != info.file_size:
        destination.unlink(missing_ok=True)
        raise DataValidationError(
            f"ZIP member size changed while reading {info.filename}: expected {info.file_size}, got {written}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a strict VAI holdout submission against test/images by scene.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Holdout scene root containing scene/test/images.",
    )
    parser.add_argument("--submission", type=Path, required=True, help="Submission folder or ZIP to score.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON metrics output path.")
    parser.add_argument(
        "--match-by-stem",
        action="store_true",
        help="Match output/GT images by stem, ignoring extension.",
    )
    parser.add_argument(
        "--psnr-max",
        type=float,
        default=DEFAULT_PSNR_MAX,
        help="Local proxy normalization cap (default: 50); BTC has not published the official PSNR_MAX.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = score_submission(
        data_root=args.data_root,
        submission=args.submission,
        match_by_stem=args.match_by_stem,
        psnr_max=args.psnr_max,
    )
    payload = json.dumps(result, indent=2, allow_nan=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
