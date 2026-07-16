from __future__ import annotations

import argparse
import json
import math
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from bts_nvs.evaluate import DEFAULT_PSNR_MAX, IMAGE_SUFFIXES, evaluate_directories
from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import discover_vai_scenes

LOCAL_PROXY_SCORE_LABEL = "local_proxy_not_official_btc_score"


def score_submission(
    data_root: Path | str,
    submission: Path | str,
    match_by_stem: bool = False,
    psnr_max: float = DEFAULT_PSNR_MAX,
) -> dict[str, Any]:
    _validate_psnr_max(psnr_max)
    submission_path = Path(submission)
    if submission_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="bts_nvs_submission_") as temp_dir:
            _extract_zip_submission(submission_path, Path(temp_dir))
            return _score_submission_folder(
                data_root=Path(data_root),
                submission_dir=Path(temp_dir),
                match_by_stem=match_by_stem,
                psnr_max=psnr_max,
            )
    return _score_submission_folder(
        data_root=Path(data_root),
        submission_dir=submission_path,
        match_by_stem=match_by_stem,
        psnr_max=psnr_max,
    )


def _score_submission_folder(
    data_root: Path,
    submission_dir: Path,
    match_by_stem: bool,
    psnr_max: float,
) -> dict[str, Any]:
    if not submission_dir.exists() or not submission_dir.is_dir():
        raise DataValidationError(f"Submission folder does not exist: {submission_dir}")

    scenes = discover_vai_scenes(data_root)
    _validate_scene_directories(submission_dir, scenes)

    scene_results: list[dict[str, Any]] = []
    for scene in scenes:
        pred_dir = submission_dir / scene.name
        gt_dir = scene / "test" / "images"
        if not gt_dir.is_dir():
            raise DataValidationError(f"Scene has no public ground-truth images: {gt_dir}")
        _validate_exact_image_set(pred_dir, gt_dir, match_by_stem=match_by_stem)
        metrics = evaluate_directories(pred_dir, gt_dir, match_by_stem=match_by_stem, psnr_max=psnr_max)
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


def _validate_scene_directories(submission_dir: Path, scenes: list[Path]) -> None:
    expected = {scene.name for scene in scenes}
    actual = {path.name for path in submission_dir.iterdir() if path.is_dir()}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DataValidationError(
            "Submission scene directories must exactly match the dataset; "
            f"missing={missing}, extra={extra}"
        )


def _validate_exact_image_set(pred_dir: Path, gt_dir: Path, match_by_stem: bool) -> None:
    if not pred_dir.is_dir():
        raise DataValidationError(f"Prediction directory does not exist: {pred_dir}")
    pred_keys = _image_keys(pred_dir, match_by_stem=match_by_stem, label="prediction")
    gt_keys = _image_keys(gt_dir, match_by_stem=match_by_stem, label="ground-truth")
    if pred_keys != gt_keys:
        missing = sorted(gt_keys - pred_keys)
        extra = sorted(pred_keys - gt_keys)
        raise DataValidationError(
            f"Prediction images for {pred_dir.name} must exactly match ground truth; "
            f"missing={missing}, extra={extra}"
        )


def _image_keys(directory: Path, match_by_stem: bool, label: str) -> set[str]:
    paths = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    keys: set[str] = set()
    duplicates: set[str] = set()
    for path in paths:
        key = path.stem if match_by_stem else path.name
        if key in keys:
            duplicates.add(key)
        keys.add(key)
    if duplicates:
        unit = "stem" if match_by_stem else "name"
        raise DataValidationError(f"Duplicate {label} image {unit}: {sorted(duplicates)}")
    return keys


def _extract_zip_submission(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise DataValidationError(f"Submission ZIP does not exist: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            seen: set[str] = set()
            for info in archive.infolist():
                member = info.filename
                if info.is_dir():
                    continue
                normalized = member.replace("\\", "/")
                path = PurePosixPath(normalized)
                parts = path.parts
                if not parts or path.is_absolute() or ".." in parts or ":" in parts[0]:
                    raise DataValidationError(f"Unsafe ZIP member path: {member}")
                if len(parts) != 2:
                    raise DataValidationError(f"ZIP member must be exactly scene/image, got: {member}")
                normalized = path.as_posix()
                if normalized in seen:
                    raise DataValidationError(f"Duplicate ZIP member path: {normalized}")
                seen.add(normalized)
                if info.flag_bits & 0x1:
                    raise DataValidationError(f"Encrypted ZIP members are not supported: {member}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise DataValidationError(f"Symbolic links are not supported in ZIP: {member}")
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise DataValidationError(f"ZIP CRC check failed for member: {corrupt_member}")
            archive.extractall(output_dir)
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"Submission is not a readable ZIP file: {zip_path}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a public-set BTC/VAI submission against test/images by scene.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Public VAI scene root containing scene/test/images.",
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
