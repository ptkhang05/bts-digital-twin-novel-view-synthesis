from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from bts_nvs.evaluate import DEFAULT_PSNR_MAX, compute_competition_score, evaluate_directories, normalize_psnr
from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import discover_vai_phase1_scenes


def score_submission(
    data_root: Path | str,
    submission: Path | str,
    match_by_stem: bool = False,
    psnr_max: float = DEFAULT_PSNR_MAX,
) -> dict[str, Any]:
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

    scene_results: list[dict[str, Any]] = []
    for scene in discover_vai_phase1_scenes(data_root):
        pred_dir = submission_dir / scene.name
        gt_dir = scene / "test" / "images"
        if not gt_dir.is_dir():
            raise DataValidationError(f"Scene has no public ground-truth images: {gt_dir}")
        metrics = evaluate_directories(pred_dir, gt_dir, match_by_stem=match_by_stem, psnr_max=psnr_max)
        scene_results.append({"scene": scene.name, **metrics})

    return {
        "aggregate": _aggregate_metrics(scene_results, psnr_max=psnr_max),
        "scenes": scene_results,
    }


def _aggregate_metrics(scene_results: list[dict[str, Any]], psnr_max: float) -> dict[str, Any]:
    if not scene_results:
        raise DataValidationError("No scene metrics were computed")

    total_count = sum(int(scene["count"]) for scene in scene_results)
    if total_count <= 0:
        raise DataValidationError("No image pairs were scored")

    aggregate: dict[str, Any] = {
        "count": total_count,
        "mae": _weighted_mean(scene_results, "mae", total_count),
        "mse": _weighted_mean(scene_results, "mse", total_count),
    }
    aggregate["psnr"] = _psnr_from_mse(float(aggregate["mse"]))

    if all("ssim" in scene for scene in scene_results):
        aggregate["ssim"] = _weighted_mean(scene_results, "ssim", total_count)
    if all("lpips" in scene for scene in scene_results):
        aggregate["lpips"] = _weighted_mean(scene_results, "lpips", total_count)
    if "ssim" in aggregate and "lpips" in aggregate:
        aggregate["psnr_norm"] = normalize_psnr(float(aggregate["psnr"]), psnr_max=psnr_max)
        aggregate["score"] = compute_competition_score(
            psnr=float(aggregate["psnr"]),
            ssim=float(aggregate["ssim"]),
            lpips=float(aggregate["lpips"]),
            psnr_max=psnr_max,
        )
    return aggregate


def _weighted_mean(scene_results: list[dict[str, Any]], key: str, total_count: int) -> float:
    return float(sum(float(scene[key]) * int(scene["count"]) for scene in scene_results) / total_count)


def _psnr_from_mse(mse: float) -> float:
    return math.inf if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _extract_zip_submission(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise DataValidationError(f"Submission ZIP does not exist: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                parts = Path(member).parts
                if Path(member).is_absolute() or ".." in parts:
                    raise DataValidationError(f"Unsafe ZIP member path: {member}")
                if len(parts) != 2:
                    raise DataValidationError(f"ZIP member must be exactly scene/image, got: {member}")
            archive.extractall(output_dir)
    except zipfile.BadZipFile as exc:
        raise DataValidationError(f"Submission is not a readable ZIP file: {zip_path}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a public-set BTC/VAI submission against test/images by scene.")
    parser.add_argument("--data-root", type=Path, required=True, help="Public VAI scene root containing scene/test/images.")
    parser.add_argument("--submission", type=Path, required=True, help="Submission folder or ZIP to score.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--match-by-stem", action="store_true", help="Match output/GT images by stem, ignoring extension.")
    parser.add_argument("--psnr-max", type=float, default=DEFAULT_PSNR_MAX)
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
