from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from bts_nvs.exceptions import DataValidationError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_PSNR_MAX = 40.0
MIN_LPIPS_IMAGE_SIDE = 32


def evaluate_directories(
    pred: Path | str,
    gt: Path | str,
    match_by_stem: bool = False,
    psnr_max: float = DEFAULT_PSNR_MAX,
) -> dict[str, float | int]:
    pred_dir = Path(pred)
    gt_dir = Path(gt)
    pairs = _match_images(pred_dir, gt_dir, match_by_stem=match_by_stem)
    if not pairs:
        raise DataValidationError(f"No matching images found between {pred_dir} and {gt_dir}")

    maes: list[float] = []
    mses: list[float] = []
    for pred_path, gt_path in pairs:
        pred_image = _load_rgb(pred_path)
        gt_image = _load_rgb(gt_path)
        if pred_image.shape != gt_image.shape:
            raise DataValidationError(f"Shape mismatch for {pred_path.name}: {pred_image.shape} vs {gt_image.shape}")
        diff = pred_image - gt_image
        maes.append(float(np.mean(np.abs(diff))))
        mses.append(float(np.mean(diff * diff)))

    mse = float(np.mean(mses))
    psnr = math.inf if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    result: dict[str, float | int] = {
        "count": len(pairs),
        "mae": float(np.mean(maes)),
        "mse": mse,
        "psnr": psnr,
    }
    ssim = _try_compute_ssim(pairs)
    if ssim is not None:
        result["ssim"] = ssim
    lpips_score = _try_compute_lpips(pairs)
    if lpips_score is not None:
        result["lpips"] = lpips_score
    if ssim is not None and lpips_score is not None:
        result["psnr_norm"] = normalize_psnr(psnr, psnr_max=psnr_max)
        result["score"] = compute_competition_score(psnr=psnr, ssim=ssim, lpips=lpips_score, psnr_max=psnr_max)
    return result


def normalize_psnr(psnr: float, psnr_max: float = DEFAULT_PSNR_MAX) -> float:
    if psnr_max <= 0:
        raise ValueError("psnr_max must be positive")
    if math.isinf(psnr):
        return 1.0
    return max(0.0, min(float(psnr) / float(psnr_max), 1.0))


def compute_competition_score(
    psnr: float,
    ssim: float,
    lpips: float,
    psnr_max: float = DEFAULT_PSNR_MAX,
) -> float:
    psnr_norm = normalize_psnr(psnr, psnr_max=psnr_max)
    return 0.4 * (1.0 - float(lpips)) + 0.3 * float(ssim) + 0.3 * psnr_norm


def _match_images(pred_dir: Path, gt_dir: Path, match_by_stem: bool) -> list[tuple[Path, Path]]:
    if not pred_dir.exists():
        raise DataValidationError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise DataValidationError(f"Ground-truth directory does not exist: {gt_dir}")
    pred_by_name = {path.name: path for path in pred_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    gt_by_name = {path.name: path for path in gt_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    if not match_by_stem:
        return [(pred_by_name[name], gt_by_name[name]) for name in sorted(pred_by_name.keys() & gt_by_name.keys())]

    pred_by_stem = _unique_by_stem(pred_by_name.values(), "prediction")
    gt_by_stem = _unique_by_stem(gt_by_name.values(), "ground-truth")
    return [(pred_by_stem[stem], gt_by_stem[stem]) for stem in sorted(pred_by_stem.keys() & gt_by_stem.keys())]


def _unique_by_stem(paths, label: str) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for path in paths:
        key = path.stem
        if key in by_stem:
            raise DataValidationError(f"Duplicate {label} image stem: {key}")
        by_stem[key] = path
    return by_stem


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


def _try_compute_ssim(pairs: list[tuple[Path, Path]]) -> float | None:
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None
    scores = []
    for pred_path, gt_path in pairs:
        pred = _load_rgb(pred_path)
        gt = _load_rgb(gt_path)
        min_side = min(pred.shape[0], pred.shape[1], gt.shape[0], gt.shape[1])
        if min_side < 3:
            return None
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        scores.append(float(structural_similarity(gt, pred, channel_axis=2, data_range=255, win_size=win_size)))
    return float(np.mean(scores))


def _try_compute_lpips(pairs: list[tuple[Path, Path]]) -> float | None:
    if not _pairs_are_large_enough_for_lpips(pairs):
        return None

    try:
        import lpips
        import torch
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)
    scores = []
    with torch.no_grad():
        for pred_path, gt_path in pairs:
            pred = _load_rgb(pred_path)
            gt = _load_rgb(gt_path)
            pred_tensor = _image_to_lpips_tensor(pred, torch, device)
            gt_tensor = _image_to_lpips_tensor(gt, torch, device)
            scores.append(float(loss_fn(pred_tensor, gt_tensor).item()))
    return float(np.mean(scores))


def _pairs_are_large_enough_for_lpips(pairs: list[tuple[Path, Path]]) -> bool:
    for pred_path, gt_path in pairs:
        pred = _load_rgb(pred_path)
        gt = _load_rgb(gt_path)
        min_side = min(pred.shape[0], pred.shape[1], gt.shape[0], gt.shape[1])
        if min_side < MIN_LPIPS_IMAGE_SIDE:
            return False
    return True


def _image_to_lpips_tensor(image: np.ndarray, torch_module, device: str):
    tensor = torch_module.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rendered RGB images against ground truth.")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--match-by-stem", action="store_true", help="Match pred/gt images by filename stem, ignoring extension.")
    parser.add_argument(
        "--psnr-max",
        type=float,
        default=DEFAULT_PSNR_MAX,
        help="PSNR value that maps to psnr_norm=1.0 for BTC aggregate score.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_directories(args.pred, args.gt, match_by_stem=args.match_by_stem, psnr_max=args.psnr_max)
    payload = json.dumps(result, indent=2, allow_nan=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
