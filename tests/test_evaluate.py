from pathlib import Path

import math

from PIL import Image

from bts_nvs.evaluate import compute_competition_score, evaluate_directories, normalize_psnr


def test_evaluate_identical_images_reports_zero_mae_and_infinite_psnr(tmp_path: Path):
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    Image.new("RGB", (4, 4), color=(32, 64, 96)).save(pred / "a.png")
    Image.new("RGB", (4, 4), color=(32, 64, 96)).save(gt / "a.png")

    result = evaluate_directories(pred, gt)

    assert result["count"] == 1
    assert result["mae"] == 0.0
    assert math.isinf(result["psnr"])


def test_evaluate_black_vs_white_image_has_zero_psnr_for_8bit_range(tmp_path: Path):
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(pred / "a.png")
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(gt / "a.png")

    result = evaluate_directories(pred, gt)

    assert result["mae"] == 255.0
    assert result["psnr"] == 0.0


def test_evaluate_can_match_png_predictions_to_jpg_ground_truth_by_stem(tmp_path: Path):
    pred = tmp_path / "pred"
    gt = tmp_path / "gt"
    pred.mkdir()
    gt.mkdir()
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(pred / "target_000.png")
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(gt / "target_000.JPG")

    result = evaluate_directories(pred, gt, match_by_stem=True)

    assert result["count"] == 1
    assert result["mae"] == 0.0


def test_normalize_psnr_clamps_to_unit_interval():
    assert normalize_psnr(-1.0, psnr_max=40.0) == 0.0
    assert normalize_psnr(20.0, psnr_max=40.0) == 0.5
    assert normalize_psnr(80.0, psnr_max=40.0) == 1.0
    assert normalize_psnr(math.inf, psnr_max=40.0) == 1.0


def test_compute_competition_score_uses_btc_weighting():
    score = compute_competition_score(psnr=20.0, ssim=0.8, lpips=0.25, psnr_max=40.0)

    assert math.isclose(score, 0.4 * (1.0 - 0.25) + 0.3 * 0.8 + 0.3 * 0.5)
