from pathlib import Path

import math

from PIL import Image

from bts_nvs.evaluate import evaluate_directories


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
