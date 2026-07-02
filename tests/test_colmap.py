from pathlib import Path

import numpy as np
from PIL import Image

from bts_nvs.colmap import colmap_model_to_nerfstudio, read_colmap_model


def test_reads_colmap_text_model_and_converts_pose(tmp_path: Path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (640, 480), color=(10, 20, 30)).save(images_dir / "frame_000.png")

    (sparse / "cameras.txt").write_text(
        "# Camera list\n"
        "1 PINHOLE 640 480 500 510 320 240\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 -2 1 frame_000.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text(
        "# Point list\n"
        "7 1.0 2.0 3.0 255 128 0 0.1 1 0\n",
        encoding="utf-8",
    )

    model = read_colmap_model(sparse)
    transforms, points = colmap_model_to_nerfstudio(tmp_path, model)

    assert transforms["fl_x"] == 500.0
    assert transforms["fl_y"] == 510.0
    assert transforms["w"] == 640
    assert transforms["h"] == 480
    assert transforms["frames"][0]["file_path"] == "images/frame_000.png"
    np.testing.assert_allclose(
        np.array(transforms["frames"][0]["transform_matrix"])[:3, 3],
        [0.0, 0.0, 2.0],
        atol=1e-8,
    )
    assert points[0].xyz == (1.0, 2.0, 3.0)
    assert points[0].rgb == (255, 128, 0)
