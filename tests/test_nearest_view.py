from pathlib import Path

from PIL import Image

from bts_nvs.nearest_view import render_nearest_dataset


def _write_nearest_scene(scene: Path, test_pose_name: str = "test_poses.csv") -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "near_red.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "near_blue.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 near_red.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 near_blue.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / test_pose_name).write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def _write_temporal_scene(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "DJI_20250101000001_0001_V.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "DJI_20250101000003_0003_V.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 DJI_20250101000001_0001_V.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 DJI_20250101000003_0003_V.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "DJI_20250101000002_0002_V.png,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def _write_temporal_gap_scene(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    images = scene / "train" / "images"
    test = scene / "test"
    sparse.mkdir(parents=True)
    images.mkdir(parents=True)
    test.mkdir(parents=True)

    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "DJI_20250101000001_0001_V.png")
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(images / "DJI_20250101000004_0004_V.png")

    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 4 4 3 3 2 2\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 DJI_20250101000001_0001_V.png\n"
        "\n"
        "2 1 0 0 0 -10 0 0 1 DJI_20250101000004_0004_V.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "DJI_20250101000002_0002_V.png,1,0,0,0,9,0,0,3,3,2,2,2,2\n",
        encoding="utf-8",
    )


def test_render_nearest_dataset_writes_exact_target_name_by_default(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_nearest_scene(root / "scene_a")

    result = render_nearest_dataset(root=root, output=tmp_path / "submission")

    output = tmp_path / "submission" / "scene_a" / "target.JPG"
    assert result.scene_count == 1
    assert result.image_count == 1
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (2, 2)


def test_render_nearest_dataset_accepts_singular_test_pose_csv_name(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_nearest_scene(root / "scene_a", test_pose_name="test_pose.csv")

    result = render_nearest_dataset(root=root, output=tmp_path / "submission")

    assert result.image_count == 1
    assert (tmp_path / "submission" / "scene_a" / "target.JPG").exists()


def test_render_nearest_dataset_can_force_png_stem_names(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_nearest_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", name_policy="png", image_format="png")

    output = tmp_path / "submission" / "scene_a" / "target.png"
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_render_nearest_dataset_temporal_blend_uses_bracketing_frame_indices(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_temporal_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", selection_mode="temporal-blend")

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    assert output.exists()
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (2, 2)
        red, green, blue = image.getpixel((0, 0))
        assert 120 <= red <= 135
        assert green == 0
        assert 120 <= blue <= 135


def test_render_nearest_dataset_temporal_blend_defaults_to_linear_frame_weight(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_temporal_gap_scene(root / "scene_a")

    render_nearest_dataset(root=root, output=tmp_path / "submission", selection_mode="temporal-blend")

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    with Image.open(output) as image:
        red, green, blue = image.getpixel((0, 0))
        assert 165 <= red <= 175
        assert green == 0
        assert 80 <= blue <= 90


def test_render_nearest_dataset_temporal_blend_can_use_midpoint_weight(tmp_path: Path):
    root = tmp_path / "private_set1"
    _write_temporal_gap_scene(root / "scene_a")

    render_nearest_dataset(
        root=root,
        output=tmp_path / "submission",
        selection_mode="temporal-blend",
        blend_weight_policy="midpoint",
    )

    output = tmp_path / "submission" / "scene_a" / "DJI_20250101000002_0002_V.png"
    with Image.open(output) as image:
        red, green, blue = image.getpixel((0, 0))
        assert 120 <= red <= 135
        assert green == 0
        assert 120 <= blue <= 135
