from pathlib import Path

from PIL import Image

from bts_nvs.prepare_dataset import prepare_dataset


def _write_tiny_vai_scene(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "train" / "images" / "keep.png")
    (sparse / "cameras.txt").write_text("1 PINHOLE 16 12 10 11 8 6\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 -2 1 keep.png\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )


def test_prepare_dataset_processes_each_vai_scene_under_root(tmp_path: Path):
    root = tmp_path / "public_set"
    _write_tiny_vai_scene(root / "scene_a")
    _write_tiny_vai_scene(root / "scene_b")

    result = prepare_dataset(root=root, output=tmp_path / "processed")

    assert result.scene_count == 2
    assert result.image_count == 2
    assert (tmp_path / "processed" / "scene_a" / "transforms.json").exists()
    assert (tmp_path / "processed" / "scene_b" / "target_cameras.json").exists()
