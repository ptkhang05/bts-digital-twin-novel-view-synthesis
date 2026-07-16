import json
import struct
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.path_safety as path_safety_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.prepare import build_arg_parser, prepare_scene


def _write_minimal_scene(scene: Path, image_name: str = "frame.png") -> None:
    (scene / "images").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(100, 120, 140)).save(scene / "images" / image_name)
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": f"images/{image_name}",
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_target_cameras(scene: Path, count: int = 2) -> None:
    frames = []
    for index in range(count):
        frames.append(
            {
                "file_path": f"target_{index:03d}.png",
                "transform_matrix": [
                    [1, 0, 0, index],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": frames,
    }
    (scene / "target_cameras.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_vai_scene_with_sparse_point(scene: Path) -> None:
    sparse = scene / "train" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "train" / "images" / "keep.png")
    (sparse / "cameras.txt").write_text("1 PINHOLE 16 12 10 11 8 6\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 -2 1 keep.png\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("1 1 2 3 10 20 30 0.5\n", encoding="utf-8")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )


def test_prepare_scene_copies_images_and_writes_nerfstudio_transforms(tmp_path: Path):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)

    result = prepare_scene(scene=scene, output=output)

    assert result.transforms_path == output / "transforms.json"
    assert (output / "images" / "frame.png").exists()
    transforms = json.loads((output / "transforms.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert transforms["frames"][0]["file_path"] == "images/frame.png"
    assert metadata["image_count"] == 1
    assert metadata["provenance_status"] == "partial"
    assert metadata["transforms"] == "transforms.json"


def test_prepare_scene_writes_verified_relative_provenance_metadata(tmp_path: Path):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)
    _write_target_cameras(scene, count=2)

    prepare_scene(
        scene=scene,
        output=output,
        dataset_id="vai_nvs_round2",
        dataset_manifest_sha256="a" * 64,
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["provenance_status"] == "verified"
    assert metadata["dataset_id"] == "vai_nvs_round2"
    assert metadata["dataset_manifest_sha256"] == "a" * 64
    assert metadata["source_scene"] == "raw"
    assert metadata["transforms"] == "transforms.json"
    assert metadata["target_cameras"] == "target_cameras.json"
    for key in ("source_scene", "transforms", "target_cameras"):
        value = Path(metadata[key])
        assert not value.is_absolute()
        assert ".." not in value.parts


@pytest.mark.parametrize(
    ("dataset_id", "manifest_sha256"),
    [
        (None, "a" * 64),
        ("vai_nvs_round2", None),
        ("vai_nvs_round2", "not-a-sha256"),
        ("../vai_nvs_round2", "a" * 64),
        (r"C:\\datasets\\vai_nvs_round2", "a" * 64),
    ],
)
def test_prepare_scene_rejects_incomplete_or_unsafe_provenance_before_overwrite(
    tmp_path: Path,
    dataset_id: str | None,
    manifest_sha256: str | None,
):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")

    with pytest.raises(DataValidationError, match="provenance|dataset_id|SHA-256"):
        prepare_scene(
            scene=scene,
            output=output,
            overwrite=True,
            dataset_id=dataset_id,
            dataset_manifest_sha256=manifest_sha256,
        )

    assert sentinel.read_text(encoding="utf-8") == "last-known-good"


def test_prepare_scene_cli_accepts_explicit_provenance_inputs(tmp_path: Path):
    args = build_arg_parser().parse_args(
        [
            "--scene",
            str(tmp_path / "scene"),
            "--out",
            str(tmp_path / "processed"),
            "--dataset-id",
            "vai_nvs_round2",
            "--manifest-sha256",
            "c" * 64,
        ]
    )

    assert args.dataset_id == "vai_nvs_round2"
    assert args.manifest_sha256 == "c" * 64


def test_prepare_scene_copies_target_cameras_when_present(tmp_path: Path):
    scene = tmp_path / "raw"
    output = tmp_path / "processed"
    _write_minimal_scene(scene)
    _write_target_cameras(scene, count=2)

    result = prepare_scene(scene=scene, output=output)

    targets = json.loads((output / "target_cameras.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert result.target_count == 2
    assert len(targets["frames"]) == 2
    assert metadata["target_count"] == 2


def test_prepare_scene_rejects_missing_frame_image(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene, image_name="missing.png")
    (scene / "images" / "missing.png").unlink()

    with pytest.raises(DataValidationError, match="missing.png"):
        prepare_scene(scene=scene, output=tmp_path / "processed")


def test_prepare_scene_rejects_output_equal_to_source_before_deleting_it(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene)
    source_image = scene / "images" / "frame.png"

    with pytest.raises(DataValidationError, match="overlap"):
        prepare_scene(scene=scene, output=scene, overwrite=True)

    assert source_image.exists()


@pytest.mark.parametrize("frame_path_kind", ["parent", "absolute"])
def test_prepare_scene_rejects_frame_images_outside_scene(tmp_path: Path, frame_path_kind: str):
    scene = tmp_path / "raw"
    scene.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (16, 12), color=(100, 120, 140)).save(outside)
    frame_path = "../outside.png" if frame_path_kind == "parent" else str(outside.resolve())
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": [
            {
                "file_path": frame_path,
                "transform_matrix": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="inside the scene"):
        prepare_scene(scene=scene, output=tmp_path / "processed")


def test_prepare_scene_rejects_frame_image_symlink(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene)
    source = scene / "images" / "frame.png"
    outside = tmp_path / "outside.png"
    source.replace(outside)
    try:
        source.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    output = tmp_path / "processed"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")

    with pytest.raises(DataValidationError, match="symlink or junction"):
        prepare_scene(scene=scene, output=output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "last-known-good"


def test_prepare_scene_preflights_source_links_before_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene)
    source = (scene / "images" / "frame.png").absolute()
    output = tmp_path / "processed"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")
    real_checker = path_safety_module.is_link_or_junction

    monkeypatch.setattr(
        path_safety_module,
        "is_link_or_junction",
        lambda path: path == source or real_checker(path),
    )

    with pytest.raises(DataValidationError, match="symlink or junction"):
        prepare_scene(scene=scene, output=output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "last-known-good"


def test_prepare_scene_rejects_symlink_copy_mode_before_overwrite(tmp_path: Path):
    scene = tmp_path / "raw"
    _write_minimal_scene(scene)
    output = tmp_path / "processed"
    output.mkdir()
    sentinel = output / "previous.txt"
    sentinel.write_text("last-known-good", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Unsupported copy mode"):
        prepare_scene(scene=scene, output=output, copy_mode="symlink", overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "last-known-good"


def test_prepare_scene_supports_vai_layout_and_filters_sparse_images_by_exact_train_filenames(tmp_path: Path):
    scene = tmp_path / "official_scene"
    sparse = scene / "train" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "train" / "images" / "keep.png")

    (sparse / "cameras.txt").write_text(
        "# Camera list\n"
        "1 SIMPLE_RADIAL 16 12 10 8 6 -0.1\n",
        encoding="utf-8",
    )
    (sparse / "images.txt").write_text(
        "# Image list\n"
        "1 1 0 0 0 0 0 -2 1 keep.png\n"
        "\n"
        "2 1 0 0 0 0 0 -3 1 missing.png\n"
        "\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("1 1 2 3 10 20 30 0.5\n", encoding="utf-8")
    source_ply = b"ply\nformat ascii 1.0\ncomment official source\nelement vertex 1\nend_header\n"
    (sparse / "points3D.ply").write_bytes(source_ply)
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )

    result = prepare_scene(scene=scene, output=tmp_path / "processed")

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    targets = json.loads((tmp_path / "processed" / "target_cameras.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "processed" / "metadata.json").read_text(encoding="utf-8"))
    assert result.image_count == 1
    assert result.source_format == "vai"
    assert transforms["frames"][0]["file_path"] == "images/keep.png"
    assert [frame["file_path"] for frame in transforms["frames"]] == ["images/keep.png"]
    assert result.target_count == 1
    assert targets["frames"][0]["file_path"] == "target.JPG"
    assert targets["frames"][0]["transform_matrix"][0][3] == -1.0
    assert targets["k1"] == -0.1
    assert transforms["ply_file_path"] == "sparse_pc.ply"
    assert (tmp_path / "processed" / "sparse_pc.ply").read_bytes() == source_ply
    assert metadata["source_scene"] == "official_scene"
    assert not Path(metadata["source_scene"]).is_absolute()


def test_prepare_scene_allows_only_vai_chair_to_synthesize_missing_sparse_ply(tmp_path: Path):
    scene = tmp_path / "chair"
    _write_vai_scene_with_sparse_point(scene)

    result = prepare_scene(scene=scene, output=tmp_path / "processed")

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert result.point_count == 1
    assert transforms["ply_file_path"] == "sparse_pc.ply"
    assert (tmp_path / "processed" / "sparse_pc.ply").read_text(encoding="ascii").startswith("ply\n")


def test_prepare_scene_rejects_missing_sparse_ply_for_non_chair_vai_scene(tmp_path: Path):
    scene = tmp_path / "hcm_scene"
    _write_vai_scene_with_sparse_point(scene)

    with pytest.raises(DataValidationError, match=r"points3D\.ply.*chair"):
        prepare_scene(scene=scene, output=tmp_path / "processed")

    assert not (tmp_path / "processed" / "sparse_pc.ply").exists()


def test_prepare_scene_generates_sparse_ply_from_colmap_binary_when_source_ply_is_missing(tmp_path: Path):
    scene = tmp_path / "colmap_scene"
    sparse = scene / "sparse" / "0"
    images = scene / "images"
    sparse.mkdir(parents=True)
    images.mkdir()
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(images / "frame.png")
    (sparse / "cameras.bin").write_bytes(
        struct.pack("<QiiQQdddd", 1, 1, 1, 16, 12, 10.0, 11.0, 8.0, 6.0)
    )
    (sparse / "images.bin").write_bytes(
        struct.pack("<Qidddddddi", 1, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 1)
        + b"frame.png\x00"
        + struct.pack("<Q", 0)
    )
    (sparse / "points3D.bin").write_bytes(
        struct.pack("<QQdddBBBdQ", 1, 1, 1.0, 2.0, 3.0, 10, 20, 30, 0.5, 0)
    )

    result = prepare_scene(scene=scene, output=tmp_path / "processed")

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert result.point_count == 1
    assert transforms["ply_file_path"] == "sparse_pc.ply"
    assert (tmp_path / "processed" / "sparse_pc.ply").read_text(encoding="ascii").startswith("ply\n")


def test_prepare_scene_can_write_filename_holdout_split(tmp_path: Path):
    scene = tmp_path / "raw"
    (scene / "images").mkdir(parents=True)
    frames = []
    for index in range(4):
        name = f"frame_{index}.png"
        Image.new("RGB", (16, 12), color=(index, index, index)).save(scene / "images" / name)
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": [
                    [1, 0, 0, index],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": frames,
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")

    prepare_scene(scene=scene, output=tmp_path / "processed", holdout_interval=2)

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert transforms["train_filenames"] == ["images/frame_0.png", "images/frame_2.png"]
    assert transforms["val_filenames"] == ["images/frame_1.png", "images/frame_3.png"]
    assert transforms["test_filenames"] == ["images/frame_1.png", "images/frame_3.png"]


def test_holdout_split_sorts_filenames_before_selecting_positions(tmp_path: Path):
    scene = tmp_path / "raw"
    (scene / "images").mkdir(parents=True)
    frames = []
    for name in ("frame_30.png", "frame_10.png", "frame_20.png"):
        Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "images" / name)
        frames.append(
            {
                "file_path": f"images/{name}",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    payload = {
        "camera_model": "OPENCV",
        "fl_x": 10.0,
        "fl_y": 11.0,
        "cx": 8.0,
        "cy": 6.0,
        "w": 16,
        "h": 12,
        "frames": frames,
    }
    (scene / "train_cameras.json").write_text(json.dumps(payload), encoding="utf-8")

    prepare_scene(scene=scene, output=tmp_path / "processed", holdout_interval=2)

    transforms = json.loads((tmp_path / "processed" / "transforms.json").read_text(encoding="utf-8"))
    assert transforms["train_filenames"] == ["images/frame_10.png", "images/frame_30.png"]
    assert transforms["val_filenames"] == ["images/frame_20.png"]
    assert transforms["test_filenames"] == ["images/frame_20.png"]
