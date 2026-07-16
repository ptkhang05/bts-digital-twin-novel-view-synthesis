import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from bts_nvs.audit import (
    audit_dataset,
    check_audit_manifest,
    write_audit_manifest,
)
from bts_nvs.exceptions import DataValidationError

CSV_HEADER = "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"


def _write_auditable_scene(root: Path, *, target_name: str = "target_001.JPG", target_tx: float = 1.0) -> Path:
    scene = root / "scene_a"
    image_dir = scene / "train" / "images"
    sparse_dir = scene / "train" / "sparse" / "0"
    test_dir = scene / "test"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    Image.new("RGB", (8, 6), color=(1, 2, 3)).save(image_dir / "train_001.JPG", format="JPEG")
    (sparse_dir / "cameras.txt").write_text("1 PINHOLE 8 6 10 10 4 3\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 train_001.JPG\n\n"
        "2 1 0 0 0 1 0 0 1 target_001.JPG\n\n",
        encoding="utf-8",
    )
    (test_dir / "test_poses.csv").write_text(
        CSV_HEADER + f"{target_name},1,0,0,0,{target_tx},0,0,10,10,4,3,8,6\n",
        encoding="utf-8",
    )
    (scene / "README.txt").write_text("fixture\n", encoding="utf-8")
    return scene


def test_audit_dataset_writes_deterministic_sorted_manifest(tmp_path: Path):
    data_root = tmp_path / "VAI_NVS_DATA_ROUND2"
    _write_auditable_scene(data_root)
    manifest_path = tmp_path / "manifests" / "vai_nvs_round2.json"

    first = write_audit_manifest(data_root, manifest_path)
    first_bytes = manifest_path.read_bytes()
    second = write_audit_manifest(data_root, manifest_path)

    assert first == second
    assert manifest_path.read_bytes() == first_bytes
    assert first["dataset_id"] == "VAI_NVS_DATA_ROUND2"
    assert first["scene_count"] == 1
    assert first["train_image_count"] == 1
    assert first["target_image_count"] == 1
    paths = [item["path"] for item in first["files"]]
    assert paths == sorted(paths)
    assert all("\\" not in path and not Path(path).is_absolute() for path in paths)

    digest = hashlib.sha256()
    for item in first["files"]:
        digest.update(f'{item["path"]}\0{item["size"]}\0{item["sha256"]}\n'.encode())
    assert first["overall_sha256"] == digest.hexdigest()
    assert json.loads(first_bytes) == first


def test_audit_dataset_rejects_unsafe_target_name(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_auditable_scene(data_root, target_name="../target_001.JPG")

    with pytest.raises(DataValidationError, match="unsafe image_name"):
        audit_dataset(data_root)


def test_audit_dataset_rejects_corrupt_train_jpeg(tmp_path: Path):
    data_root = tmp_path / "data"
    scene = _write_auditable_scene(data_root)
    (scene / "train" / "images" / "train_001.JPG").write_bytes(b"not a jpeg")

    with pytest.raises(DataValidationError, match="not a readable JPEG"):
        audit_dataset(data_root)


def test_audit_dataset_rejects_rotated_exif_orientation(tmp_path: Path):
    data_root = tmp_path / "data"
    scene = _write_auditable_scene(data_root)
    image_path = scene / "train" / "images" / "train_001.JPG"
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    exif = image.getexif()
    exif[274] = 6
    image.save(image_path, format="JPEG", exif=exif)

    with pytest.raises(DataValidationError, match="EXIF Orientation must be absent or 1"):
        audit_dataset(data_root)


def test_audit_dataset_rejects_target_pose_that_disagrees_with_colmap(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_auditable_scene(data_root, target_tx=2.0)

    with pytest.raises(DataValidationError, match="pose does not match COLMAP"):
        audit_dataset(data_root)


def test_audit_manifest_must_not_be_inside_data_root(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_auditable_scene(data_root)

    with pytest.raises(DataValidationError, match="outside the data root"):
        write_audit_manifest(data_root, data_root / "manifest.json")


def test_check_audit_manifest_is_read_only_and_detects_file_mismatch(tmp_path: Path):
    data_root = tmp_path / "data"
    scene = _write_auditable_scene(data_root)
    manifest_path = tmp_path / "manifest.json"
    write_audit_manifest(data_root, manifest_path)
    original_manifest = manifest_path.read_bytes()

    checked = check_audit_manifest(data_root, manifest_path)

    assert checked["overall_sha256"] == json.loads(original_manifest)["overall_sha256"]
    assert manifest_path.read_bytes() == original_manifest

    (scene / "README.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Audit manifest mismatch"):
        check_audit_manifest(data_root, manifest_path)
    assert manifest_path.read_bytes() == original_manifest
