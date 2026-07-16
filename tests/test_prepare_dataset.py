import json
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.prepare_dataset as prepare_dataset_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.prepare_dataset import build_arg_parser, prepare_dataset


def _write_tiny_vai_scene(scene: Path, test_pose_name: str = "test_poses.csv") -> None:
    sparse = scene / "train" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(scene / "train" / "images" / "keep.png")
    (sparse / "cameras.txt").write_text("1 PINHOLE 16 12 10 11 8 6\n", encoding="utf-8")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 -2 1 keep.png\n\n", encoding="utf-8")
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (scene / "test" / test_pose_name).write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,1,2,3,10,11,8,6,16,12\n",
        encoding="utf-8",
    )


def test_prepare_dataset_processes_each_vai_scene_under_current_dataset_root(tmp_path: Path):
    root = tmp_path / "public_set"
    _write_tiny_vai_scene(root / "scene_a")
    _write_tiny_vai_scene(root / "scene_b")

    result = prepare_dataset(root=root, output=tmp_path / "processed")

    assert result.scene_count == 2
    assert result.image_count == 2
    assert (tmp_path / "processed" / "scene_a" / "transforms.json").exists()
    assert (tmp_path / "processed" / "scene_b" / "target_cameras.json").exists()


def test_prepare_dataset_verifies_manifest_and_propagates_identity_to_every_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    output = tmp_path / "processed"
    manifest = tmp_path / "vai_nvs_round2.json"
    _write_tiny_vai_scene(root / "scene_a")
    _write_tiny_vai_scene(root / "scene_b")
    manifest.write_text("{}\n", encoding="utf-8")
    verified_hash = "b" * 64

    def fake_check_audit_manifest(data_root: Path, manifest_path: Path) -> dict:
        assert Path(data_root) == root
        assert Path(manifest_path) == manifest
        return {
            "dataset_id": root.name,
            "overall_sha256": verified_hash,
        }

    monkeypatch.setattr(prepare_dataset_module, "check_audit_manifest", fake_check_audit_manifest)

    result = prepare_dataset(
        root=root,
        output=output,
        dataset_id="vai_nvs_round2",
        manifest=manifest,
    )

    assert result.scene_count == 2
    for scene_name in ("scene_a", "scene_b"):
        metadata = json.loads((output / scene_name / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["provenance_status"] == "verified"
        assert metadata["dataset_id"] == "vai_nvs_round2"
        assert metadata["dataset_manifest_sha256"] == verified_hash
        assert metadata["source_scene"] == scene_name
        assert not Path(metadata["source_scene"]).is_absolute()


@pytest.mark.parametrize(
    ("dataset_id", "manifest"),
    [
        (None, "manifest.json"),
        ("vai_nvs_round2", None),
    ],
)
def test_prepare_dataset_rejects_incomplete_provenance_before_writing_output(
    tmp_path: Path,
    dataset_id: str | None,
    manifest: str | None,
):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    output = tmp_path / "processed"
    _write_tiny_vai_scene(root / "scene_a")

    with pytest.raises(DataValidationError, match="dataset_id.*manifest|manifest.*dataset_id"):
        prepare_dataset(
            root=root,
            output=output,
            dataset_id=dataset_id,
            manifest=tmp_path / manifest if manifest is not None else None,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "verified_manifest",
    [
        {"dataset_id": "wrong_root", "overall_sha256": "b" * 64},
        {"dataset_id": "VAI_NVS_DATA_ROUND2", "overall_sha256": "not-a-sha256"},
    ],
)
def test_prepare_dataset_rejects_invalid_verified_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified_manifest: dict,
):
    root = tmp_path / "VAI_NVS_DATA_ROUND2"
    output = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    _write_tiny_vai_scene(root / "scene_a")
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_dataset_module,
        "check_audit_manifest",
        lambda _root, _manifest: verified_manifest,
    )

    with pytest.raises(DataValidationError, match="identity|SHA-256"):
        prepare_dataset(
            root=root,
            output=output,
            dataset_id="vai_nvs_round2",
            manifest=manifest,
        )

    assert not output.exists()


def test_prepare_dataset_cli_accepts_production_provenance_inputs(tmp_path: Path):
    args = build_arg_parser().parse_args(
        [
            "--root",
            str(tmp_path / "VAI_NVS_DATA_ROUND2"),
            "--out",
            str(tmp_path / "processed"),
            "--dataset-id",
            "vai_nvs_round2",
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert args.dataset_id == "vai_nvs_round2"
    assert args.manifest == tmp_path / "manifest.json"


def test_prepare_dataset_rejects_noncanonical_singular_test_pose_csv_name(tmp_path: Path):
    root = tmp_path / "public_set"
    _write_tiny_vai_scene(root / "scene_a", test_pose_name="test_pose.csv")

    with pytest.raises(DataValidationError, match="test/test_poses.csv"):
        prepare_dataset(root=root, output=tmp_path / "processed")


def test_prepare_dataset_rejects_output_nested_inside_raw_dataset(tmp_path: Path):
    root = tmp_path / "public_set"
    _write_tiny_vai_scene(root / "scene_a")
    source_image = root / "scene_a" / "train" / "images" / "keep.png"

    with pytest.raises(DataValidationError, match="overlap"):
        prepare_dataset(root=root, output=root / "processed", overwrite=True)

    assert source_image.exists()
