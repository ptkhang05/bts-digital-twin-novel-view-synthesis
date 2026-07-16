import os
import stat
import struct
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.ingest as ingest_module
from bts_nvs.audit import write_audit_manifest
from bts_nvs.exceptions import DataValidationError
from bts_nvs.ingest import ingest_archive


def _write_dataset_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    data_root = tmp_path / "source" / "VAI_TEST"
    scene = data_root / "scene_a"
    images = scene / "train" / "images"
    sparse = scene / "train" / "sparse" / "0"
    test = scene / "test"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    test.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(images / "train.JPG", format="JPEG")
    (sparse / "cameras.txt").write_text("1 PINHOLE 8 6 10 10 4 3\n", encoding="utf-8")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 train.JPG\n\n"
        "2 1 0 0 0 0 0 0 1 target.JPG\n\n",
        encoding="utf-8",
    )
    (sparse / "points3D.txt").write_text("", encoding="utf-8")
    (test / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,0,0,0,10,10,4,3,8,6\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = write_audit_manifest(data_root, manifest_path)
    return data_root, manifest_path, manifest


def _write_archive(
    archive_path: Path,
    source_root: Path,
    manifest: dict,
    *,
    wrapper: str | None = None,
    omit: set[str] | None = None,
    overrides: dict[str, bytes] | None = None,
    extra: dict[str, bytes] | None = None,
    include_directories: bool = False,
) -> None:
    omit = omit or set()
    overrides = overrides or {}
    extra = extra or {}
    written_directories: set[str] = set()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in manifest["files"]:
            relative = entry["path"]
            if relative in omit:
                continue
            member = f"{wrapper}/{relative}" if wrapper else relative
            if include_directories:
                parts = Path(member).as_posix().split("/")[:-1]
                for index in range(1, len(parts) + 1):
                    directory = "/".join(parts[:index]) + "/"
                    if directory not in written_directories:
                        archive.writestr(directory, b"")
                        written_directories.add(directory)
            payload = overrides.get(relative, (source_root / relative).read_bytes())
            archive.writestr(member, payload)
        for member, payload in extra.items():
            archive.writestr(member, payload)


def _destination(tmp_path: Path) -> Path:
    parent = tmp_path / "installed"
    parent.mkdir(exist_ok=True)
    return parent / "VAI_TEST"


def _assert_no_staging(destination: Path) -> None:
    assert not list(destination.parent.glob(f".{destination.name}.ingest-*"))


@pytest.mark.parametrize("wrapped", [False, True], ids=("flat", "single-wrapper"))
def test_ingest_streams_audits_promotes_and_only_then_deletes_archive(tmp_path: Path, wrapped: bool):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(
        archive,
        source,
        manifest,
        wrapper=source.name if wrapped else None,
        include_directories=wrapped,
    )
    destination = _destination(tmp_path)

    result = ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert result.data_root == destination
    assert result.file_count == manifest["file_count"]
    assert not archive.exists()
    assert (destination / "scene_a" / "train" / "images" / "train.JPG").read_bytes() == (
        source / "scene_a" / "train" / "images" / "train.JPG"
    ).read_bytes()
    _assert_no_staging(destination)


def test_ingest_cli_accepts_required_paths(tmp_path: Path, capsys):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    destination = _destination(tmp_path)

    assert ingest_module.main(
        ["--archive", str(archive), "--data-root", str(destination), "--manifest", str(manifest_path)]
    ) == 0

    assert "Ingested" in capsys.readouterr().out
    assert destination.is_dir()
    assert not archive.exists()


@pytest.mark.parametrize(
    "unsafe_member",
    ["../escape.txt", "/absolute.txt", "scene_a\\backslash.txt", "C:/drive.txt"],
    ids=("traversal", "absolute", "backslash", "windows-drive"),
)
def test_ingest_rejects_unsafe_member_paths_without_extracting(tmp_path: Path, unsafe_member: str):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest, extra={unsafe_member: b"hostile"})
    if "\\" in unsafe_member:
        normalized_name = unsafe_member.replace("\\", "/").encode()
        unsafe_name = unsafe_member.encode()
        payload = archive.read_bytes()
        if unsafe_name not in payload:
            assert normalized_name in payload
            payload = payload.replace(normalized_name, unsafe_name)
            archive.write_bytes(payload)
        assert unsafe_name in archive.read_bytes()
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="Unsafe ZIP member"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()
    _assert_no_staging(destination)


def test_ingest_rejects_duplicate_members(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    duplicate = manifest["files"][0]["path"]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "a") as output:
            output.writestr(duplicate, (source / duplicate).read_bytes())
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="Duplicate ZIP member"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    _assert_no_staging(destination)


def test_ingest_rejects_encrypted_member(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    payload = bytearray(archive.read_bytes())
    local_offset = payload.index(b"PK\x03\x04")
    central_offset = payload.index(b"PK\x01\x02")
    struct.pack_into("<H", payload, local_offset + 6, struct.unpack_from("<H", payload, local_offset + 6)[0] | 1)
    struct.pack_into("<H", payload, central_offset + 8, struct.unpack_from("<H", payload, central_offset + 8)[0] | 1)
    archive.write_bytes(payload)
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="Encrypted ZIP member"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    _assert_no_staging(destination)


def test_ingest_rejects_link_member(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    link = zipfile.ZipInfo("scene_a/unsafe-link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(link, "../../outside")
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="Link ZIP member"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    _assert_no_staging(destination)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_ingest_rejects_missing_or_extra_files(tmp_path: Path, change: str):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    first_path = manifest["files"][0]["path"]
    _write_archive(
        archive,
        source,
        manifest,
        omit={first_path} if change == "missing" else None,
        extra={"unexpected.txt": b"extra"} if change == "extra" else None,
    )
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="members do not match manifest"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    _assert_no_staging(destination)


def test_ingest_rejects_member_size_mismatch_before_extraction(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    first_path = manifest["files"][0]["path"]
    _write_archive(
        archive,
        source,
        manifest,
        overrides={first_path: (source / first_path).read_bytes() + b"x"},
    )
    destination = _destination(tmp_path)

    with pytest.raises(DataValidationError, match="size does not match manifest"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert not destination.exists()
    _assert_no_staging(destination)


def test_audit_failure_preserves_archive_and_existing_empty_destination(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    cameras = "scene_a/train/sparse/0/cameras.txt"
    altered = (source / cameras).read_bytes().replace(b"10 10", b"11 10", 1)
    expected_sizes = {item["path"]: item["size"] for item in manifest["files"]}
    assert len(altered) == expected_sizes[cameras]
    _write_archive(archive, source, manifest, overrides={cameras: altered})
    destination = _destination(tmp_path)
    destination.mkdir()

    with pytest.raises(DataValidationError):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert destination.is_dir()
    assert not any(destination.iterdir())
    _assert_no_staging(destination)


def test_nonempty_destination_is_never_replaced(tmp_path: Path):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    destination = _destination(tmp_path)
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(DataValidationError, match="must not exist or must be empty"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    _assert_no_staging(destination)


def test_promotion_failure_restores_empty_destination_and_preserves_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    destination = _destination(tmp_path)
    destination.mkdir()
    real_replace = os.replace

    def fail_stage_promotion(source_path, destination_path):
        if Path(destination_path) == destination and Path(source_path) != destination:
            raise OSError("simulated promotion failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(ingest_module.os, "replace", fail_stage_promotion)

    with pytest.raises(OSError, match="simulated promotion failure"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    assert destination.is_dir()
    assert not any(destination.iterdir())
    _assert_no_staging(destination)


@pytest.mark.parametrize("destination_exists", [False, True], ids=("absent", "empty"))
def test_archive_deletion_failure_rolls_back_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_exists: bool,
):
    source, manifest_path, manifest = _write_dataset_fixture(tmp_path)
    archive = tmp_path / "dataset.zip"
    _write_archive(archive, source, manifest)
    destination = _destination(tmp_path)
    if destination_exists:
        destination.mkdir()
    real_unlink = Path.unlink

    def fail_archive_deletion(path: Path, *args, **kwargs):
        if path == archive:
            raise OSError("simulated archive deletion failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_archive_deletion)

    with pytest.raises(OSError, match="simulated archive deletion failure"):
        ingest_archive(archive=archive, data_root=destination, manifest=manifest_path)

    assert archive.exists()
    if destination_exists:
        assert destination.is_dir()
        assert not any(destination.iterdir())
    else:
        assert not destination.exists()
    _assert_no_staging(destination)
