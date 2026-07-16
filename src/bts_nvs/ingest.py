from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from bts_nvs.audit import check_audit_manifest
from bts_nvs.exceptions import DataValidationError

_COPY_CHUNK_SIZE = 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARACTERS = frozenset('<>:"|?*')


@dataclass(frozen=True)
class IngestResult:
    """Summary of a successfully installed dataset archive."""

    data_root: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class _ExpectedFile:
    path: str
    size: int


@dataclass(frozen=True)
class _ArchivePlan:
    members: dict[str, zipfile.ZipInfo]
    wrapper: str | None


def ingest_archive(
    archive: Path | str,
    data_root: Path | str,
    manifest: Path | str,
) -> IngestResult:
    """Validate, audit, and atomically install an untrusted dataset ZIP.

    The archive remains in place unless extraction, manifest verification, and
    promotion all succeed. Existing non-empty destinations are never replaced.
    """

    archive_path = Path(archive)
    destination = Path(data_root)
    manifest_path = Path(manifest)

    _require_regular_file(archive_path, "Dataset archive")
    _require_regular_file(manifest_path, "Audit manifest")
    expected, dataset_id = _load_expected_files(manifest_path)
    destination_existed, destination_identity = _validate_destination(
        destination,
        dataset_id=dataset_id,
    )

    staging_container: Path | None = None
    staging_data: Path | None = None
    previous_destination: Path | None = None
    destination_moved = False
    promoted = False

    try:
        with _open_archive(archive_path) as source_zip:
            plan = _build_archive_plan(source_zip, archive_path, expected)

            staging_container = Path(
                tempfile.mkdtemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.ingest-",
                )
            )
            staging_data = staging_container / destination.name
            staging_data.mkdir()
            _extract_plan(source_zip, plan, staging_data, expected)

        audited = check_audit_manifest(staging_data, manifest_path)
        _revalidate_destination(
            destination,
            existed=destination_existed,
            identity=destination_identity,
        )

        if destination_existed:
            previous_destination = staging_container / ".previous-empty-destination"
            os.replace(destination, previous_destination)
            destination_moved = True

        try:
            os.replace(staging_data, destination)
            promoted = True
        except BaseException:
            if destination_moved and previous_destination is not None:
                _restore_empty_destination(previous_destination, destination)
                destination_moved = False
            raise

        try:
            archive_path.unlink()
        except BaseException:
            _roll_back_promotion(
                destination=destination,
                staging_data=staging_data,
                previous_destination=previous_destination if destination_moved else None,
            )
            promoted = False
            destination_moved = False
            raise

        if destination_moved and previous_destination is not None:
            previous_destination.rmdir()
            destination_moved = False

        return IngestResult(
            data_root=destination,
            file_count=int(audited["file_count"]),
            total_bytes=int(audited["total_bytes"]),
        )
    finally:
        if destination_moved and previous_destination is not None and not promoted:
            _restore_empty_destination(previous_destination, destination)
        if staging_container is not None:
            shutil.rmtree(staging_container, ignore_errors=False)


def _require_regular_file(path: Path, label: str) -> None:
    _reject_link_or_junction(path, label)
    if not path.exists() or not path.is_file():
        raise DataValidationError(f"{label} does not exist or is not a regular file: {path}")


def _load_expected_files(manifest_path: Path) -> tuple[dict[str, _ExpectedFile], str]:
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"Audit manifest is not readable JSON: {manifest_path} ({exc})") from exc

    if not isinstance(manifest, dict):
        raise DataValidationError(f"Audit manifest must contain a JSON object: {manifest_path}")

    dataset_id = manifest.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise DataValidationError("Audit manifest dataset_id must be a non-empty string")
    canonical_dataset_id = _safe_member_path(
        dataset_id,
        is_directory=False,
        context="Audit manifest dataset_id",
    )
    if "/" in canonical_dataset_id:
        raise DataValidationError("Audit manifest dataset_id must be a safe directory basename")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise DataValidationError("Audit manifest files must be a non-empty list")

    expected: dict[str, _ExpectedFile] = {}
    folded_paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise DataValidationError(f"Audit manifest files[{index}] must be an object")
        path = raw_entry.get("path")
        size = raw_entry.get("size")
        digest = raw_entry.get("sha256")
        if not isinstance(path, str):
            raise DataValidationError(f"Audit manifest files[{index}].path must be a string")
        canonical = _safe_member_path(path, is_directory=False, context="Audit manifest path")
        if canonical != path:
            raise DataValidationError(f"Audit manifest path is not canonical: {path!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise DataValidationError(f"Audit manifest size must be a non-negative integer: {path}")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise DataValidationError(f"Audit manifest sha256 is invalid: {path}")
        if path in expected or path.casefold() in folded_paths:
            raise DataValidationError(f"Audit manifest contains duplicate or case-colliding path: {path}")
        expected[path] = _ExpectedFile(path=path, size=size)
        folded_paths.add(path.casefold())

    file_count = manifest.get("file_count")
    total_bytes = manifest.get("total_bytes")
    if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count != len(expected):
        raise DataValidationError("Audit manifest file_count does not match its files list")
    expected_total = sum(item.size for item in expected.values())
    if not isinstance(total_bytes, int) or isinstance(total_bytes, bool) or total_bytes != expected_total:
        raise DataValidationError("Audit manifest total_bytes does not match its files list")

    return expected, dataset_id


def _validate_destination(destination: Path, *, dataset_id: str) -> tuple[bool, tuple[int, int] | None]:
    if destination.name != dataset_id:
        raise DataValidationError(
            f"Destination basename must match manifest dataset_id {dataset_id!r}: {destination}"
        )
    if not destination.name or destination.parent == destination:
        raise DataValidationError(f"Destination is not a safe dataset directory: {destination}")

    parent = destination.parent
    _reject_link_or_junction(parent, "Destination parent")
    if not parent.exists() or not parent.is_dir():
        raise DataValidationError(f"Destination parent does not exist or is not a directory: {parent}")

    _reject_link_or_junction(destination, "Destination")
    if not destination.exists():
        return False, None
    if not destination.is_dir() or next(destination.iterdir(), None) is not None:
        raise DataValidationError(f"Destination must not exist or must be empty: {destination}")
    metadata = destination.stat()
    return True, (metadata.st_dev, metadata.st_ino)


def _revalidate_destination(
    destination: Path,
    *,
    existed: bool,
    identity: tuple[int, int] | None,
) -> None:
    _reject_link_or_junction(destination, "Destination")
    if not existed:
        if destination.exists():
            raise DataValidationError(f"Destination changed while ingest was running: {destination}")
        return
    if not destination.exists() or not destination.is_dir():
        raise DataValidationError(f"Destination changed while ingest was running: {destination}")
    metadata = destination.stat()
    if (metadata.st_dev, metadata.st_ino) != identity or next(destination.iterdir(), None) is not None:
        raise DataValidationError(f"Destination changed while ingest was running: {destination}")


def _open_archive(archive_path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(archive_path, mode="r")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise DataValidationError(f"Dataset archive is not a readable ZIP: {archive_path} ({exc})") from exc


def _build_archive_plan(
    source_zip: zipfile.ZipFile,
    archive_path: Path,
    expected: dict[str, _ExpectedFile],
) -> _ArchivePlan:
    infos = source_zip.infolist()
    if not infos:
        raise DataValidationError("ZIP file members do not match manifest: archive is empty")

    duplicate_names = sorted(
        name for name, count in Counter(info.orig_filename for info in infos).items() if count > 1
    )
    if duplicate_names:
        raise DataValidationError(f"Duplicate ZIP member: {duplicate_names[0]!r}")

    canonical_infos: list[tuple[str, zipfile.ZipInfo]] = []
    folded_names: dict[str, str] = {}
    with archive_path.open("rb") as raw_archive:
        for info in infos:
            local_flags, local_name = _local_header_metadata(raw_archive, info)
            if info.flag_bits & 0x1 or local_flags & 0x1:
                raise DataValidationError(f"Encrypted ZIP member is not allowed: {info.orig_filename!r}")
            if b"\\" in local_name:
                raise DataValidationError(f"Unsafe ZIP member contains a backslash: {info.orig_filename!r}")
            _reject_link_member(info)
            canonical = _safe_member_path(
                info.orig_filename,
                is_directory=info.is_dir(),
                context="Unsafe ZIP member",
            )
            folded = canonical.casefold()
            if folded in folded_names:
                raise DataValidationError(
                    f"Duplicate ZIP member after case normalization: "
                    f"{folded_names[folded]!r}, {info.orig_filename!r}"
                )
            folded_names[folded] = info.orig_filename
            canonical_infos.append((canonical, info))

    file_infos = [(name, info) for name, info in canonical_infos if not info.is_dir()]
    actual_paths = {name for name, _ in file_infos}
    expected_paths = set(expected)
    wrapper: str | None = None
    if actual_paths == expected_paths:
        mapped_infos = {name: info for name, info in file_infos}
    else:
        first_components = {PurePosixPath(name).parts[0] for name, _ in file_infos}
        if len(first_components) != 1 or any(len(PurePosixPath(name).parts) < 2 for name, _ in file_infos):
            raise DataValidationError("ZIP file members do not match manifest")
        wrapper = next(iter(first_components))
        stripped = [(PurePosixPath(name).relative_to(wrapper).as_posix(), info) for name, info in file_infos]
        stripped_paths = {name for name, _ in stripped}
        if stripped_paths != expected_paths:
            raise DataValidationError("ZIP file members do not match manifest")
        mapped_infos = {name: info for name, info in stripped}

    if len(mapped_infos) != len(expected):
        raise DataValidationError("ZIP file members do not match manifest")
    for path, expected_file in expected.items():
        info = mapped_infos[path]
        if info.file_size != expected_file.size:
            raise DataValidationError(
                f"ZIP member size does not match manifest for {path}: "
                f"archive={info.file_size}, manifest={expected_file.size}"
            )

    _validate_directory_members(canonical_infos, expected_paths, wrapper)
    return _ArchivePlan(members=mapped_infos, wrapper=wrapper)


def _local_header_metadata(raw_archive, info: zipfile.ZipInfo) -> tuple[int, bytes]:
    try:
        raw_archive.seek(info.header_offset)
        header = raw_archive.read(30)
    except OSError as exc:
        raise DataValidationError(f"Cannot read ZIP local header for {info.filename!r}: {exc}") from exc
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise DataValidationError(f"Invalid ZIP local header for member: {info.filename!r}")
    flags = struct.unpack_from("<H", header, 6)[0]
    name_size = struct.unpack_from("<H", header, 26)[0]
    local_name = raw_archive.read(name_size)
    if len(local_name) != name_size:
        raise DataValidationError(f"Truncated ZIP local header for member: {info.filename!r}")
    return flags, local_name


def _reject_link_member(info: zipfile.ZipInfo) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise DataValidationError(f"Link ZIP member is not allowed: {info.filename!r}")
    if info.create_system != 3:
        return
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise DataValidationError(f"Special ZIP member is not allowed: {info.filename!r}")
    if (file_type == stat.S_IFDIR) != info.is_dir() and file_type != 0:
        raise DataValidationError(f"Unsafe ZIP member has inconsistent file type: {info.filename!r}")


def _safe_member_path(name: str, *, is_directory: bool, context: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise DataValidationError(f"{context}: {name!r}")
    if is_directory:
        if not name.endswith("/") or name.endswith("//"):
            raise DataValidationError(f"{context}: {name!r}")
        canonical = name[:-1]
    else:
        if name.endswith("/"):
            raise DataValidationError(f"{context}: {name!r}")
        canonical = name
    if not canonical or canonical.startswith("/") or "//" in canonical:
        raise DataValidationError(f"{context}: {name!r}")

    posix_path = PurePosixPath(canonical)
    windows_path = PureWindowsPath(canonical)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.as_posix() != canonical
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise DataValidationError(f"{context}: {name!r}")
    for part in posix_path.parts:
        _validate_path_component(part, context=context)
    return canonical


def _validate_path_component(component: str, *, context: str) -> None:
    if (
        not component
        or component != component.strip()
        or component.endswith(".")
        or any(character in _WINDOWS_FORBIDDEN_CHARACTERS for character in component)
        or any(ord(character) < 32 for character in component)
        or component.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise DataValidationError(f"{context}: unsafe path component {component!r}")


def _validate_directory_members(
    canonical_infos: list[tuple[str, zipfile.ZipInfo]],
    expected_paths: set[str],
    wrapper: str | None,
) -> None:
    expected_directories = {
        PurePosixPath(*PurePosixPath(path).parts[:index]).as_posix()
        for path in expected_paths
        for index in range(1, len(PurePosixPath(path).parts))
    }
    for canonical, info in canonical_infos:
        if not info.is_dir():
            continue
        if info.file_size != 0:
            raise DataValidationError(f"ZIP directory member must be empty: {info.filename!r}")
        relative = canonical
        if wrapper is not None:
            if canonical == wrapper:
                continue
            prefix = f"{wrapper}/"
            if not canonical.startswith(prefix):
                raise DataValidationError("ZIP directory members do not match manifest")
            relative = canonical[len(prefix) :]
        if relative not in expected_directories:
            raise DataValidationError("ZIP directory members do not match manifest")


def _extract_plan(
    source_zip: zipfile.ZipFile,
    plan: _ArchivePlan,
    staging_data: Path,
    expected: dict[str, _ExpectedFile],
) -> None:
    for relative_path in sorted(expected):
        expected_file = expected[relative_path]
        info = plan.members[relative_path]
        output_path = staging_data.joinpath(*PurePosixPath(relative_path).parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        try:
            with source_zip.open(info, mode="r") as source, output_path.open("xb") as output:
                while chunk := source.read(_COPY_CHUNK_SIZE):
                    copied += len(chunk)
                    if copied > expected_file.size:
                        raise DataValidationError(
                            f"Extracted ZIP member exceeds manifest size for {relative_path}"
                        )
                    output.write(chunk)
        except DataValidationError:
            raise
        except (OSError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
            raise DataValidationError(f"Cannot extract ZIP member {relative_path}: {exc}") from exc
        if copied != expected_file.size:
            raise DataValidationError(
                f"Extracted ZIP member size does not match manifest for {relative_path}: "
                f"archive={copied}, manifest={expected_file.size}"
            )


def _roll_back_promotion(
    *,
    destination: Path,
    staging_data: Path,
    previous_destination: Path | None,
) -> None:
    try:
        os.replace(destination, staging_data)
        if previous_destination is not None:
            _restore_empty_destination(previous_destination, destination)
    except BaseException as exc:
        raise RuntimeError(
            f"Archive deletion failed and the destination could not be rolled back safely: {destination}"
        ) from exc


def _restore_empty_destination(previous_destination: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Cannot restore original empty destination because its path is occupied: {destination}")
    os.rename(previous_destination, destination)


def _reject_link_or_junction(path: Path, label: str) -> None:
    is_junction = getattr(path, "is_junction", None)
    if path.is_symlink() or (callable(is_junction) and is_junction()):
        raise DataValidationError(f"{label} must not be a symlink or junction: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely install a VAI dataset ZIP after exact manifest verification.",
    )
    parser.add_argument("--archive", type=Path, required=True, help="Dataset ZIP to ingest and delete on success.")
    parser.add_argument("--data-root", type=Path, required=True, help="Nonexistent or empty destination directory.")
    parser.add_argument("--manifest", type=Path, required=True, help="Audit manifest generated for the dataset.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = ingest_archive(args.archive, args.data_root, args.manifest)
    print(f"Ingested {result.file_count} files ({result.total_bytes} bytes) into {result.data_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
