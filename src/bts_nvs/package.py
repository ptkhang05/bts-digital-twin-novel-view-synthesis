from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.exceptions import DataValidationError
from bts_nvs.validate_submission import validate_submission


@dataclass(frozen=True)
class PackagedSubmission:
    zip_path: Path
    scene_count: int
    image_count: int
    sha256: str


def create_submission_zip(
    submission: Path | str,
    output: Path | str,
    *,
    data_root: Path | str,
    staging_dir: Path | str | None = None,
) -> PackagedSubmission:
    """Validate, package, revalidate, and atomically publish one submission ZIP."""

    submission_dir = Path(submission)
    output_path = Path(output)
    data_path = Path(data_root)
    if output_path.suffix.lower() != ".zip":
        raise DataValidationError(f"Submission output must use a .zip suffix: {output_path}")
    if output_path.exists() and output_path.is_dir():
        raise DataValidationError(f"Submission output path is a directory: {output_path}")
    if not submission_dir.exists() or not submission_dir.is_dir():
        raise DataValidationError(f"Submission directory does not exist: {submission_dir}")

    submission_resolved = submission_dir.resolve(strict=True)
    output_resolved = output_path.resolve(strict=False)
    if output_resolved == submission_resolved or output_resolved.is_relative_to(submission_resolved):
        raise DataValidationError("Submission ZIP output must be outside the submission directory")

    folder_validation = validate_submission(data_root=data_path, submission=submission_dir)
    folder_validation.raise_for_errors()
    scenes = discover_scene_outputs(submission_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(staging_dir) if staging_dir is not None else output_path.parent
    staging_resolved = staging_path.resolve(strict=False)
    if staging_resolved == submission_resolved or staging_resolved.is_relative_to(submission_resolved):
        raise DataValidationError("Submission staging directory must be outside the submission directory")
    staging_path.mkdir(parents=True, exist_ok=True)
    if staging_path.stat().st_dev != output_path.parent.stat().st_dev:
        raise DataValidationError("Submission staging and output directories must be on the same filesystem")

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=staging_path,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.zip",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        _write_zip_archive(temporary, scenes)

        zip_validation = validate_submission(data_root=data_path, submission=temporary)
        zip_validation.raise_for_errors()
        archive_sha256 = _sha256_file(temporary)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return PackagedSubmission(
        zip_path=output_path,
        scene_count=folder_validation.scene_count,
        image_count=folder_validation.image_count,
        sha256=archive_sha256,
    )


def discover_scene_outputs(submission_dir: Path) -> list[tuple[str, list[Path]]]:
    scenes: list[tuple[str, list[Path]]] = []
    for scene_dir in sorted(submission_dir.iterdir(), key=lambda candidate: candidate.name):
        if not scene_dir.is_dir():
            continue
        images = sorted(
            (path for path in scene_dir.iterdir() if path.is_file()),
            key=lambda candidate: candidate.name,
        )
        if images:
            scenes.append((scene_dir.name, images))
    return scenes


def _write_zip_archive(zip_path: Path, scenes: list[tuple[str, list[Path]]]) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for scene_id, images in scenes:
            for image_path in images:
                member = zipfile.ZipInfo(f"{scene_id}/{image_path.name}", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                with image_path.open("rb") as source, archive.open(member, mode="w", force_zip64=True) as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strictly validate and atomically package VAI/BTC predictions.")
    parser.add_argument("--data-root", type=Path, required=True, help="VAI dataset root containing target CSV files.")
    parser.add_argument(
        "--submission",
        type=Path,
        required=True,
        help="Directory containing exact scene/image outputs.",
    )
    parser.add_argument("--out", type=Path, required=True, help="ZIP file path to atomically replace after validation.")
    parser.add_argument("--staging-dir", type=Path, help="Optional same-filesystem directory for the temporary ZIP.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = create_submission_zip(
        data_root=args.data_root,
        submission=args.submission,
        output=args.out,
        staging_dir=args.staging_dir,
    )
    print(
        f"Wrote {result.zip_path} with {result.image_count} JPEGs from {result.scene_count} scenes; "
        f"sha256={result.sha256}"
    )


if __name__ == "__main__":
    main()
