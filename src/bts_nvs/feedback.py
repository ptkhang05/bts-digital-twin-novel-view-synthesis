from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bts_nvs import ledger
from bts_nvs.exceptions import DataValidationError
from bts_nvs.package import create_submission_zip
from bts_nvs.validate_submission import validate_submission

DEFAULT_CANDIDATE_DIR = Path("outputs/candidate")
DEFAULT_BEST_DIR = Path("outputs/best")
DEFAULT_ZIP_PATH = Path("submission.zip")


@dataclass(frozen=True)
class _LedgerState:
    existed: bool
    size: int
    parent_existed: bool


def process_feedback(
    *,
    submission_id: str,
    dataset_id: str,
    score: float,
    data_root: Path | str,
    candidate_dir: Path | str = DEFAULT_CANDIDATE_DIR,
    best_dir: Path | str = DEFAULT_BEST_DIR,
    zip_path: Path | str = DEFAULT_ZIP_PATH,
    ledger_path: Path | str = ledger.DEFAULT_LEDGER_PATH,
    gpu_time_seconds: float | None = None,
    zip_size_bytes: int | None = None,
    zip_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
    git_commit: str | None = None,
    container_image_digest: str | None = None,
    command: str | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    seed: int | None = None,
    iterations: int | None = None,
    distortion: str | None = None,
    jpeg_quality: int | None = None,
    hardware: dict[str, Any] | None = None,
    hypothesis: str | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    """Record BTC feedback and transition the fixed candidate/best lifecycle.

    Directory renames and ZIP replacement are rollback-protected within this process. Callers
    must ensure only one feedback command operates on these paths at a time.
    """

    candidate_path = Path(candidate_dir)
    best_path = Path(best_dir)
    submitted_zip = Path(zip_path)
    history_path = Path(ledger_path)
    dataset_path = Path(data_root)

    proposed = _validated_record(
        submission_id=submission_id,
        dataset_id=dataset_id,
        score=score,
        gpu_time_seconds=gpu_time_seconds,
        zip_size_bytes=zip_size_bytes,
        zip_sha256=zip_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        git_commit=git_commit,
        container_image_digest=container_image_digest,
        command=command,
        config=config,
        metrics=metrics,
        seed=seed,
        iterations=iterations,
        distortion=distortion,
        jpeg_quality=jpeg_quality,
        hardware=hardware,
        hypothesis=hypothesis,
        next_action=next_action,
    )

    _assert_safe_path(history_path, "ledger")
    existing_records = ledger.load_records(history_path)
    if any(record["submission_id"] == proposed["submission_id"] for record in existing_records):
        raise ledger.LedgerValidationError(f"submission_id already exists: {proposed['submission_id']}")

    staging_path = _validate_lifecycle_paths(
        data_root=dataset_path,
        candidate_dir=candidate_path,
        best_dir=best_path,
        zip_path=submitted_zip,
        ledger_path=history_path,
    )
    actual_zip_size, actual_zip_sha256 = _validate_zip_identity(dataset_path, submitted_zip)
    supplied_zip_size = proposed.get("zip_size_bytes")
    if supplied_zip_size is not None and int(supplied_zip_size) != actual_zip_size:
        raise DataValidationError(
            f"zip_size_bytes does not match submitted ZIP: supplied {supplied_zip_size}, actual {actual_zip_size}"
        )
    supplied_zip_sha256 = proposed.get("zip_sha256")
    if supplied_zip_sha256 is not None and str(supplied_zip_sha256).lower() != actual_zip_sha256:
        raise DataValidationError(
            f"zip_sha256 does not match submitted ZIP: supplied {supplied_zip_sha256}, actual {actual_zip_sha256}"
        )
    proposed["zip_size_bytes"] = actual_zip_size
    proposed["zip_sha256"] = actual_zip_sha256
    proposed = ledger._validate_record(proposed)
    _verify_candidate_matches_submitted_zip(
        data_root=dataset_path,
        candidate_dir=candidate_path,
        submitted_zip=submitted_zip,
        submitted_sha256=actual_zip_sha256,
        staging_dir=staging_path,
    )
    proposed["validator_status"] = "pass"

    incumbent = ledger.best_submission(dataset_id, ledger_path=history_path)
    proposed["score_delta_vs_best"] = (
        None
        if incumbent is None
        else float(proposed["leaderboard_score"]) - float(incumbent["leaderboard_score"])
    )
    promote = incumbent is None or ledger._candidate_is_better(proposed, incumbent)
    if incumbent is not None and not best_path.is_dir():
        raise DataValidationError(f"Ledger has a best submission but best directory is missing: {best_path}")
    if not promote and not (best_path / "rendered").is_dir():
        raise DataValidationError(
            f"Rejected candidate cannot be restored because best/rendered is missing: {best_path}"
        )

    proposed["decision"] = "promote" if promote else "reject"
    final_record = ledger._validate_record(proposed, add_defaults=True)
    ledger_state = _capture_ledger_state(history_path)
    if promote:
        return _promote_candidate(
            candidate_dir=candidate_path,
            best_dir=best_path,
            zip_path=submitted_zip,
            record=final_record,
            ledger_path=history_path,
            ledger_state=ledger_state,
            staging_dir=staging_path,
        )
    return _reject_candidate(
        data_root=dataset_path,
        candidate_dir=candidate_path,
        best_dir=best_path,
        zip_path=submitted_zip,
        record=final_record,
        ledger_path=history_path,
        ledger_state=ledger_state,
        staging_dir=staging_path,
    )


def _validated_record(
    *,
    submission_id: str,
    dataset_id: str,
    score: float,
    gpu_time_seconds: float | None,
    zip_size_bytes: int | None,
    zip_sha256: str | None,
    dataset_manifest_sha256: str | None,
    git_commit: str | None,
    container_image_digest: str | None,
    command: str | None,
    config: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    seed: int | None,
    iterations: int | None,
    distortion: str | None,
    jpeg_quality: int | None,
    hardware: dict[str, Any] | None,
    hypothesis: str | None,
    next_action: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "submission_id": submission_id,
        "dataset_id": dataset_id,
        "leaderboard_score": score,
        "feedback_status": "complete",
    }
    optional = {
        "gpu_time_seconds": gpu_time_seconds,
        "zip_size_bytes": zip_size_bytes,
        "zip_sha256": zip_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "git_commit": git_commit,
        "container_image_digest": container_image_digest,
        "command": command,
        "config": config,
        "metrics": metrics,
        "seed": seed,
        "iterations": iterations,
        "distortion": distortion,
        "jpeg_quality": jpeg_quality,
        "hardware": hardware,
        "hypothesis": hypothesis,
        "next_action": next_action,
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    return ledger._validate_record(record)


def _validate_zip_identity(data_root: Path, zip_path: Path) -> tuple[int, str]:
    validation = validate_submission(data_root=data_root, submission=zip_path)
    validation.raise_for_errors()

    digest = hashlib.sha256()
    size = 0
    with zip_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _verify_candidate_matches_submitted_zip(
    *,
    data_root: Path,
    candidate_dir: Path,
    submitted_zip: Path,
    submitted_sha256: str,
    staging_dir: Path,
) -> None:
    """Prove that the live candidate is the artifact whose ZIP received feedback."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=staging_dir,
        prefix=f".{submitted_zip.name}.feedback-identity-",
        suffix=".zip",
    )
    os.close(descriptor)
    candidate_zip = Path(temporary_name)
    try:
        packaged = create_submission_zip(
            data_root=data_root,
            submission=candidate_dir / "rendered",
            output=candidate_zip,
            staging_dir=staging_dir,
        )
        if packaged.sha256 == submitted_sha256:
            return
        if _zip_members_are_identical(candidate_zip, submitted_zip):
            return
        raise DataValidationError(
            "Candidate rendered output does not match the submitted ZIP; "
            "refusing to update candidate, best, ZIP or ledger"
        )
    finally:
        candidate_zip.unlink(missing_ok=True)


def _zip_members_are_identical(first: Path, second: Path) -> bool:
    """Compare exact member names and bytes while ignoring ZIP container metadata."""

    try:
        with zipfile.ZipFile(first) as first_archive, zipfile.ZipFile(second) as second_archive:
            first_members = {info.filename: info for info in first_archive.infolist()}
            second_members = {info.filename: info for info in second_archive.infolist()}
            if set(first_members) != set(second_members):
                return False
            for name in sorted(first_members):
                first_info = first_members[name]
                second_info = second_members[name]
                if first_info.file_size != second_info.file_size:
                    return False
                with first_archive.open(first_info) as first_stream, second_archive.open(second_info) as second_stream:
                    while True:
                        first_chunk = first_stream.read(1024 * 1024)
                        second_chunk = second_stream.read(1024 * 1024)
                        if first_chunk != second_chunk:
                            return False
                        if not first_chunk:
                            break
    except (EOFError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DataValidationError(f"Could not compare candidate and submitted ZIP contents: {exc}") from exc
    return True


def _validate_lifecycle_paths(
    *,
    data_root: Path,
    candidate_dir: Path,
    best_dir: Path,
    zip_path: Path,
    ledger_path: Path,
) -> Path:
    for path, label in (
        (data_root, "data root"),
        (candidate_dir, "candidate directory"),
        (best_dir, "best directory"),
        (zip_path, "submission ZIP"),
        (ledger_path, "ledger"),
    ):
        _assert_safe_path(path, label)

    if not data_root.is_dir():
        raise DataValidationError(f"Data root does not exist: {data_root}")
    if not candidate_dir.is_dir():
        raise DataValidationError(f"Candidate directory does not exist: {candidate_dir}")
    if not (candidate_dir / "rendered").is_dir():
        raise DataValidationError(f"Candidate must contain a rendered directory: {candidate_dir}")
    if not zip_path.is_file():
        raise DataValidationError(f"Submitted ZIP does not exist: {zip_path}")
    if best_dir.exists() and not best_dir.is_dir():
        raise DataValidationError(f"Best path is not a directory: {best_dir}")
    if ledger_path.exists() and not ledger_path.is_file():
        raise DataValidationError(f"Ledger path is not a file: {ledger_path}")
    if not best_dir.parent.is_dir():
        raise DataValidationError(f"Best directory parent does not exist: {best_dir.parent}")

    _assert_safe_tree(candidate_dir, "candidate directory")
    if best_dir.exists():
        _assert_safe_tree(best_dir, "best directory")

    candidate_abs = _absolute(candidate_dir)
    best_abs = _absolute(best_dir)
    if _overlaps(candidate_abs, best_abs):
        raise DataValidationError("Candidate and best directories must be separate, non-nested paths")
    if candidate_abs.parent != best_abs.parent:
        raise DataValidationError("Candidate and best directories must share one outputs parent")
    for path, label in ((zip_path, "submission ZIP"), (ledger_path, "ledger"), (data_root, "data root")):
        absolute = _absolute(path)
        if _overlaps(absolute, candidate_abs) or _overlaps(absolute, best_abs):
            raise DataValidationError(f"{label} must not overlap candidate or best directories")

    if candidate_dir.stat().st_dev != best_dir.parent.stat().st_dev:
        raise DataValidationError("Candidate and best directories must be on the same filesystem")

    staging_dir = candidate_abs.parent / ".staging"
    _assert_safe_path(staging_dir, "feedback staging directory")
    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise DataValidationError(f"Feedback staging path is not a directory: {staging_dir}")
        _assert_safe_tree(staging_dir, "feedback staging directory")
    else:
        staging_dir.mkdir()
    return staging_dir


def _promote_candidate(
    *,
    candidate_dir: Path,
    best_dir: Path,
    zip_path: Path,
    record: dict[str, Any],
    ledger_path: Path,
    ledger_state: _LedgerState,
    staging_dir: Path,
) -> dict[str, Any]:
    backup = _unused_staging_path(staging_dir, "feedback-best-backup") if best_dir.exists() else None
    best_was_moved = False
    candidate_was_moved = False
    try:
        if backup is not None:
            os.replace(best_dir, backup)
            best_was_moved = True
        os.replace(candidate_dir, best_dir)
        candidate_was_moved = True
        result = ledger.append_record(record, ledger_path=ledger_path)
    except BaseException:
        rollback_errors: list[BaseException] = []
        if candidate_was_moved:
            _attempt_rollback(lambda: os.replace(best_dir, candidate_dir), rollback_errors)
        if best_was_moved and backup is not None:
            _attempt_rollback(lambda: os.replace(backup, best_dir), rollback_errors)
        _attempt_rollback(lambda: _restore_ledger(ledger_path, ledger_state), rollback_errors)
        if rollback_errors:
            raise RuntimeError("Feedback promotion failed and rollback was incomplete") from rollback_errors[0]
        raise

    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return result


def _reject_candidate(
    *,
    data_root: Path,
    candidate_dir: Path,
    best_dir: Path,
    zip_path: Path,
    record: dict[str, Any],
    ledger_path: Path,
    ledger_state: _LedgerState,
    staging_dir: Path,
) -> dict[str, Any]:
    zip_backup = _copy_to_backup(zip_path, staging_dir=staging_dir)
    rejected = _unused_staging_path(staging_dir, "feedback-rejected-candidate")
    candidate_was_moved = False
    try:
        create_submission_zip(
            data_root=data_root,
            submission=best_dir / "rendered",
            output=zip_path,
            staging_dir=staging_dir,
        )
        os.replace(candidate_dir, rejected)
        candidate_was_moved = True
        result = ledger.append_record(record, ledger_path=ledger_path)
    except BaseException:
        rollback_errors: list[BaseException] = []
        if candidate_was_moved:
            _attempt_rollback(lambda: os.replace(rejected, candidate_dir), rollback_errors)
        _attempt_rollback(lambda: os.replace(zip_backup, zip_path), rollback_errors)
        _attempt_rollback(lambda: _restore_ledger(ledger_path, ledger_state), rollback_errors)
        if rollback_errors:
            raise RuntimeError("Feedback rejection failed and rollback was incomplete") from rollback_errors[0]
        raise

    shutil.rmtree(rejected, ignore_errors=True)
    try:
        zip_backup.unlink(missing_ok=True)
    except OSError:
        # The lifecycle and ledger have committed; a hidden backup is safer than reporting
        # failure after the rejected candidate has already been removed.
        pass
    return result


def _copy_to_backup(path: Path, *, staging_dir: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=staging_dir,
        prefix=".feedback-submission-backup-",
        suffix=".tmp",
    )
    backup = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as destination, path.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        backup.unlink(missing_ok=True)
        raise
    return backup


def _capture_ledger_state(path: Path) -> _LedgerState:
    return _LedgerState(
        existed=path.exists(),
        size=path.stat().st_size if path.exists() else 0,
        parent_existed=path.parent.exists(),
    )


def _restore_ledger(path: Path, state: _LedgerState) -> None:
    if state.existed:
        with path.open("r+b") as handle:
            handle.truncate(state.size)
            handle.flush()
            os.fsync(handle.fileno())
        return
    path.unlink(missing_ok=True)
    if not state.parent_existed:
        try:
            path.parent.rmdir()
        except OSError:
            pass


def _unused_staging_path(staging_dir: Path, purpose: str) -> Path:
    while True:
        candidate = staging_dir / f".{purpose}-{uuid.uuid4().hex}"
        if not candidate.exists():
            return candidate


def _assert_safe_path(path: Path, label: str) -> None:
    current = _absolute(path)
    for component in (current, *current.parents):
        if (component.exists() or component.is_symlink()) and _is_link_or_junction(component):
            raise DataValidationError(f"{label} uses an unsafe symlink or junction: {component}")


def _assert_safe_tree(root: Path, label: str) -> None:
    with os.scandir(root) as entries:
        for entry in entries:
            entry_path = Path(entry.path)
            if entry.is_symlink() or _is_junction(entry_path):
                raise DataValidationError(f"{label} contains an unsafe symlink or junction: {entry_path}")
            if entry.is_dir(follow_symlinks=False):
                _assert_safe_tree(entry_path, label)


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or _is_junction(path)


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker is not None and checker())


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _overlaps(first: Path, second: Path) -> bool:
    first_text = os.path.normcase(str(first))
    second_text = os.path.normcase(str(second))
    try:
        common = os.path.normcase(os.path.commonpath((first_text, second_text)))
    except ValueError:
        return False
    return common in {first_text, second_text}


def _attempt_rollback(action, errors: list[BaseException]) -> None:
    try:
        action()
    except BaseException as exc:  # pragma: no cover - catastrophic filesystem failure
        errors.append(exc)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record BTC feedback and update the fixed candidate/best lifecycle.")
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--score", type=ledger._finite_float_argument, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        "--candidate-dir",
        dest="candidate_dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
    )
    parser.add_argument("--best", "--best-dir", dest="best_dir", type=Path, default=DEFAULT_BEST_DIR)
    parser.add_argument("--zip", dest="zip_path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--ledger", dest="ledger_path", type=Path, default=ledger.DEFAULT_LEDGER_PATH)
    parser.add_argument("--gpu-time-seconds", type=ledger._nonnegative_float_argument)
    parser.add_argument("--zip-size-bytes", type=ledger._nonnegative_int_argument)
    parser.add_argument("--zip-sha256")
    parser.add_argument("--dataset-manifest-sha256")
    parser.add_argument("--git-commit")
    parser.add_argument("--container-image-digest")
    parser.add_argument("--command")
    parser.add_argument("--config-json", type=ledger._json_object_argument)
    parser.add_argument("--metrics-json", type=ledger._json_object_argument)
    parser.add_argument("--seed", type=ledger._nonnegative_int_argument)
    parser.add_argument("--iterations", type=ledger._nonnegative_int_argument)
    parser.add_argument("--distortion", choices=("auto", "on", "off"))
    parser.add_argument("--jpeg-quality", type=ledger._jpeg_quality_argument)
    parser.add_argument("--hardware-json", type=ledger._json_object_argument)
    parser.add_argument("--hypothesis")
    parser.add_argument("--next-action")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = process_feedback(
            submission_id=args.submission_id,
            dataset_id=args.dataset_id,
            score=args.score,
            data_root=args.data_root,
            candidate_dir=args.candidate_dir,
            best_dir=args.best_dir,
            zip_path=args.zip_path,
            ledger_path=args.ledger_path,
            gpu_time_seconds=args.gpu_time_seconds,
            zip_size_bytes=args.zip_size_bytes,
            zip_sha256=args.zip_sha256,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            git_commit=args.git_commit,
            container_image_digest=args.container_image_digest,
            command=args.command,
            config=args.config_json,
            metrics=args.metrics_json,
            seed=args.seed,
            iterations=args.iterations,
            distortion=args.distortion,
            jpeg_quality=args.jpeg_quality,
            hardware=args.hardware_json,
            hypothesis=args.hypothesis,
            next_action=args.next_action,
        )
    except (DataValidationError, ledger.LedgerValidationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
