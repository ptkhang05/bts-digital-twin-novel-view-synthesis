from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_PATH = Path("experiments/submission_history.jsonl")
MAX_RECORD_BYTES = 1_000_000
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class LedgerValidationError(ValueError):
    """Raised when a ledger record or ledger file violates its contract."""


def load_records(ledger_path: Path | str = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Load and validate every immutable submission record in a JSONL ledger."""

    path = Path(ledger_path)
    if not path.exists():
        return []
    if not path.is_file():
        raise LedgerValidationError(f"Ledger path is not a file: {path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
                raise LedgerValidationError(f"Ledger line {line_number} exceeds {MAX_RECORD_BYTES} bytes")
            try:
                raw = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, LedgerValidationError) as exc:
                raise LedgerValidationError(f"Invalid JSON on ledger line {line_number}: {exc}") from exc
            record = _validate_record(raw)
            submission_id = record["submission_id"]
            if submission_id in seen_ids:
                raise LedgerValidationError(
                    f"Duplicate submission_id in ledger on line {line_number}: {submission_id}"
                )
            seen_ids.add(submission_id)
            records.append(record)
    return records


def append_record(
    record: dict[str, Any],
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Validate and append one canonical record; existing lines are never rewritten."""

    path = Path(ledger_path)
    validated = _validate_record(record, add_defaults=True)
    existing = load_records(path)
    if any(item["submission_id"] == validated["submission_id"] for item in existing):
        raise LedgerValidationError(f"submission_id already exists: {validated['submission_id']}")

    serialized = json.dumps(validated, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > MAX_RECORD_BYTES:
        raise LedgerValidationError(f"Ledger record exceeds {MAX_RECORD_BYTES} bytes")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return validated


def add_feedback(
    *,
    submission_id: str,
    dataset_id: str,
    score: float,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
    gpu_time_seconds: float | None = None,
    zip_size_bytes: int | None = None,
    zip_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
    git_commit: str | None = None,
    container_image_digest: str | None = None,
    command: str | None = None,
    config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    hypothesis: str | None = None,
    decision: str | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    """Append complete BTC feedback plus optional reproducibility metadata."""

    record: dict[str, Any] = {
        "submission_id": submission_id,
        "dataset_id": dataset_id,
        "leaderboard_score": score,
        "feedback_status": "complete",
    }
    optional_fields = {
        "gpu_time_seconds": gpu_time_seconds,
        "zip_size_bytes": zip_size_bytes,
        "zip_sha256": zip_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "git_commit": git_commit,
        "container_image_digest": container_image_digest,
        "command": command,
        "config": config,
        "metrics": metrics,
        "hypothesis": hypothesis,
        "decision": decision,
        "next_action": next_action,
    }
    record.update({key: value for key, value in optional_fields.items() if value is not None})
    return append_record(record, ledger_path=ledger_path)


def best_submission(
    dataset_id: str,
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
) -> dict[str, Any] | None:
    """Return the best complete official result for one dataset, or ``None``."""

    _validate_identifier(dataset_id, "dataset_id")
    best: dict[str, Any] | None = None
    for record in load_records(ledger_path):
        if record["dataset_id"] != dataset_id or record.get("leaderboard_score") is None:
            continue
        if best is None or _candidate_is_better(record, best):
            best = record
    return deepcopy(best) if best is not None else None


def _candidate_is_better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_score = float(candidate["leaderboard_score"])
    incumbent_score = float(incumbent["leaderboard_score"])
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score

    candidate_gpu = candidate.get("gpu_time_seconds")
    incumbent_gpu = incumbent.get("gpu_time_seconds")
    if candidate_gpu is None or incumbent_gpu is None:
        return False
    if float(candidate_gpu) != float(incumbent_gpu):
        return float(candidate_gpu) < float(incumbent_gpu)

    candidate_size = candidate.get("zip_size_bytes")
    incumbent_size = incumbent.get("zip_size_bytes")
    if candidate_size is None or incumbent_size is None:
        return False
    return int(candidate_size) < int(incumbent_size)


def _validate_record(record: Any, add_defaults: bool = False) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise LedgerValidationError("Ledger record must be a JSON object")
    try:
        validated = deepcopy(record)
    except Exception as exc:  # pragma: no cover - defensive boundary for unusual caller objects
        raise LedgerValidationError("Ledger record could not be copied safely") from exc

    if "submission_id" not in validated:
        raise LedgerValidationError("Ledger record is missing submission_id")
    if "dataset_id" not in validated:
        raise LedgerValidationError("Ledger record is missing dataset_id")
    _validate_identifier(validated["submission_id"], "submission_id")
    _validate_identifier(validated["dataset_id"], "dataset_id")

    if add_defaults:
        validated.setdefault("leaderboard_score", None)
        default_status = "complete" if validated["leaderboard_score"] is not None else "incomplete"
        validated.setdefault("feedback_status", default_status)
        validated.setdefault("recorded_at", _utc_now())
    score = validated.get("leaderboard_score")
    if score is not None:
        _validate_finite_number(score, "leaderboard_score")
    status = validated.get("feedback_status")
    if "feedback_status" in validated and status not in {"complete", "incomplete"}:
        raise LedgerValidationError("feedback_status must be 'complete' or 'incomplete'")
    if score is None and status == "complete":
        raise LedgerValidationError("complete feedback requires leaderboard_score")
    if score is not None and status == "incomplete":
        raise LedgerValidationError("incomplete feedback cannot contain leaderboard_score")

    if "recorded_at" in validated:
        _validate_timestamp(validated["recorded_at"])
    if "gpu_time_seconds" in validated:
        _validate_finite_number(validated["gpu_time_seconds"], "gpu_time_seconds", minimum=0.0)
    if "zip_size_bytes" in validated:
        _validate_nonnegative_integer(validated["zip_size_bytes"], "zip_size_bytes")
    for field in ("zip_sha256", "dataset_manifest_sha256"):
        if field in validated:
            _validate_sha256(validated[field], field)
    for field in ("config", "metrics"):
        if field in validated and not isinstance(validated[field], dict):
            raise LedgerValidationError(f"{field} must be a JSON object")
    for field in (
        "git_commit",
        "container_image_digest",
        "command",
        "hypothesis",
        "next_action",
        "reproducibility_status",
    ):
        if field in validated and (not isinstance(validated[field], str) or len(validated[field]) > 65_536):
            raise LedgerValidationError(f"{field} must be a string of at most 65536 characters")
    if "decision" in validated and validated["decision"] not in {"promote", "reject", "pending"}:
        raise LedgerValidationError("decision must be 'promote', 'reject' or 'pending'")

    try:
        serialized = json.dumps(validated, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LedgerValidationError(f"Ledger record is not finite JSON data: {exc}") from exc
    if len(serialized.encode("utf-8")) > MAX_RECORD_BYTES:
        raise LedgerValidationError(f"Ledger record exceeds {MAX_RECORD_BYTES} bytes")
    return validated


def _validate_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise LedgerValidationError(
            f"{field} must be 1-128 characters using only letters, digits, '.', '_' or '-', and start alphanumeric"
        )


def _validate_finite_number(value: Any, field: str, minimum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LedgerValidationError(f"{field} must be a finite number")
    if minimum is not None and float(value) < minimum:
        raise LedgerValidationError(f"{field} must be at least {minimum:g}")


def _validate_nonnegative_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerValidationError(f"{field} must be a non-negative integer")


def _validate_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise LedgerValidationError(f"{field} must be exactly 64 hexadecimal characters")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise LedgerValidationError("recorded_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerValidationError("recorded_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise LedgerValidationError("recorded_at must include a timezone")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _reject_json_constant(value: str) -> None:
    raise LedgerValidationError(f"non-finite JSON number is not allowed: {value}")


def _json_object_argument(raw: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
        raise argparse.ArgumentTypeError("JSON object is too large")
    try:
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, LedgerValidationError) as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return value


def _finite_float_argument(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite")
    return value


def _nonnegative_float_argument(raw: str) -> float:
    value = _finite_float_argument(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _nonnegative_int_argument(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append BTC feedback and rank immutable submission ledger records.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help=f"JSONL ledger (default: {DEFAULT_LEDGER_PATH})",
    )
    commands = parser.add_subparsers(dest="command_name", required=True)

    add = commands.add_parser("add-feedback", help="Append one complete official BTC feedback record.")
    add.add_argument("--submission-id", required=True)
    add.add_argument("--dataset-id", required=True)
    add.add_argument("--score", type=_finite_float_argument, required=True, help="Official BTC leaderboard score.")
    add.add_argument("--gpu-time-seconds", type=_nonnegative_float_argument)
    add.add_argument("--zip-size-bytes", type=_nonnegative_int_argument)
    add.add_argument("--zip-sha256")
    add.add_argument("--dataset-manifest-sha256")
    add.add_argument("--git-commit")
    add.add_argument("--container-image-digest")
    add.add_argument("--command")
    add.add_argument("--config-json", type=_json_object_argument)
    add.add_argument("--metrics-json", type=_json_object_argument)
    add.add_argument("--hypothesis")
    add.add_argument("--decision", choices=("promote", "reject", "pending"))
    add.add_argument("--next-action")

    best = commands.add_parser("best", help="Print the best complete official result for one dataset.")
    best.add_argument("--dataset-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.command_name == "add-feedback":
            result = add_feedback(
                submission_id=args.submission_id,
                dataset_id=args.dataset_id,
                score=args.score,
                ledger_path=args.ledger,
                gpu_time_seconds=args.gpu_time_seconds,
                zip_size_bytes=args.zip_size_bytes,
                zip_sha256=args.zip_sha256,
                dataset_manifest_sha256=args.dataset_manifest_sha256,
                git_commit=args.git_commit,
                container_image_digest=args.container_image_digest,
                command=args.command,
                config=args.config_json,
                metrics=args.metrics_json,
                hypothesis=args.hypothesis,
                decision=args.decision,
                next_action=args.next_action,
            )
        else:
            result = best_submission(args.dataset_id, ledger_path=args.ledger)
    except (LedgerValidationError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
