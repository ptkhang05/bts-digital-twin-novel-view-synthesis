import json
from pathlib import Path

import pytest

from bts_nvs.ledger import (
    LedgerValidationError,
    add_feedback,
    append_record,
    best_submission,
    load_records,
    main,
)


def _record(submission_id: str, dataset_id: str = "vai_round2", score=50.0, **overrides):
    record = {
        "submission_id": submission_id,
        "dataset_id": dataset_id,
        "leaderboard_score": score,
    }
    record.update(overrides)
    return record


def test_append_record_is_jsonl_and_rejects_duplicate_submission_id(tmp_path: Path):
    ledger = tmp_path / "submission_history.jsonl"
    append_record(_record("run-001"), ledger_path=ledger)

    with pytest.raises(LedgerValidationError, match="already exists"):
        append_record(_record("run-001", dataset_id="another_dataset"), ledger_path=ledger)

    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["submission_id"] == "run-001"


def test_best_ignores_incomplete_feedback_and_other_datasets(tmp_path: Path):
    ledger = tmp_path / "submission_history.jsonl"
    append_record(_record("pending", score=None), ledger_path=ledger)
    append_record(_record("other", dataset_id="other_dataset", score=999.0), ledger_path=ledger)
    append_record(_record("complete", score=51.25), ledger_path=ledger)

    best = best_submission("vai_round2", ledger_path=ledger)

    assert best is not None
    assert best["submission_id"] == "complete"
    assert best_submission("missing_dataset", ledger_path=ledger) is None


def test_best_uses_gpu_time_then_zip_size_for_exact_score_ties(tmp_path: Path):
    ledger = tmp_path / "submission_history.jsonl"
    append_record(_record("incumbent", gpu_time_seconds=120.0, zip_size_bytes=900), ledger_path=ledger)
    append_record(_record("faster", gpu_time_seconds=100.0, zip_size_bytes=2_000), ledger_path=ledger)
    append_record(_record("same_time_smaller", gpu_time_seconds=100.0, zip_size_bytes=800), ledger_path=ledger)
    append_record(_record("missing_tie_data", zip_size_bytes=1), ledger_path=ledger)

    best = best_submission("vai_round2", ledger_path=ledger)

    assert best is not None
    assert best["submission_id"] == "same_time_smaller"


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_record("../escape"), "submission_id"),
        (_record("run-nan", score=float("nan")), "leaderboard_score"),
        (_record("run-gpu", gpu_time_seconds=-1), "gpu_time_seconds"),
        (_record("run-size", zip_size_bytes=True), "zip_size_bytes"),
        (_record("run-hash", zip_sha256="not-a-sha"), "zip_sha256"),
    ],
)
def test_append_record_rejects_unsafe_or_non_finite_values(tmp_path: Path, record: dict, message: str):
    with pytest.raises(LedgerValidationError, match=message):
        append_record(record, ledger_path=tmp_path / "submission_history.jsonl")


def test_add_feedback_records_optional_reproducibility_metadata(tmp_path: Path):
    ledger = tmp_path / "submission_history.jsonl"
    record = add_feedback(
        submission_id="round2-001",
        dataset_id="vai_round2",
        score=58.3,
        ledger_path=ledger,
        gpu_time_seconds=123.5,
        zip_size_bytes=456,
        zip_sha256="a" * 64,
        config={"method": "splatfacto-big", "iterations": 30_000},
    )

    assert record["leaderboard_score"] == 58.3
    assert record["feedback_status"] == "complete"
    assert record["config"]["iterations"] == 30_000
    assert record["recorded_at"].endswith("Z")
    assert record["seed"] is None
    assert record["iterations"] is None
    assert record["distortion"] is None
    assert record["jpeg_quality"] is None
    assert record["hardware"] is None
    assert record["validator_status"] is None
    assert record["score_delta_vs_best"] is None
    assert record["reproducibility_status"] == "partial"
    assert load_records(ledger)[0] == record


def test_add_feedback_accepts_complete_current_run_metadata(tmp_path: Path):
    ledger = tmp_path / "submission_history.jsonl"
    record = add_feedback(
        submission_id="round2-complete",
        dataset_id="vai_nvs_round2",
        score=61.2,
        ledger_path=ledger,
        gpu_time_seconds=321.0,
        zip_size_bytes=456,
        zip_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        git_commit="c" * 40,
        container_image_digest="sha256:" + "d" * 64,
        command="python -m bts_nvs.train ...",
        config={"rasterization": "classic"},
        metrics={"scenes": {"chair": {"score": 0.5}}},
        seed=42,
        iterations=30_000,
        distortion="auto",
        jpeg_quality=95,
        hardware={"gpu": "24 GB class"},
        validator_status="pass",
        score_delta_vs_best=1.2,
        hypothesis="classic baseline",
        decision="promote",
        next_action="compare antialiased",
    )

    assert record["reproducibility_status"] == "complete"
    assert record["seed"] == 42
    assert record["iterations"] == 30_000
    assert record["jpeg_quality"] == 95
    assert record["score_delta_vs_best"] == 1.2


def test_cli_add_feedback_then_best_outputs_json(tmp_path: Path, capsys):
    ledger = tmp_path / "submission_history.jsonl"
    assert main([
        "--ledger", str(ledger), "add-feedback",
        "--submission-id", "cli-001", "--dataset-id", "vai_round2", "--score", "60.5",
        "--gpu-time-seconds", "90", "--zip-size-bytes", "1000", "--zip-sha256", "b" * 64,
        "--config-json", '{"rasterization":"classic"}',
    ]) == 0
    added = json.loads(capsys.readouterr().out)
    assert added["submission_id"] == "cli-001"

    assert main(["--ledger", str(ledger), "best", "--dataset-id", "vai_round2"]) == 0
    best = json.loads(capsys.readouterr().out)
    assert best["submission_id"] == "cli-001"
