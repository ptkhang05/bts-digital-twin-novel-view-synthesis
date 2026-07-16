import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

import bts_nvs.feedback as feedback_module
from bts_nvs.exceptions import DataValidationError
from bts_nvs.ledger import LedgerValidationError, append_record, load_records
from bts_nvs.package import create_submission_zip


def _write_dataset(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    scene = data_root / "scene_a"
    (scene / "train" / "images").mkdir(parents=True)
    (scene / "train" / "sparse" / "0").mkdir(parents=True)
    (scene / "test").mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(scene / "train" / "images" / "train.JPG", format="JPEG")
    (scene / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "target.JPG,1,0,0,0,0,0,0,10,10,4,3,8,6\n",
        encoding="utf-8",
    )
    return data_root


def _write_rendered(root: Path, color: tuple[int, int, int]) -> None:
    scene = root / "rendered" / "scene_a"
    scene.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=color).save(scene / "target.JPG", format="JPEG")
    (root / "run-marker.json").write_text(json.dumps({"color": color}), encoding="utf-8")


def _write_submitted_zip(data_root: Path, candidate_dir: Path, zip_path: Path) -> bytes:
    create_submission_zip(data_root=data_root, submission=candidate_dir / "rendered", output=zip_path)
    return zip_path.read_bytes()


def _paths(tmp_path: Path) -> dict[str, Path]:
    outputs = tmp_path / "outputs"
    return {
        "candidate_dir": outputs / "candidate",
        "best_dir": outputs / "best",
        "zip_path": tmp_path / "submission.zip",
        "ledger_path": tmp_path / "experiments" / "submission_history.jsonl",
    }


def _record(submission_id: str, dataset_id: str = "vai_round2", score: float = 50.0, **metadata):
    return {
        "submission_id": submission_id,
        "dataset_id": dataset_id,
        "leaderboard_score": score,
        **metadata,
    }


def test_first_feedback_promotes_entire_candidate_and_keeps_submitted_zip(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])

    result = feedback_module.process_feedback(
        submission_id="round2-001",
        dataset_id="vai_round2",
        score=50.0,
        data_root=data_root,
        **paths,
    )

    assert result["decision"] == "promote"
    assert not paths["candidate_dir"].exists()
    assert json.loads((paths["best_dir"] / "run-marker.json").read_text(encoding="utf-8"))["color"] == [200, 10, 10]
    assert paths["zip_path"].read_bytes() == submitted
    assert result["zip_size_bytes"] == len(submitted)
    assert result["zip_sha256"] == hashlib.sha256(submitted).hexdigest()
    assert result["validator_status"] == "pass"
    assert result["score_delta_vs_best"] is None
    assert load_records(paths["ledger_path"]) == [result]


def test_cli_records_complete_current_metadata_and_automatic_validation(tmp_path: Path, capsys):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])

    assert feedback_module.main(
        [
            "--submission-id",
            "round2-complete-metadata",
            "--dataset-id",
            "vai_round2",
            "--score",
            "50.0",
            "--data-root",
            str(data_root),
            "--candidate",
            str(paths["candidate_dir"]),
            "--best",
            str(paths["best_dir"]),
            "--zip",
            str(paths["zip_path"]),
            "--ledger",
            str(paths["ledger_path"]),
            "--dataset-manifest-sha256",
            "a" * 64,
            "--git-commit",
            "abc123",
            "--container-image-digest",
            "sha256:container",
            "--command",
            "python -m bts_nvs.train --preset quality",
            "--config-json",
            '{"rasterization":"classic"}',
            "--metrics-json",
            '{"chair":{"score":0.75}}',
            "--seed",
            "42",
            "--iterations",
            "30000",
            "--distortion",
            "auto",
            "--jpeg-quality",
            "95",
            "--hardware-json",
            '{"gpu":"RTX 4090","vram_gb":24}',
            "--gpu-time-seconds",
            "123.5",
            "--hypothesis",
            "classic baseline",
            "--next-action",
            "compare antialiased",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["seed"] == 42
    assert result["iterations"] == 30_000
    assert result["distortion"] == "auto"
    assert result["jpeg_quality"] == 95
    assert result["hardware"] == {"gpu": "RTX 4090", "vram_gb": 24}
    assert result["validator_status"] == "pass"
    assert result["score_delta_vs_best"] is None
    assert result["decision"] == "promote"
    assert result["reproducibility_status"] == "complete"


def test_feedback_accepts_identical_member_bytes_with_different_zip_metadata(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    deterministic_zip = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    with zipfile.ZipFile(paths["zip_path"]) as archive:
        member_payload = archive.read("scene_a/target.JPG")
    rewritten = tmp_path / "rewritten.zip"
    with zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("scene_a/target.JPG", member_payload, compress_type=zipfile.ZIP_STORED)
    os.replace(rewritten, paths["zip_path"])
    rewritten_payload = paths["zip_path"].read_bytes()
    assert rewritten_payload != deterministic_zip

    result = feedback_module.process_feedback(
        submission_id="round2-metadata-only-difference",
        dataset_id="vai_round2",
        score=50.0,
        data_root=data_root,
        **paths,
    )

    assert result["decision"] == "promote"
    assert result["zip_sha256"] == hashlib.sha256(rewritten_payload).hexdigest()
    assert paths["zip_path"].read_bytes() == rewritten_payload
    assert (paths["best_dir"] / "rendered" / "scene_a" / "target.JPG").read_bytes() == member_payload


def test_lower_score_regenerates_zip_from_best_before_removing_candidate(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("incumbent", score=60.0), paths["ledger_path"])

    result = feedback_module.process_feedback(
        submission_id="round2-002",
        dataset_id="vai_round2",
        score=59.0,
        data_root=data_root,
        **paths,
    )

    assert result["decision"] == "reject"
    assert result["validator_status"] == "pass"
    assert result["score_delta_vs_best"] == -1.0
    assert not paths["candidate_dir"].exists()
    assert paths["best_dir"].exists()
    with zipfile.ZipFile(paths["zip_path"]) as archive:
        with archive.open("scene_a/target.JPG") as member:
            assert member.read() == (paths["best_dir"] / "rendered" / "scene_a" / "target.JPG").read_bytes()
    assert [record["submission_id"] for record in load_records(paths["ledger_path"])] == [
        "incumbent",
        "round2-002",
    ]


def test_feedback_places_all_transient_artifacts_in_fixed_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("incumbent", score=60.0), paths["ledger_path"])
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(feedback_module.os, "replace", record_replace)

    feedback_module.process_feedback(
        submission_id="staging-contract",
        dataset_id="vai_round2",
        score=59.0,
        data_root=data_root,
        **paths,
    )

    staging = paths["candidate_dir"].parent / ".staging"
    transient_paths = [
        path
        for replacement in replacements
        for path in replacement
        if "feedback" in path.name
    ]
    assert transient_paths
    assert all(path.parent == staging for path in transient_paths)
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_promotion_failure_rolls_back_best_candidate_zip_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("incumbent", score=50.0), paths["ledger_path"])
    real_replace = os.replace
    failed = False

    def fail_candidate_promotion(source, destination):
        nonlocal failed
        if not failed and Path(source) == paths["candidate_dir"] and Path(destination) == paths["best_dir"]:
            failed = True
            raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(feedback_module.os, "replace", fail_candidate_promotion)

    with pytest.raises(OSError, match="simulated promotion failure"):
        feedback_module.process_feedback(
            submission_id="round2-002",
            dataset_id="vai_round2",
            score=51.0,
            data_root=data_root,
            **paths,
        )

    candidate_marker = json.loads((paths["candidate_dir"] / "run-marker.json").read_text(encoding="utf-8"))
    assert candidate_marker["color"] == [200, 10, 10]
    assert json.loads((paths["best_dir"] / "run-marker.json").read_text(encoding="utf-8"))["color"] == [10, 200, 10]
    assert paths["zip_path"].read_bytes() == submitted
    assert [record["submission_id"] for record in load_records(paths["ledger_path"])] == ["incumbent"]


def test_package_failure_restores_zip_and_preserves_best_candidate_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("incumbent", score=60.0), paths["ledger_path"])

    def corrupt_then_fail(*_args, **kwargs):
        Path(kwargs["output"]).write_bytes(b"partial replacement")
        raise OSError("simulated package failure")

    monkeypatch.setattr(feedback_module, "create_submission_zip", corrupt_then_fail)

    with pytest.raises(OSError, match="simulated package failure"):
        feedback_module.process_feedback(
            submission_id="round2-002",
            dataset_id="vai_round2",
            score=59.0,
            data_root=data_root,
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["best_dir"].exists()
    assert paths["zip_path"].read_bytes() == submitted
    assert [record["submission_id"] for record in load_records(paths["ledger_path"])] == ["incumbent"]


def test_ledger_append_failure_after_rejection_rolls_back_zip_candidate_and_partial_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("incumbent", score=60.0), paths["ledger_path"])
    ledger_before = paths["ledger_path"].read_bytes()
    real_append = feedback_module.ledger.append_record

    def append_then_fail(record, ledger_path):
        real_append(record, ledger_path)
        raise OSError("simulated ledger fsync failure")

    monkeypatch.setattr(feedback_module.ledger, "append_record", append_then_fail)

    with pytest.raises(OSError, match="simulated ledger fsync failure"):
        feedback_module.process_feedback(
            submission_id="round2-002",
            dataset_id="vai_round2",
            score=59.0,
            data_root=data_root,
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["best_dir"].exists()
    assert paths["zip_path"].read_bytes() == submitted
    assert paths["ledger_path"].read_bytes() == ledger_before


def test_duplicate_submission_id_is_rejected_before_inspecting_candidate(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    paths["candidate_dir"].mkdir(parents=True)
    paths["zip_path"].write_bytes(b"candidate zip")
    append_record(_record("duplicate"), paths["ledger_path"])
    before = paths["ledger_path"].read_bytes()

    with pytest.raises(LedgerValidationError, match="already exists"):
        feedback_module.process_feedback(
            submission_id="duplicate",
            dataset_id="vai_round2",
            score=999.0,
            data_root=data_root,
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["zip_path"].read_bytes() == b"candidate zip"
    assert paths["ledger_path"].read_bytes() == before


def test_candidate_tree_with_junction_is_rejected_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    junction = paths["candidate_dir"] / "unsafe-junction"
    junction.mkdir()
    monkeypatch.setattr(feedback_module, "_is_junction", lambda path: path == junction)

    with pytest.raises(DataValidationError, match="unsafe symlink or junction"):
        feedback_module.process_feedback(
            submission_id="unsafe-path",
            dataset_id="vai_round2",
            score=50.0,
            data_root=data_root,
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["zip_path"].read_bytes() == submitted
    assert not paths["ledger_path"].exists()


def test_best_from_another_dataset_is_ignored(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    append_record(_record("other-best", dataset_id="other_dataset", score=999.0), paths["ledger_path"])

    result = feedback_module.process_feedback(
        submission_id="round2-001",
        dataset_id="vai_round2",
        score=1.0,
        data_root=data_root,
        **paths,
    )

    assert result["decision"] == "promote"
    assert json.loads((paths["best_dir"] / "run-marker.json").read_text(encoding="utf-8"))["color"] == [200, 10, 10]


@pytest.mark.parametrize(
    ("candidate_gpu", "incumbent_gpu", "incumbent_size_delta", "expected"),
    [
        (9.0, 10.0, -1, True),
        (10.0, 10.0, 100, True),
        (None, 10.0, 100, False),
        (10.0, 10.0, None, False),
        (10.0, 10.0, 0, False),
    ],
    ids=("faster", "same-time-smaller", "missing-gpu", "missing-size", "exact-tie"),
)
def test_exact_score_tie_uses_gpu_time_then_zip_size_and_retains_incumbent_when_data_missing(
    tmp_path: Path,
    candidate_gpu: float | None,
    incumbent_gpu: float,
    incumbent_size_delta: int | None,
    expected: bool,
):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    _write_rendered(paths["best_dir"], (10, 200, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    incumbent_metadata = {"gpu_time_seconds": incumbent_gpu}
    if incumbent_size_delta is not None:
        incumbent_metadata["zip_size_bytes"] = len(submitted) + incumbent_size_delta
    append_record(_record("incumbent", **incumbent_metadata), paths["ledger_path"])

    result = feedback_module.process_feedback(
        submission_id="challenger",
        dataset_id="vai_round2",
        score=50.0,
        data_root=data_root,
        gpu_time_seconds=candidate_gpu,
        **paths,
    )

    assert (result["decision"] == "promote") is expected


def test_invalid_submitted_zip_is_rejected_before_mutation(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    paths["zip_path"].write_bytes(b"not a zip")

    with pytest.raises(DataValidationError, match="not a readable ZIP"):
        feedback_module.process_feedback(
            submission_id="invalid-zip",
            dataset_id="vai_round2",
            score=50.0,
            data_root=data_root,
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["zip_path"].read_bytes() == b"not a zip"
    assert not paths["ledger_path"].exists()


def test_candidate_modified_after_submission_is_rejected_without_mutation(tmp_path: Path):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    changed_image = paths["candidate_dir"] / "rendered" / "scene_a" / "target.JPG"
    Image.new("RGB", (8, 6), color=(10, 10, 200)).save(changed_image, format="JPEG")
    changed_payload = changed_image.read_bytes()

    with pytest.raises(DataValidationError, match="does not match the submitted ZIP"):
        feedback_module.process_feedback(
            submission_id="candidate-changed",
            dataset_id="vai_round2",
            score=50.0,
            data_root=data_root,
            **paths,
        )

    assert changed_image.read_bytes() == changed_payload
    assert not paths["best_dir"].exists()
    assert paths["zip_path"].read_bytes() == submitted
    assert not paths["ledger_path"].exists()
    assert not list(tmp_path.glob(".submission.zip.feedback-identity-*.zip"))


@pytest.mark.parametrize("field", ["zip_size_bytes", "zip_sha256"])
def test_supplied_zip_identity_must_match_actual_artifact(tmp_path: Path, field: str):
    data_root = _write_dataset(tmp_path)
    paths = _paths(tmp_path)
    _write_rendered(paths["candidate_dir"], (200, 10, 10))
    submitted = _write_submitted_zip(data_root, paths["candidate_dir"], paths["zip_path"])
    supplied = len(submitted) + 1 if field == "zip_size_bytes" else "0" * 64

    with pytest.raises(DataValidationError, match=f"{field} does not match"):
        feedback_module.process_feedback(
            submission_id=f"mismatch-{field}",
            dataset_id="vai_round2",
            score=50.0,
            data_root=data_root,
            **{field: supplied},
            **paths,
        )

    assert paths["candidate_dir"].exists()
    assert paths["zip_path"].read_bytes() == submitted
    assert not paths["ledger_path"].exists()
