import sys
import warnings
from pathlib import Path

import pytest

from bts_nvs.exceptions import ExternalCommandError
from bts_nvs.train import build_train_command, default_train_log_path, run_external_command, warn_extended_training


def test_build_train_command_uses_splatfacto_for_fast_preset(tmp_path: Path):
    command = build_train_command(scene=tmp_path / "processed", preset="fast")

    assert command == [
        "ns-train",
        "splatfacto",
        "nerfstudio-data",
        "--data",
        str(tmp_path / "processed"),
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
    ]


def test_build_train_command_uses_splatfacto_big_for_quality_preset(tmp_path: Path):
    command = build_train_command(scene=tmp_path / "processed", preset="quality", output_dir=tmp_path / "outputs")

    assert command[:2] == ["ns-train", "splatfacto-big"]
    assert "--output-dir" in command
    assert str(tmp_path / "outputs") in command


def test_build_train_command_enables_antialiasing_for_quality_aa_preset(tmp_path: Path):
    command = build_train_command(scene=tmp_path / "processed", preset="quality-aa")

    assert command == [
        "ns-train",
        "splatfacto-big",
        "--pipeline.model.rasterize-mode",
        "antialiased",
        "nerfstudio-data",
        "--data",
        str(tmp_path / "processed"),
        "--orientation-method",
        "none",
        "--center-method",
        "none",
        "--auto-scale-poses",
        "False",
    ]


def test_warn_extended_training_flags_splatfacto_runs_over_30000_steps():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_extended_training("quality", ["--max-num-iterations", "60000"])

    assert len(caught) == 1
    assert "30000" in str(caught[0].message)


def test_warn_extended_training_accepts_default_iteration_budget():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_extended_training("quality", ["--max-num-iterations", "30000"])

    assert caught == []


def test_build_train_command_can_enable_pose_normalization_only_as_debug_override(tmp_path: Path):
    scene = tmp_path / "processed"

    command = build_train_command(
        scene=scene,
        preset="fast",
        extra_args=["--max-num-iterations", "5000"],
        debug_allow_pose_normalization=True,
    )

    assert command == [
        "ns-train",
        "splatfacto",
        "--max-num-iterations",
        "5000",
        "--data",
        str(scene),
    ]


def test_default_train_log_path_writes_inside_processed_scene(tmp_path: Path):
    scene = tmp_path / "processed_scene"

    assert default_train_log_path(scene) == scene / "training.log"


def test_run_external_command_streams_stdout_and_stderr_to_log(tmp_path: Path):
    log_path = tmp_path / "logs" / "training.log"

    run_external_command(
        [
            sys.executable,
            "-c",
            "import sys; print('train stdout'); print('train stderr', file=sys.stderr)",
        ],
        log_path=log_path,
    )

    log_text = log_path.read_text(encoding="utf-8")
    assert "train stdout" in log_text
    assert "train stderr" in log_text


def test_run_external_command_keeps_log_when_command_fails(tmp_path: Path):
    log_path = tmp_path / "training.log"

    with pytest.raises(ExternalCommandError):
        run_external_command(
            [sys.executable, "-c", "import sys; print('before failure'); sys.exit(7)"],
            log_path=log_path,
        )

    assert "before failure" in log_path.read_text(encoding="utf-8")
