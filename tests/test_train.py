from pathlib import Path

from bts_nvs.train import build_train_command


def test_build_train_command_uses_splatfacto_for_fast_preset(tmp_path: Path):
    command = build_train_command(scene=tmp_path / "processed", preset="fast")

    assert command == ["ns-train", "splatfacto", "--data", str(tmp_path / "processed")]


def test_build_train_command_uses_splatfacto_big_for_quality_preset(tmp_path: Path):
    command = build_train_command(scene=tmp_path / "processed", preset="quality", output_dir=tmp_path / "outputs")

    assert command[:2] == ["ns-train", "splatfacto-big"]
    assert "--output-dir" in command
    assert str(tmp_path / "outputs") in command
