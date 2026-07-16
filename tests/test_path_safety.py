from pathlib import Path

import pytest

from bts_nvs.exceptions import DataValidationError
from bts_nvs.path_safety import assert_path_has_no_links


def test_assert_path_has_no_links_rejects_symlink_ancestor(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(DataValidationError, match="symlink or junction"):
        assert_path_has_no_links(link / "output", "Output")


def test_assert_path_has_no_links_rejects_junction_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    junction = tmp_path / "junction"
    junction.mkdir()

    monkeypatch.setattr(Path, "is_junction", lambda path: path == junction, raising=False)

    with pytest.raises(DataValidationError, match="symlink or junction"):
        assert_path_has_no_links(junction / "output", "Output")
