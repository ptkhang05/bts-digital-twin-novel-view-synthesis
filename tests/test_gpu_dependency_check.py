from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

CHECKER_PATH = (
    Path(__file__).parents[1] / "infra" / "gpu" / "check_pip_environment.py"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_pip_environment", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accepts_only_the_pinned_upstream_ninja_platform_warning() -> None:
    checker = _load_checker()

    checker.validate_pip_check(
        returncode=1,
        output="ninja 1.11.1.1 is not supported on this platform\n",
    )


def test_rejects_any_additional_dependency_problem() -> None:
    checker = _load_checker()

    with pytest.raises(RuntimeError, match="broken-package"):
        checker.validate_pip_check(
            returncode=1,
            output=(
                "ninja 1.11.1.1 is not supported on this platform\n"
                "broken-package 1.0 requires missing-package\n"
            ),
        )


def test_rejects_a_failed_check_without_diagnostics() -> None:
    checker = _load_checker()

    with pytest.raises(RuntimeError, match="no diagnostics"):
        checker.validate_pip_check(returncode=1, output="")
