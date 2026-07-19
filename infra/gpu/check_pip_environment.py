from __future__ import annotations

import subprocess
import sys

ALLOWED_UPSTREAM_FAILURES = {
    "ninja 1.11.1.1 is not supported on this platform",
}


def validate_pip_check(*, returncode: int, output: str) -> None:
    """Reject pip-check failures except the pinned base image's ninja metadata."""
    if returncode == 0:
        return

    diagnostics = {line.strip() for line in output.splitlines() if line.strip()}
    if not diagnostics:
        raise RuntimeError("pip check failed with no diagnostics")

    unexpected = diagnostics - ALLOWED_UPSTREAM_FAILURES
    if unexpected:
        raise RuntimeError(
            "pip check found unexpected dependency problems:\n"
            + "\n".join(sorted(unexpected))
        )

    print(
        "Accepted pinned Nerfstudio base-image warning: "
        + ", ".join(sorted(diagnostics)),
        file=sys.stderr,
    )


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    if output:
        print(output, end="")
    validate_pip_check(returncode=result.returncode, output=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
