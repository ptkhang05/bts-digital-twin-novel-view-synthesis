from __future__ import annotations

import argparse
import os
from pathlib import Path

from bts_nvs.exceptions import DataValidationError
from bts_nvs.package import create_submission_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the single validated VAI/BTC submission ZIP.")
    parser.add_argument("--data-root", type=Path, required=True, help="VAI dataset root containing target CSV files.")
    parser.add_argument("--submission", type=Path, required=True, help="outputs/candidate/rendered directory.")
    parser.add_argument("--out", type=Path, required=True, help="Final submission.zip path.")
    return parser


def _validate_fixed_artifact_layout(submission: Path, output: Path) -> None:
    submission_absolute = Path(os.path.abspath(submission))
    output_absolute = Path(os.path.abspath(output))
    if (
        submission_absolute.name != "rendered"
        or submission_absolute.parent.name != "candidate"
        or submission_absolute.parent.parent.name != "outputs"
    ):
        raise DataValidationError("Submission source must be outputs/candidate/rendered")
    expected_output = submission_absolute.parent.parent.parent / "submission.zip"
    if output_absolute != expected_output:
        raise DataValidationError("Submission output must be the single repository-root submission.zip")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        _validate_fixed_artifact_layout(args.submission, args.out)
    except DataValidationError as exc:
        parser.error(str(exc))
    result = create_submission_zip(
        data_root=args.data_root,
        submission=args.submission,
        output=args.out,
        staging_dir=args.out.parent / "outputs" / ".staging",
    )
    print(
        f"Submission ready: {result.zip_path} ({result.image_count} JPEGs, {result.scene_count} scenes, "
        f"sha256={result.sha256})"
    )


if __name__ == "__main__":
    main()
