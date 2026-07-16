from __future__ import annotations

import argparse
from pathlib import Path

from bts_nvs.package import create_submission_zip


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the single validated VAI/BTC submission ZIP.")
    parser.add_argument("--data-root", type=Path, required=True, help="VAI dataset root containing target CSV files.")
    parser.add_argument("--submission", type=Path, required=True, help="outputs/candidate/rendered directory.")
    parser.add_argument("--out", type=Path, required=True, help="Final submission.zip path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
