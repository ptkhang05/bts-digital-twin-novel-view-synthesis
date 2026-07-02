from __future__ import annotations

import argparse
import zipfile
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.contest import DEFAULT_CONTEST_PHASE, validate_target_view_count
from bts_nvs.exceptions import DataValidationError


@dataclass(frozen=True)
class PackagedSubmission:
    zip_path: Path
    scene_count: int
    image_count: int


def create_submission_zip(
    submission: Path | str,
    output: Path | str,
    strict_contest: bool = True,
    contest_phase: str = DEFAULT_CONTEST_PHASE,
) -> PackagedSubmission:
    submission_dir = Path(submission)
    output_path = Path(output)
    if not submission_dir.exists():
        raise DataValidationError(f"Submission directory does not exist: {submission_dir}")

    scenes = discover_scene_outputs(submission_dir)
    if not scenes:
        raise DataValidationError(f"No PNG predictions found under {submission_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene_id, images in scenes:
            if strict_contest:
                validate_target_view_count(len(images), phase=contest_phase)
            for image_path in images:
                archive.write(image_path, f"{scene_id}/{image_path.name}")
                image_count += 1
    return PackagedSubmission(output_path, scene_count=len(scenes), image_count=image_count)


def discover_scene_outputs(submission_dir: Path) -> list[tuple[str, list[Path]]]:
    scene_dirs = [path for path in sorted(submission_dir.iterdir()) if path.is_dir()]
    scenes: list[tuple[str, list[Path]]] = []
    for scene_dir in scene_dirs:
        images = sorted(path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
        if images:
            scenes.append((scene_dir.name, images))
    if scenes:
        return scenes

    root_images = sorted(path for path in submission_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    if root_images:
        return [(submission_dir.name, root_images)]
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package rendered scene predictions into a contest ZIP file.")
    parser.add_argument("--submission", type=Path, required=True, help="Directory containing scene_id/*.png outputs.")
    parser.add_argument("--out", type=Path, required=True, help="ZIP file path to write.")
    parser.add_argument(
        "--no-strict-contest",
        action="store_true",
        help="Do not enforce target-view limits from the selected rule set.",
    )
    parser.add_argument(
        "--contest-phase",
        default=DEFAULT_CONTEST_PHASE,
        help="Contest rule set for strict validation. Known values: phase1, overview.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = create_submission_zip(
        args.submission,
        args.out,
        strict_contest=not args.no_strict_contest,
        contest_phase=args.contest_phase,
    )
    print(f"Wrote {result.zip_path} with {result.image_count} PNGs from {result.scene_count} scenes")


if __name__ == "__main__":
    main()
