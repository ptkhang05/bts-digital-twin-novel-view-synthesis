from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from bts_nvs.camera import compute_vertical_fov_degrees
from bts_nvs.contest import DEFAULT_CONTEST_PHASE, validate_target_view_count
from bts_nvs.exceptions import DataValidationError
from bts_nvs.schema import frame_intrinsics, load_json, validate_transforms, write_json
from bts_nvs.train import run_external_command

JPEG_SUFFIXES = {".jpg", ".jpeg"}
PNG_SUFFIXES = {".png"}
SUBMISSION_IMAGE_SUFFIXES = JPEG_SUFFIXES | PNG_SUFFIXES


def build_camera_path(
    targets_path: Path | str,
    strict_contest: bool = False,
    contest_phase: str = DEFAULT_CONTEST_PHASE,
) -> tuple[dict, list[str]]:
    targets_file = Path(targets_path)
    targets = validate_transforms(load_json(targets_file))
    if strict_contest:
        validate_target_view_count(len(targets["frames"]), phase=contest_phase)
    names: list[str] = []
    camera_entries: list[dict] = []
    first_intrinsics = frame_intrinsics(targets["frames"][0], targets)
    for index, frame in enumerate(targets["frames"]):
        intrinsics = frame_intrinsics(frame, targets)
        width = int(intrinsics["w"])
        height = int(intrinsics["h"])
        if width != int(first_intrinsics["w"]) or height != int(first_intrinsics["h"]):
            raise DataValidationError("All target cameras must share render width/height for one image-sequence render")
        names.append(Path(frame.get("file_path") or f"{index:05d}.png").name)
        camera_entries.append(
            {
                "camera_to_world": frame["transform_matrix"],
                "fov": compute_vertical_fov_degrees(height=height, fy=float(intrinsics["fl_y"])),
                "aspect": width / height,
            }
        )
    return (
        {
            "camera_type": "perspective",
            "render_height": int(first_intrinsics["h"]),
            "render_width": int(first_intrinsics["w"]),
            "seconds": max(len(camera_entries) / 24.0, 1.0 / 24.0),
            "is_cycle": False,
            "camera_path": camera_entries,
        },
        names,
    )


def rendered_image_directory(output_path: Path | str) -> Path:
    path = Path(output_path)
    return path.parent / path.stem


def build_render_command(checkpoint: Path | str, camera_path_file: Path | str, output_path: Path | str) -> list[str]:
    return [
        "ns-render",
        "camera-path",
        "--load-config",
        str(checkpoint),
        "--camera-path-filename",
        str(camera_path_file),
        "--output-path",
        str(output_path),
        "--output-format",
        "images",
        "--image-format",
        "png",
    ]


def build_exact_render_command(
    checkpoint: Path | str,
    targets: Path | str,
    output: Path | str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "bts_nvs.render_exact",
        "--checkpoint",
        str(checkpoint),
        "--targets",
        str(targets),
        "--out",
        str(output),
    ]


def render_targets(
    checkpoint: Path | str,
    targets: Path | str,
    output: Path | str,
    dry_run: bool = False,
    strict_contest: bool = False,
    contest_phase: str = DEFAULT_CONTEST_PHASE,
    apply_lens_distortion: bool = False,
) -> list[str]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_submission_images(output_dir)
    camera_path, target_names = build_camera_path(
        targets,
        strict_contest=strict_contest,
        contest_phase=contest_phase,
    )
    if apply_lens_distortion:
        command = build_exact_render_command(checkpoint, targets, output_dir)
        if not dry_run:
            run_external_command(command)
        return command
    camera_path_file = output_dir / "camera_path.json"
    nerfstudio_output = output_dir / "targets"
    write_json(camera_path_file, camera_path)
    command = build_render_command(checkpoint, camera_path_file, nerfstudio_output)
    if dry_run:
        return command
    run_external_command(command)
    _rename_rendered_images(rendered_image_directory(nerfstudio_output), output_dir, target_names)
    return command


def _rename_rendered_images(render_dir: Path, output_dir: Path, target_names: list[str]) -> None:
    for index, target_name in enumerate(target_names):
        source = render_dir / f"{index:05d}.png"
        if not source.exists():
            raise DataValidationError(f"Expected rendered image is missing: {source}")
        destination = output_dir / target_name
        if destination.resolve() == source.resolve():
            continue
        _write_submission_image(source, destination)


def _remove_stale_submission_images(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUBMISSION_IMAGE_SUFFIXES:
            path.unlink()


def _write_submission_image(source: Path, destination: Path) -> None:
    suffix = destination.suffix.lower()
    if suffix in PNG_SUFFIXES:
        shutil.copy2(source, destination)
        return
    if suffix in JPEG_SUFFIXES:
        with Image.open(source) as image:
            image.convert("RGB").save(destination, format="JPEG", quality=95, optimize=True, progressive=False)
        return
    raise DataValidationError(f"Unsupported target image extension for submission render: {destination.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render target cameras with a trained Nerfstudio config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Nerfstudio config.yml.")
    parser.add_argument("--targets", type=Path, required=True, help="Target camera JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Submission image output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write camera path and print command without running.")
    parser.add_argument("--strict-contest", action="store_true", help="Enforce target-view limits from the selected rule set.")
    parser.add_argument(
        "--apply-lens-distortion",
        action="store_true",
        help="Render with exact rectified intrinsics, then map pixels back through the source lens model.",
    )
    parser.add_argument(
        "--contest-phase",
        default=DEFAULT_CONTEST_PHASE,
        help="Contest rule set for --strict-contest. Known values: phase1, overview.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    command = render_targets(
        args.checkpoint,
        args.targets,
        args.out,
        dry_run=args.dry_run,
        strict_contest=args.strict_contest,
        contest_phase=args.contest_phase,
        apply_lens_distortion=args.apply_lens_distortion,
    )
    print(" ".join(command))


if __name__ == "__main__":
    main()
