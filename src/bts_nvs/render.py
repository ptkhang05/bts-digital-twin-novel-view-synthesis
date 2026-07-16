from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
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
DISTORTION_MODES = ("auto", "on", "off")


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


def targets_have_lens_distortion(targets_path: Path | str) -> bool:
    """Return whether any target camera has a non-zero distortion coefficient."""
    targets = validate_transforms(load_json(Path(targets_path)))
    for frame in targets["frames"]:
        intrinsics = frame_intrinsics(frame, targets)
        packed = intrinsics.get("distortion_params")
        if packed is not None:
            if not isinstance(packed, list) or len(packed) != 6:
                raise DataValidationError("distortion_params must contain [k1, k2, k3, k4, p1, p2]")
            if any(abs(float(value)) > 0.0 for value in packed):
                return True
        if any(abs(float(intrinsics.get(key, 0.0))) > 0.0 for key in ("k1", "k2", "k3", "k4", "p1", "p2")):
            return True
    return False


def render_targets(
    checkpoint: Path | str,
    targets: Path | str,
    output: Path | str,
    dry_run: bool = False,
    strict_contest: bool = False,
    contest_phase: str = DEFAULT_CONTEST_PHASE,
    distortion: str = "auto",
) -> list[str]:
    output_dir = Path(output)
    if distortion not in DISTORTION_MODES:
        raise DataValidationError(f"Unknown distortion mode '{distortion}'. Expected one of: {', '.join(DISTORTION_MODES)}")
    camera_path, target_names = build_camera_path(
        targets,
        strict_contest=strict_contest,
        contest_phase=contest_phase,
    )
    use_exact_renderer = distortion == "on" or (distortion == "auto" and targets_have_lens_distortion(targets))
    if dry_run:
        if use_exact_renderer:
            return build_exact_render_command(checkpoint, targets, output_dir)
        return build_render_command(checkpoint, output_dir / "camera_path.json", output_dir / "targets")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        if use_exact_renderer:
            command = build_exact_render_command(checkpoint, targets, staging)
            run_external_command(command)
            _verify_rendered_images(staging, target_names)
            _promote_render_directory(staging, output_dir)
            return command

        camera_path_file = staging / "camera_path.json"
        nerfstudio_output = staging / "targets"
        write_json(camera_path_file, camera_path)
        command = build_render_command(checkpoint, camera_path_file, nerfstudio_output)
        run_external_command(command)
        render_dir = rendered_image_directory(nerfstudio_output)
        _rename_rendered_images(render_dir, staging, target_names)
        if render_dir.exists():
            shutil.rmtree(render_dir)
        camera_path_file.unlink(missing_ok=True)
        _verify_rendered_images(staging, target_names)
        _promote_render_directory(staging, output_dir)
        return command
    finally:
        if staging.exists():
            shutil.rmtree(staging)


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


def _verify_rendered_images(output_dir: Path, target_names: list[str]) -> None:
    expected = set(target_names)
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUBMISSION_IMAGE_SUFFIXES
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DataValidationError(f"Rendered image set mismatch; missing={missing}, extra={extra}")


def _promote_render_directory(staging: Path, output: Path) -> None:
    """Promote a complete staging directory while keeping the previous output recoverable."""
    backup = output.parent / f".{output.name}.backup"
    if backup.exists():
        if output.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, output)
    if output.exists() and not output.is_dir():
        raise DataValidationError(f"Render output exists and is not a directory: {output}")
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except Exception:
        if backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


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
        "--distortion",
        choices=DISTORTION_MODES,
        default="auto",
        help="Lens-distortion handling: auto detects non-zero coefficients, on forces exact rendering, off disables it.",
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
        distortion=args.distortion,
    )
    print(" ".join(command))


if __name__ == "__main__":
    main()
