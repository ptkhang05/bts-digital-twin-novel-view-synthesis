from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from bts_nvs.camera import compute_vertical_fov_degrees
from bts_nvs.contest import validate_target_view_count
from bts_nvs.exceptions import DataValidationError
from bts_nvs.schema import frame_intrinsics, load_json, validate_transforms, write_json
from bts_nvs.train import run_external_command


def build_camera_path(targets_path: Path | str, strict_contest: bool = False) -> tuple[dict, list[str]]:
    targets_file = Path(targets_path)
    targets = validate_transforms(load_json(targets_file))
    if strict_contest:
        validate_target_view_count(len(targets["frames"]))
    names: list[str] = []
    camera_entries: list[dict] = []
    first_intrinsics = frame_intrinsics(targets["frames"][0], targets)
    for index, frame in enumerate(targets["frames"]):
        intrinsics = frame_intrinsics(frame, targets)
        width = int(intrinsics["w"])
        height = int(intrinsics["h"])
        if width != int(first_intrinsics["w"]) or height != int(first_intrinsics["h"]):
            raise DataValidationError("All target cameras must share render width/height for one image-sequence render")
        name = Path(frame.get("file_path") or f"{index:05d}.png").name
        if not name.lower().endswith(".png"):
            name = f"{Path(name).stem}.png"
        names.append(name)
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


def render_targets(
    checkpoint: Path | str,
    targets: Path | str,
    output: Path | str,
    dry_run: bool = False,
    strict_contest: bool = False,
) -> list[str]:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_path, target_names = build_camera_path(targets, strict_contest=strict_contest)
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
        shutil.copy2(source, destination)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render target cameras with a trained Nerfstudio config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Nerfstudio config.yml.")
    parser.add_argument("--targets", type=Path, required=True, help="Target camera JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Submission image output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Write camera path and print command without running.")
    parser.add_argument("--strict-contest", action="store_true", help="Enforce 20-50 target views per scene.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    command = render_targets(args.checkpoint, args.targets, args.out, dry_run=args.dry_run, strict_contest=args.strict_contest)
    print(" ".join(command))


if __name__ == "__main__":
    main()
