from __future__ import annotations

import argparse
import shlex
import subprocess
import warnings
from pathlib import Path

from bts_nvs.exceptions import ExternalCommandError


PRESETS = {
    "fast": "splatfacto",
    "quality": "splatfacto-big",
    "quality-aa": "splatfacto-big",
}

PRESET_ARGS = {
    "quality-aa": ["--pipeline.model.rasterize-mode", "antialiased"],
}

RECOMMENDED_MAX_ITERATIONS = 30_000


def build_train_command(
    scene: Path | str,
    preset: str = "fast",
    output_dir: Path | str | None = None,
    experiment_name: str | None = None,
    extra_args: list[str] | None = None,
    disable_pose_normalization: bool = False,
) -> list[str]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Expected one of: {', '.join(PRESETS)}")
    command = ["ns-train", PRESETS[preset]]
    command.extend(PRESET_ARGS.get(preset, []))
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if experiment_name is not None:
        command.extend(["--experiment-name", experiment_name])
    if extra_args:
        command.extend(extra_args)
    if disable_pose_normalization:
        command.extend(
            [
                "nerfstudio-data",
                "--data",
                str(scene),
                "--orientation-method",
                "none",
                "--center-method",
                "none",
                "--auto-scale-poses",
                "False",
            ]
        )
    else:
        command.extend(["--data", str(scene)])
    return command


def warn_extended_training(preset: str, extra_args: list[str] | None) -> None:
    """Warn when a Splatfacto run exceeds Nerfstudio's tuned schedule."""
    if preset not in PRESETS or not extra_args:
        return

    iterations: int | None = None
    for index, argument in enumerate(extra_args):
        if argument == "--max-num-iterations" and index + 1 < len(extra_args):
            try:
                iterations = int(extra_args[index + 1])
            except ValueError:
                return
            break
        if argument.startswith("--max-num-iterations="):
            try:
                iterations = int(argument.split("=", 1)[1])
            except ValueError:
                return
            break

    if iterations is not None and iterations > RECOMMENDED_MAX_ITERATIONS:
        warnings.warn(
            f"Nerfstudio Splatfacto schedules are tuned for {RECOMMENDED_MAX_ITERATIONS} iterations; "
            f"requested {iterations}. Validate extended training on the complete public set before private runs.",
            UserWarning,
            stacklevel=2,
        )


def default_train_log_path(scene: Path | str) -> Path:
    return Path(scene) / "training.log"


def run_external_command(command: list[str], log_path: Path | str | None = None) -> None:
    if log_path is None:
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise ExternalCommandError(
                f"Command not found: {command[0]}. Install Nerfstudio in this environment."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise ExternalCommandError(
                f"External command failed with exit code {exc.returncode}: {' '.join(command)}"
            ) from exc
        return

    resolved_log_path = Path(log_path)
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise ExternalCommandError(
                f"Command not found: {command[0]}. Install Nerfstudio in this environment."
            ) from exc

        if process.stdout is None:
            raise ExternalCommandError(f"External command did not expose stdout: {' '.join(command)}")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise ExternalCommandError(f"External command failed with exit code {return_code}: {' '.join(command)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Nerfstudio splatfacto model for a prepared scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Prepared scene directory.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="fast")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path for captured ns-train output. Defaults to <scene>/training.log.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument(
        "--disable-pose-normalization",
        action="store_true",
        help="Use nerfstudio-data with orientation/centering/auto-scale disabled. "
        "Use this when target cameras are already in the same COLMAP coordinate frame as training poses.",
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed after -- to ns-train.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    warn_extended_training(args.preset, extra_args)
    command = build_train_command(
        scene=args.scene,
        preset=args.preset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        extra_args=extra_args,
        disable_pose_normalization=args.disable_pose_normalization,
    )
    if args.dry_run:
        print(" ".join(command))
        return
    run_external_command(command, log_path=args.log_file or default_train_log_path(args.scene))


if __name__ == "__main__":
    main()
