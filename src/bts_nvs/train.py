from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from bts_nvs.exceptions import ExternalCommandError


PRESETS = {
    "fast": "splatfacto",
    "quality": "splatfacto-big",
}


def build_train_command(
    scene: Path | str,
    preset: str = "fast",
    output_dir: Path | str | None = None,
    experiment_name: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Expected one of: {', '.join(PRESETS)}")
    command = ["ns-train", PRESETS[preset]]
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if experiment_name is not None:
        command.extend(["--experiment-name", experiment_name])
    command.extend(["--data", str(scene)])
    if extra_args:
        command.extend(extra_args)
    return command


def run_external_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ExternalCommandError(f"Command not found: {command[0]}. Install Nerfstudio in this environment.") from exc
    except subprocess.CalledProcessError as exc:
        raise ExternalCommandError(f"External command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Nerfstudio splatfacto model for a prepared scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Prepared scene directory.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="fast")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed after -- to ns-train.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    command = build_train_command(
        scene=args.scene,
        preset=args.preset,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        extra_args=extra_args,
    )
    if args.dry_run:
        print(" ".join(command))
        return
    run_external_command(command)


if __name__ == "__main__":
    main()
