from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.prepare import PreparedScene, prepare_scene
from bts_nvs.vai import discover_vai_phase1_scenes


@dataclass(frozen=True)
class PreparedDataset:
    output_dir: Path
    scenes: list[PreparedScene]

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def image_count(self) -> int:
        return sum(scene.image_count for scene in self.scenes)

    @property
    def target_count(self) -> int:
        return sum(scene.target_count or 0 for scene in self.scenes)


def prepare_dataset(
    root: Path | str,
    output: Path | str,
    copy_mode: str = "copy",
    overwrite: bool = False,
    strict_contest: bool = False,
) -> PreparedDataset:
    root_path = Path(root)
    output_path = Path(output)
    prepared: list[PreparedScene] = []
    for scene_dir in discover_vai_phase1_scenes(root_path):
        prepared.append(
            prepare_scene(
                scene=scene_dir,
                output=output_path / scene_dir.name,
                copy_mode=copy_mode,
                overwrite=overwrite,
                strict_contest=strict_contest,
            )
        )
    return PreparedDataset(output_path, prepared)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare every VAI phase1 scene under a dataset root.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, e.g. VAI_NVS_DATA/phase1/public_set.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for processed scene folders.")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-scene output directories.")
    parser.add_argument("--strict-contest", action="store_true", help="Enforce public phase1 train/target count constraints.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_dataset(
        root=args.root,
        output=args.out,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        strict_contest=args.strict_contest,
    )
    print(f"Wrote {result.scene_count} scenes, {result.image_count} train frames, {result.target_count} target poses")


if __name__ == "__main__":
    main()
