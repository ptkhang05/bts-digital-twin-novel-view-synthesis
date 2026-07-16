from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from bts_nvs.audit import check_audit_manifest
from bts_nvs.exceptions import DataValidationError
from bts_nvs.path_safety import assert_paths_do_not_overlap, assert_tree_has_no_links
from bts_nvs.prepare import COPY_MODES, PreparedScene, prepare_scene, validate_preparation_provenance
from bts_nvs.vai import discover_vai_scenes


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
    dataset_id: str | None = None,
    manifest: Path | str | None = None,
) -> PreparedDataset:
    root_path = Path(root)
    output_path = Path(output)
    if (dataset_id is None) != (manifest is None):
        raise DataValidationError("Preparation provenance requires both dataset_id and manifest")

    manifest_sha256: str | None = None
    if dataset_id is not None and manifest is not None:
        verified = check_audit_manifest(root_path, Path(manifest))
        if verified.get("dataset_id") != root_path.name:
            raise DataValidationError("Verified manifest identity does not match the dataset root")
        manifest_sha256 = verified.get("overall_sha256")
        try:
            dataset_id, manifest_sha256 = validate_preparation_provenance(dataset_id, manifest_sha256)
        except DataValidationError as exc:
            raise DataValidationError(f"Verified manifest identity or SHA-256 is invalid: {exc}") from exc

    assert_paths_do_not_overlap(
        root_path,
        output_path,
        first_label="Raw dataset root",
        second_label="Prepared dataset output",
    )
    assert_tree_has_no_links(root_path, "Raw dataset root")
    prepared: list[PreparedScene] = []
    for scene_dir in discover_vai_scenes(root_path):
        prepared.append(
            prepare_scene(
                scene=scene_dir,
                output=output_path / scene_dir.name,
                copy_mode=copy_mode,
                overwrite=overwrite,
                dataset_id=dataset_id,
                dataset_manifest_sha256=manifest_sha256,
            )
        )
    return PreparedDataset(output_path, prepared)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare every VAI scene under a dataset root.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, e.g. VAI_NVS_DATA_ROUND2.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for processed scene folders.")
    parser.add_argument("--copy-mode", choices=COPY_MODES, default="copy")
    parser.add_argument("--dataset-id", help="Logical dataset ID stored in every scene metadata.json.")
    parser.add_argument("--manifest", type=Path, help="Verified audit manifest for the raw dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-scene output directories.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_dataset(
        root=args.root,
        output=args.out,
        copy_mode=args.copy_mode,
        overwrite=args.overwrite,
        dataset_id=args.dataset_id,
        manifest=args.manifest,
    )
    print(f"Wrote {result.scene_count} scenes, {result.image_count} train frames, {result.target_count} target poses")


if __name__ == "__main__":
    main()
