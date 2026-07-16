from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from bts_nvs.exceptions import DataValidationError


def absolute_path(path: Path | str) -> Path:
    """Return a normalized absolute path without following links."""

    return Path(os.path.abspath(path))


def is_link_or_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(checker is not None and checker())


def assert_path_has_no_links(path: Path | str, label: str) -> Path:
    """Reject a path if it or any existing ancestor is a symlink/junction."""

    absolute = absolute_path(path)
    for component in (absolute, *absolute.parents):
        if (component.exists() or component.is_symlink()) and is_link_or_junction(component):
            raise DataValidationError(f"{label} uses an unsafe symlink or junction: {component}")
    return absolute


def assert_tree_has_no_links(root: Path | str, label: str) -> Path:
    """Reject links anywhere in an existing directory tree without following them."""

    root_absolute = assert_path_has_no_links(root, label)
    if not root_absolute.is_dir():
        raise DataValidationError(f"{label} is not a directory: {root_absolute}")
    pending = [root_absolute]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink() or is_link_or_junction(entry_path):
                    raise DataValidationError(f"{label} contains an unsafe symlink or junction: {entry_path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry_path)
    return root_absolute


def paths_overlap(first: Path | str, second: Path | str) -> bool:
    """Return whether either normalized path contains the other."""

    first_text = os.path.normcase(str(absolute_path(first)))
    second_text = os.path.normcase(str(absolute_path(second)))
    try:
        common = os.path.normcase(os.path.commonpath((first_text, second_text)))
    except ValueError:
        return False
    return common in {first_text, second_text}


def assert_paths_do_not_overlap(
    first: Path | str,
    second: Path | str,
    *,
    first_label: str,
    second_label: str,
) -> tuple[Path, Path]:
    first_absolute = assert_path_has_no_links(first, first_label)
    second_absolute = assert_path_has_no_links(second, second_label)
    if paths_overlap(first_absolute, second_absolute):
        raise DataValidationError(f"{first_label} and {second_label} must not overlap")
    return first_absolute, second_absolute


def assert_output_separate_from(
    output: Path | str,
    protected_paths: Iterable[tuple[Path | str, str]],
    *,
    output_label: str,
) -> Path:
    output_absolute = assert_path_has_no_links(output, output_label)
    for protected, protected_label in protected_paths:
        protected_absolute = assert_path_has_no_links(protected, protected_label)
        if paths_overlap(output_absolute, protected_absolute):
            raise DataValidationError(f"{output_label} must not overlap {protected_label}")
    return output_absolute


def require_path_within(root: Path | str, path: Path | str, *, label: str) -> Path:
    """Return an absolute safe path that remains below ``root``."""

    root_absolute = assert_path_has_no_links(root, "scene directory")
    candidate_absolute = absolute_path(path)
    if candidate_absolute == root_absolute or not candidate_absolute.is_relative_to(root_absolute):
        raise DataValidationError(f"{label} must stay inside the scene directory: {path}")
    return assert_path_has_no_links(candidate_absolute, label)


def promote_directory(staging: Path | str, output: Path | str, *, label: str) -> None:
    """Atomically replace ``output`` with a complete same-filesystem staging tree."""

    staging_path, output_path = assert_paths_do_not_overlap(
        staging,
        output,
        first_label=f"{label} staging directory",
        second_label=f"{label} output directory",
    )
    if not staging_path.is_dir():
        raise DataValidationError(f"{label} staging path is not a directory: {staging_path}")
    assert_tree_has_no_links(staging_path, f"{label} staging directory")
    if output_path.exists() and not output_path.is_dir():
        raise DataValidationError(f"{label} output exists and is not a directory: {output_path}")
    if output_path.exists():
        assert_tree_has_no_links(output_path, f"{label} output directory")
    if staging_path.stat().st_dev != output_path.parent.stat().st_dev:
        raise DataValidationError(f"{label} staging and output must be on the same filesystem")

    backup = output_path.parent / f".{output_path.name}.backup"
    assert_path_has_no_links(backup, f"{label} backup directory")
    if backup.exists():
        if not backup.is_dir():
            raise DataValidationError(f"{label} backup exists and is not a directory: {backup}")
        assert_tree_has_no_links(backup, f"{label} backup directory")
        if output_path.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, output_path)

    if output_path.exists():
        os.replace(output_path, backup)
    try:
        os.replace(staging_path, output_path)
    except BaseException:
        if backup.exists() and not output_path.exists():
            os.replace(backup, output_path)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
