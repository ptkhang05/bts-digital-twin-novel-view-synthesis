from __future__ import annotations

import argparse
import csv
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from bts_nvs.exceptions import DataValidationError
from bts_nvs.vai import TEST_POSE_COLUMNS, discover_vai_phase1_scenes, find_test_poses_csv

SUBMISSION_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SubmissionIssue:
    message: str


@dataclass(frozen=True)
class SubmissionValidationResult:
    valid: bool
    scene_count: int
    image_count: int
    issues: list[SubmissionIssue] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if self.valid:
            return
        messages = "\n".join(f"- {issue.message}" for issue in self.issues)
        raise DataValidationError(f"Submission validation failed:\n{messages}")


@dataclass(frozen=True)
class ExpectedImage:
    name: str
    width: int
    height: int


def validate_submission(data_root: Path | str, submission: Path | str) -> SubmissionValidationResult:
    expected = _expected_outputs(Path(data_root))
    submission_path = Path(submission)
    if submission_path.suffix.lower() == ".zip":
        return _validate_zip_submission(expected, submission_path)
    return _validate_folder_submission(expected, submission_path)


def _expected_outputs(data_root: Path) -> dict[str, dict[str, ExpectedImage]]:
    scenes: dict[str, dict[str, ExpectedImage]] = {}
    for scene in discover_vai_phase1_scenes(data_root):
        scenes[scene.name] = _read_expected_images(find_test_poses_csv(scene))
    return scenes


def _read_expected_images(csv_path: Path) -> dict[str, ExpectedImage]:
    if not csv_path.exists():
        raise DataValidationError(f"target pose CSV does not exist: {csv_path}")
    expected: dict[str, ExpectedImage] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in TEST_POSE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise DataValidationError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        for row_index, row in enumerate(reader, start=2):
            image_name = Path(row["image_name"].strip()).name
            if not image_name:
                raise DataValidationError(f"Missing image_name in {csv_path} row {row_index}")
            if image_name in expected:
                raise DataValidationError(f"Duplicate image_name in {csv_path}: {image_name}")
            expected[image_name] = ExpectedImage(
                name=image_name,
                width=_positive_int(row["width"], "width", csv_path, row_index),
                height=_positive_int(row["height"], "height", csv_path, row_index),
            )
    if not expected:
        raise DataValidationError(f"{csv_path} contains no target poses")
    return expected


def _positive_int(value: str, column: str, csv_path: Path, row_index: int) -> int:
    try:
        parsed = int(float(value))
    except ValueError as exc:
        raise DataValidationError(f"Invalid integer for {column} in {csv_path} row {row_index}: {value}") from exc
    if parsed <= 0:
        raise DataValidationError(f"{column} must be positive in {csv_path} row {row_index}")
    return parsed


def _validate_folder_submission(
    expected: dict[str, dict[str, ExpectedImage]],
    submission_dir: Path,
) -> SubmissionValidationResult:
    issues: list[SubmissionIssue] = []
    if not submission_dir.exists():
        issues.append(SubmissionIssue(f"Submission path does not exist: {submission_dir}"))
        return _result(expected, issues, image_count=0)
    if not submission_dir.is_dir():
        issues.append(SubmissionIssue(f"Submission path must be a directory or .zip: {submission_dir}"))
        return _result(expected, issues, image_count=0)

    root_files = [path.name for path in submission_dir.iterdir() if path.is_file()]
    for name in sorted(root_files):
        issues.append(SubmissionIssue(f"Unexpected root file in submission folder: {name}"))

    actual_scenes = {path.name: path for path in submission_dir.iterdir() if path.is_dir()}
    _validate_scene_names(expected, set(actual_scenes), issues)

    image_count = 0
    for scene_name in sorted(set(expected) & set(actual_scenes)):
        image_count += _validate_folder_scene(expected[scene_name], scene_name, actual_scenes[scene_name], issues)
    return _result(expected, issues, image_count=image_count)


def _validate_folder_scene(
    expected_images: dict[str, ExpectedImage],
    scene_name: str,
    scene_dir: Path,
    issues: list[SubmissionIssue],
) -> int:
    for child in sorted(scene_dir.iterdir()):
        if child.is_dir():
            issues.append(SubmissionIssue(f"Nested folders are not allowed inside scene output: {scene_name}/{child.name}"))

    actual_files = {path.name: path for path in scene_dir.iterdir() if _is_submission_image(path)}
    _validate_image_names(scene_name, expected_images, set(actual_files), issues)
    for image_name in sorted(set(expected_images) & set(actual_files)):
        _validate_image_stream(
            scene_name=scene_name,
            image_name=image_name,
            expected=expected_images[image_name],
            opener=lambda path=actual_files[image_name]: path.open("rb"),
            issues=issues,
        )
    return len(actual_files)


def _validate_zip_submission(
    expected: dict[str, dict[str, ExpectedImage]],
    zip_path: Path,
) -> SubmissionValidationResult:
    issues: list[SubmissionIssue] = []
    if not zip_path.exists():
        issues.append(SubmissionIssue(f"Submission ZIP does not exist: {zip_path}"))
        return _result(expected, issues, image_count=0)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member for member in archive.namelist() if not member.endswith("/")]
            unsafe = [name for name in members if Path(name).is_absolute() or ".." in Path(name).parts]
            for name in unsafe:
                issues.append(SubmissionIssue(f"Unsafe ZIP member path: {name}"))

            actual_by_scene: dict[str, dict[str, str]] = {}
            for name in members:
                parts = Path(name).parts
                if len(parts) != 2:
                    issues.append(SubmissionIssue(f"ZIP member must be exactly scene/image, got: {name}"))
                    continue
                scene_name, image_name = parts
                if Path(image_name).suffix.lower() not in SUBMISSION_IMAGE_SUFFIXES:
                    issues.append(SubmissionIssue(f"Unsupported image suffix in ZIP member: {name}"))
                    continue
                actual_by_scene.setdefault(scene_name, {})[image_name] = name

            _validate_scene_names(expected, set(actual_by_scene), issues)
            image_count = sum(len(images) for images in actual_by_scene.values())
            for scene_name in sorted(set(expected) & set(actual_by_scene)):
                actual_images = actual_by_scene[scene_name]
                _validate_image_names(scene_name, expected[scene_name], set(actual_images), issues)
                for image_name in sorted(set(expected[scene_name]) & set(actual_images)):
                    _validate_image_stream(
                        scene_name=scene_name,
                        image_name=image_name,
                        expected=expected[scene_name][image_name],
                        opener=lambda member=actual_images[image_name]: archive.open(member),
                        issues=issues,
                    )
    except zipfile.BadZipFile:
        issues.append(SubmissionIssue(f"Submission is not a readable ZIP file: {zip_path}"))
        image_count = 0
    return _result(expected, issues, image_count=image_count)


def _validate_scene_names(
    expected: dict[str, dict[str, ExpectedImage]],
    actual_scene_names: set[str],
    issues: list[SubmissionIssue],
) -> None:
    expected_scene_names = set(expected)
    for scene_name in sorted(expected_scene_names - actual_scene_names):
        issues.append(SubmissionIssue(f"Missing scene folder: {scene_name}"))
    for scene_name in sorted(actual_scene_names - expected_scene_names):
        issues.append(SubmissionIssue(f"Extra scene folder: {scene_name}"))


def _validate_image_names(
    scene_name: str,
    expected_images: dict[str, ExpectedImage],
    actual_image_names: set[str],
    issues: list[SubmissionIssue],
) -> None:
    expected_names = set(expected_images)
    for image_name in sorted(expected_names - actual_image_names):
        issues.append(SubmissionIssue(f"Missing output image: {scene_name}/{image_name}"))
    for image_name in sorted(actual_image_names - expected_names):
        issues.append(SubmissionIssue(f"Extra output image: {scene_name}/{image_name}"))


def _validate_image_stream(
    scene_name: str,
    image_name: str,
    expected: ExpectedImage,
    opener,
    issues: list[SubmissionIssue],
) -> None:
    try:
        with opener() as handle:
            data = handle.read()
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            if image.size != (expected.width, expected.height):
                issues.append(
                    SubmissionIssue(
                        f"Image size mismatch for {scene_name}/{image_name}: "
                        f"got {image.size}, expected {(expected.width, expected.height)}"
                    )
                )
            if image.mode != "RGB":
                issues.append(SubmissionIssue(f"Image mode mismatch for {scene_name}/{image_name}: got {image.mode}, expected RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        issues.append(SubmissionIssue(f"Output image is not readable: {scene_name}/{image_name} ({exc})"))


def _is_submission_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUBMISSION_IMAGE_SUFFIXES


def _result(
    expected: dict[str, dict[str, ExpectedImage]],
    issues: list[SubmissionIssue],
    image_count: int,
) -> SubmissionValidationResult:
    return SubmissionValidationResult(
        valid=not issues,
        scene_count=len(expected),
        image_count=image_count,
        issues=issues,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a VAI/BTC submission folder or ZIP against test_poses.csv.")
    parser.add_argument("--data-root", type=Path, required=True, help="VAI scene root or dataset root containing scene folders.")
    parser.add_argument("--submission", type=Path, required=True, help="Submission folder or ZIP to validate.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = validate_submission(data_root=args.data_root, submission=args.submission)
    if result.valid:
        print(f"Valid submission: {result.image_count} images from {result.scene_count} scenes")
        return
    for issue in result.issues:
        print(f"- {issue.message}")
    result.raise_for_errors()


if __name__ == "__main__":
    main()
