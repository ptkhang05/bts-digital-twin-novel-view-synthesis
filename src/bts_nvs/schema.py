from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from PIL import Image

from bts_nvs.camera import validate_transform_matrix
from bts_nvs.exceptions import DataValidationError
from bts_nvs.path_safety import require_path_within

INTRINSIC_KEYS = ("fl_x", "fl_y", "cx", "cy", "w", "h")
DISTORTION_KEYS = ("k1", "k2", "k3", "k4", "p1", "p2", "distortion_params")
PERSPECTIVE_MODELS = {
    "PINHOLE",
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "FULL_OPENCV",
    "OPENCV_FISHEYE",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise DataValidationError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"{path} must contain a JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def frame_value(frame: dict[str, Any], meta: dict[str, Any], key: str) -> Any:
    if key in frame:
        return frame[key]
    if key in meta:
        return meta[key]
    raise DataValidationError(f"Missing camera intrinsic '{key}' for frame {frame.get('file_path', '<unknown>')}")


def frame_intrinsics(frame: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    values = {key: frame_value(frame, meta, key) for key in INTRINSIC_KEYS}
    for key in DISTORTION_KEYS:
        if key in frame:
            values[key] = frame[key]
        elif key in meta:
            values[key] = meta[key]
    values["camera_model"] = frame.get("camera_model", meta.get("camera_model", "OPENCV"))
    return values


def resolve_frame_image(scene: Path, file_path: str) -> Path:
    candidate = Path(file_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DataValidationError(f"Frame image must stay inside the scene directory: {file_path}")
    candidates = [
        scene / candidate,
        scene / "images" / candidate.name,
        scene / "train" / "images" / candidate.name,
    ]
    for path in candidates:
        safe_path = require_path_within(scene, path, label="Frame image")
        if safe_path.is_file():
            return safe_path
    raise DataValidationError(f"Frame image is missing: {file_path}")


def normalized_image_relpath(file_path: str) -> Path:
    path = Path(file_path)
    if path.parts and path.parts[0] == "images":
        return path
    return Path("images") / path.name


def validate_transforms(meta: dict[str, Any], scene: Path | None = None) -> dict[str, Any]:
    clean = copy.deepcopy(meta)
    frames = clean.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DataValidationError("Camera JSON must contain a non-empty 'frames' list")

    camera_model = str(clean.get("camera_model", "OPENCV")).upper()
    if camera_model not in PERSPECTIVE_MODELS:
        raise DataValidationError(f"Unsupported camera_model for splatfacto baseline: {camera_model}")
    clean["camera_model"] = camera_model

    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise DataValidationError(f"Frame {index} must be an object")
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise DataValidationError(f"Frame {index} must contain a non-empty file_path")
        try:
            matrix = validate_transform_matrix(frame["transform_matrix"])
        except KeyError as exc:
            raise DataValidationError(f"Frame {file_path} is missing transform_matrix") from exc
        except ValueError as exc:
            raise DataValidationError(f"Invalid transform_matrix for {file_path}: {exc}") from exc
        frame["transform_matrix"] = matrix.tolist()

        intrinsics = frame_intrinsics(frame, clean)
        for key in ("fl_x", "fl_y"):
            if float(intrinsics[key]) <= 0:
                raise DataValidationError(f"{key} must be positive for frame {file_path}")
        for key in ("w", "h"):
            if int(intrinsics[key]) <= 0:
                raise DataValidationError(f"{key} must be positive for frame {file_path}")

        if scene is not None:
            image_path = resolve_frame_image(scene, file_path)
            with Image.open(image_path) as image:
                width, height = image.size
            if (width, height) != (int(intrinsics["w"]), int(intrinsics["h"])):
                raise DataValidationError(
                    f"Image resolution mismatch for {file_path}: "
                    f"metadata has {intrinsics['w']}x{intrinsics['h']}, file has {width}x{height}"
                )
    return clean
