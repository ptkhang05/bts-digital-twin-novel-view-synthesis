from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from bts_nvs.camera import opencv_w2c_to_nerfstudio_c2w
from bts_nvs.exceptions import DataValidationError


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float


@dataclass(frozen=True)
class ColmapModel:
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3d: dict[int, ColmapPoint3D]


CAMERA_MODELS_BY_ID: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODELS_BY_NAME = {name: (model_id, params) for model_id, (name, params) in CAMERA_MODELS_BY_ID.items()}


def find_colmap_sparse_dir(scene: Path) -> Path | None:
    for candidate in (scene / "sparse" / "0", scene / "sparse", scene / "colmap" / "sparse" / "0"):
        if _has_colmap_files(candidate):
            return candidate
    return None


def _has_colmap_files(path: Path) -> bool:
    return path.exists() and (
        (path / "cameras.bin").exists()
        or (path / "cameras.txt").exists()
    ) and ((path / "images.bin").exists() or (path / "images.txt").exists())


def read_colmap_model(path: Path) -> ColmapModel:
    if not path.exists():
        raise DataValidationError(f"COLMAP sparse directory does not exist: {path}")
    if (path / "cameras.bin").exists() and (path / "images.bin").exists():
        return ColmapModel(
            cameras=_read_cameras_binary(path / "cameras.bin"),
            images=_read_images_binary(path / "images.bin"),
            points3d=_read_points3d_binary(path / "points3D.bin") if (path / "points3D.bin").exists() else {},
        )
    if (path / "cameras.txt").exists() and (path / "images.txt").exists():
        return ColmapModel(
            cameras=_read_cameras_text(path / "cameras.txt"),
            images=_read_images_text(path / "images.txt"),
            points3d=_read_points3d_text(path / "points3D.txt") if (path / "points3D.txt").exists() else {},
        )
    raise DataValidationError(f"Could not find supported COLMAP cameras/images files in {path}")


def _read_cameras_text(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise DataValidationError(f"Invalid cameras.txt line: {line}")
        camera_id = int(parts[0])
        model = parts[1].upper()
        width = int(parts[2])
        height = int(parts[3])
        params = tuple(float(value) for value in parts[4:])
        expected = CAMERA_MODELS_BY_NAME.get(model)
        if expected is None:
            raise DataValidationError(f"Unsupported COLMAP camera model: {model}")
        if len(params) != expected[1]:
            raise DataValidationError(
                f"Camera {camera_id} model {model} expects {expected[1]} params, got {len(params)}"
            )
        cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    if not cameras:
        raise DataValidationError(f"No cameras found in {path}")
    return cameras


def _read_images_text(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            raise DataValidationError(f"Invalid images.txt line: {line}")
        image_id = int(parts[0])
        qvec = tuple(float(value) for value in parts[1:5])
        tvec = tuple(float(value) for value in parts[5:8])
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
        if index < len(lines):
            index += 1
    if not images:
        raise DataValidationError(f"No images found in {path}")
    return images


def _read_points3d_text(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            raise DataValidationError(f"Invalid points3D.txt line: {line}")
        point_id = int(parts[0])
        xyz = tuple(float(value) for value in parts[1:4])
        rgb = tuple(int(value) for value in parts[4:7])
        error = float(parts[7])
        points[point_id] = ColmapPoint3D(point_id, xyz, rgb, error)
    return points


def _read_next_bytes(handle: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize("<" + fmt)
    data = handle.read(size)
    if len(data) != size:
        raise DataValidationError("Unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def _read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        (count,) = _read_next_bytes(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read_next_bytes(handle, "iiQQ")
            if model_id not in CAMERA_MODELS_BY_ID:
                raise DataValidationError(f"Unsupported COLMAP camera model id: {model_id}")
            model, param_count = CAMERA_MODELS_BY_ID[model_id]
            params = tuple(float(value) for value in _read_next_bytes(handle, "d" * param_count))
            cameras[camera_id] = ColmapCamera(camera_id, model, width, height, params)
    return cameras


def _read_null_terminated_string(handle: BinaryIO) -> str:
    chars = bytearray()
    while True:
        char = handle.read(1)
        if char == b"":
            raise DataValidationError("Unexpected end of COLMAP binary string")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def _read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as handle:
        (count,) = _read_next_bytes(handle, "Q")
        for _ in range(count):
            values = _read_next_bytes(handle, "idddddddi")
            image_id = int(values[0])
            qvec = tuple(float(value) for value in values[1:5])
            tvec = tuple(float(value) for value in values[5:8])
            camera_id = int(values[8])
            name = _read_null_terminated_string(handle)
            (point_count,) = _read_next_bytes(handle, "Q")
            if point_count:
                handle.seek(struct.calcsize("<ddq") * point_count, 1)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name)
    return images


def _read_points3d_binary(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as handle:
        (count,) = _read_next_bytes(handle, "Q")
        for _ in range(count):
            values = _read_next_bytes(handle, "QdddBBBd")
            point_id = int(values[0])
            xyz = tuple(float(value) for value in values[1:4])
            rgb = tuple(int(value) for value in values[4:7])
            error = float(values[7])
            (track_length,) = _read_next_bytes(handle, "Q")
            if track_length:
                handle.seek(struct.calcsize("<ii") * track_length, 1)
            points[point_id] = ColmapPoint3D(point_id, xyz, rgb, error)
    return points


def camera_to_nerfstudio_intrinsics(camera: ColmapCamera) -> dict[str, float | int | str]:
    model = camera.model
    params = camera.params
    if model == "SIMPLE_PINHOLE":
        fl_x = fl_y = params[0]
        cx, cy = params[1], params[2]
        extra: dict[str, float] = {}
    elif model == "PINHOLE":
        fl_x, fl_y, cx, cy = params
        extra = {}
    elif model == "SIMPLE_RADIAL":
        fl_x = fl_y = params[0]
        cx, cy, k1 = params[1], params[2], params[3]
        extra = {"k1": k1}
    elif model == "RADIAL":
        fl_x = fl_y = params[0]
        cx, cy, k1, k2 = params[1], params[2], params[3], params[4]
        extra = {"k1": k1, "k2": k2}
    elif model == "OPENCV":
        fl_x, fl_y, cx, cy, k1, k2, p1, p2 = params
        extra = {"k1": k1, "k2": k2, "p1": p1, "p2": p2}
    elif model == "OPENCV_FISHEYE":
        fl_x, fl_y, cx, cy, k1, k2, k3, k4 = params
        extra = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
    else:
        raise DataValidationError(f"Unsupported COLMAP camera model for conversion: {model}")
    return {
        "camera_model": "OPENCV_FISHEYE" if model == "OPENCV_FISHEYE" else "OPENCV",
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(camera.width),
        "h": int(camera.height),
        **extra,
    }


def colmap_model_to_nerfstudio(
    scene: Path,
    model: ColmapModel,
    image_names: set[str] | None = None,
) -> tuple[dict, list[ColmapPoint3D]]:
    frames = []
    intrinsics_by_image: list[dict] = []
    images = sorted(model.images.values(), key=lambda item: item.image_id)
    if image_names is not None:
        images = [image for image in images if Path(image.name).name in image_names]
    for image in images:
        if image.camera_id not in model.cameras:
            raise DataValidationError(f"Image {image.name} references missing camera {image.camera_id}")
        camera = model.cameras[image.camera_id]
        intrinsics = camera_to_nerfstudio_intrinsics(camera)
        intrinsics_by_image.append(intrinsics)
        image_rel = _colmap_image_relpath(scene, image.name)
        frame = {
            "file_path": image_rel.as_posix(),
            "transform_matrix": opencv_w2c_to_nerfstudio_c2w(image.qvec, image.tvec).tolist(),
        }
        frame.update(intrinsics)
        frames.append(frame)

    if not frames:
        raise DataValidationError("COLMAP model contains no registered images with matching files")

    first_intrinsics = intrinsics_by_image[0]
    shared_keys = tuple(first_intrinsics.keys())
    all_shared = all({key: item.get(key) for key in shared_keys} == first_intrinsics for item in intrinsics_by_image)
    transforms = {"frames": frames}
    if all_shared:
        transforms.update(first_intrinsics)
        for frame in frames:
            for key in shared_keys:
                frame.pop(key, None)
    return transforms, list(model.points3d.values())


def _colmap_image_relpath(scene: Path, image_name: str) -> Path:
    direct = scene / image_name
    under_images = scene / "images" / image_name
    under_train_images = scene / "train" / "images" / image_name
    if direct.exists():
        return Path(image_name)
    if under_images.exists():
        return Path("images") / image_name
    if under_train_images.exists():
        return Path("images") / Path(image_name).name
    return Path("images") / Path(image_name).name


def write_ascii_ply(points: Iterable[ColmapPoint3D], path: Path) -> int:
    point_list = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(point_list)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point in point_list:
            x, y, z = point.xyz
            r, g, b = point.rgb
            handle.write(f"{x} {y} {z} {r} {g} {b}\n")
    return len(point_list)
