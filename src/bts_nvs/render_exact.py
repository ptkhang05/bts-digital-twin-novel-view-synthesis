from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bts_nvs.distortion import RectifiedCalibration, rectify_intrinsics, redistort_image
from bts_nvs.exceptions import DataValidationError
from bts_nvs.render import _promote_render_directory
from bts_nvs.schema import frame_intrinsics, load_json, validate_transforms


@dataclass(frozen=True)
class ExactTarget:
    name: str
    transform_matrix: list[list[float]]
    calibration: RectifiedCalibration


def load_exact_targets(path: Path | str) -> list[ExactTarget]:
    targets = validate_transforms(load_json(Path(path)))
    exact_targets: list[ExactTarget] = []
    for frame in targets["frames"]:
        intrinsics = frame_intrinsics(frame, targets)
        camera_model = str(intrinsics["camera_model"]).upper()
        if camera_model == "OPENCV_FISHEYE":
            raise DataValidationError("Exact target rendering currently supports perspective camera models only")
        distortion = _distortion_values(intrinsics)
        calibration = rectify_intrinsics(
            width=int(intrinsics["w"]),
            height=int(intrinsics["h"]),
            fx=float(intrinsics["fl_x"]),
            fy=float(intrinsics["fl_y"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            preserve_all_pixels=True,
            **distortion,
        )
        exact_targets.append(
            ExactTarget(
                name=Path(frame["file_path"]).name,
                transform_matrix=frame["transform_matrix"],
                calibration=calibration,
            )
        )
    return exact_targets


def render_exact_targets(checkpoint: Path | str, targets: Path | str, output: Path | str) -> None:
    try:
        import torch
        from nerfstudio.cameras.cameras import Cameras, CameraType
        from nerfstudio.utils.eval_utils import eval_setup
    except ImportError as exc:
        raise DataValidationError(
            "Exact target rendering must run in the Python environment where Nerfstudio is installed."
        ) from exc

    exact_targets = load_exact_targets(targets)
    _, pipeline, _, _ = eval_setup(Path(checkpoint), test_mode="inference")
    output_dir = Path(output)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for index, target in enumerate(exact_targets, start=1):
            print(f"[{index}/{len(exact_targets)}] Rendering {target.name}", flush=True)
            calibration = target.calibration
            camera = Cameras(
                camera_to_worlds=torch.tensor(target.transform_matrix, dtype=torch.float32)[None, :3, :4],
                fx=calibration.render_fx,
                fy=calibration.render_fy,
                cx=calibration.render_cx,
                cy=calibration.render_cy,
                width=calibration.render_width,
                height=calibration.render_height,
                camera_type=CameraType.PERSPECTIVE,
            ).to(pipeline.device)
            # This is the same public rendering API used by Nerfstudio's ns-render:
            # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/render.py
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera(camera)
            if "rgb" not in outputs:
                raise DataValidationError(f"Nerfstudio model did not return an rgb output for {target.name}")
            rectified = np.clip(outputs["rgb"].detach().cpu().numpy(), 0.0, 1.0)
            distorted = redistort_image(rectified, calibration)
            encoded = np.clip(np.rint(distorted * 255.0), 0, 255).astype(np.uint8)
            _save_submission_image(encoded, staging / target.name)
        _promote_render_directory(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _distortion_values(intrinsics: dict[str, Any]) -> dict[str, float]:
    values = {key: float(intrinsics.get(key, 0.0)) for key in ("k1", "k2", "k3", "k4", "p1", "p2")}
    packed = intrinsics.get("distortion_params")
    if packed is not None:
        if not isinstance(packed, list) or len(packed) != 6:
            raise DataValidationError("distortion_params must contain [k1, k2, k3, k4, p1, p2]")
        values.update(dict(zip(("k1", "k2", "k3", "k4", "p1", "p2"), map(float, packed))))
    return values


def _save_submission_image(array: np.ndarray, destination: Path) -> None:
    image = Image.fromarray(array).convert("RGB")
    suffix = destination.suffix.lower()
    if suffix == ".png":
        image.save(destination, format="PNG")
    elif suffix in {".jpg", ".jpeg"}:
        image.save(destination, format="JPEG", quality=95, optimize=True, progressive=False)
    else:
        raise DataValidationError(f"Unsupported target image extension for submission render: {destination.name}")


def _remove_submission_images(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            path.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render exact target intrinsics and restore source lens distortion.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    render_exact_targets(args.checkpoint, args.targets, args.out)


if __name__ == "__main__":
    main()
