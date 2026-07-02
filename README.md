# BTS Digital Twin Novel View Synthesis Baseline

This repository provides a practical baseline pipeline for Viettel AI Race 2026
VAR (`var-2026`), "BTS Digital Twin (Novel View Synthesis)". It prepares
COLMAP/NeRF-style posed drone-image scenes for Nerfstudio, trains a 3D Gaussian
Splatting model with `splatfacto`, renders RGB target views as PNG image
sequences, evaluates predictions against holdout ground truth when available,
and packages multi-scene predictions into a ZIP submission.

The general public problem statement says each scene contains 100-300 RGB images
with camera intrinsics/poses and 20-50 target views. The round 1 brief is more
specific: 150-300 train images and 40-70 target views. The released phase1 data
uses COLMAP sparse reconstructions under `train/sparse/0` and target poses under
`test/test_poses.csv`. Public phase metadata lists `FILE_ZIP` submissions on
`GPU` workers. The BTC PDF briefs specify a ZIP submission containing rendered
PNG files grouped by scene, so this project packages predictions as
`scene_id/*.png`.

Source: https://competition.viettel.vn/contests/var-2026

## Install

```powershell
python -m pip install -e ".[dev]"
```

Install Nerfstudio separately in the environment used for training/rendering.
The local tests do not require Nerfstudio because this package treats
`ns-train` and `ns-render` as external commands.

## Scene Layout

Supported raw scene inputs:

- Viettel/VAI phase1 scene layout:
  - `train/images/`
  - `train/sparse/0/{cameras,images,points3D}.bin`
  - `test/test_poses.csv`
- Nerfstudio/NeRF-style `train_cameras.json` or `transforms.json` plus `images/`.
- COLMAP sparse reconstruction in `sparse/0`, `sparse`, or `colmap/sparse/0`
  plus `images/`.

Target views use the same JSON camera schema as `transforms.json`.

When a raw scene contains `target_cameras.json`, `prepare` validates and copies
it to the processed scene. For VAI phase1 scenes, `prepare` converts
`test/test_poses.csv` into `target_cameras.json`. Use `--strict-contest` with
`--contest-phase` to enforce a known BTC rule set:

- `phase1`: 150-300 training images, 40-70 target cameras.
- `overview`: 100-300 training images, 20-50 target cameras.

## Commands

```powershell
python -m bts_nvs.prepare_dataset --root VAI_NVS_DATA/phase1/public_set --out processed/public_set --copy-mode hardlink --strict-contest
python -m bts_nvs.prepare --scene raw_scene --out processed_scene --strict-contest
python -m bts_nvs.train --scene processed_scene --preset fast
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets processed_scene/target_cameras.json --out submission/scene_id --strict-contest
python -m bts_nvs.evaluate --pred submission/scene_id --gt VAI_NVS_DATA/phase1/public_set/scene_id/test/images --match-by-stem --psnr-max 40
python -m bts_nvs.package --submission submission --out submission.zip
```

Non-dry-run training captures the external `ns-train` output to
`<processed_scene>/training.log` by default. Override it with `--log-file` when
you want logs outside the processed scene directory.

For a dry run that prints the external command without running Nerfstudio:

```powershell
python -m bts_nvs.train --scene processed_scene --dry-run
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets target_cameras.json --out submission/scene_id --dry-run
```

## Submission Layout

The package command writes only PNG files into the archive:

```text
submission.zip
  scene_001/
    <target_image_name>.png
    <another_target_image_name>.png
  scene_002/
    <target_image_name>.png
```

For VAI phase1, rendered filenames follow the `image_name` column in
`test/test_poses.csv` with a PNG extension. The general problem statement also
shows numbered filenames such as `0001.png`; treat that as a round-specific
adapter concern if BTC later requires numbered output instead of phase1
`image_name` output.

## VAI Phase1 Notes

- `train/sparse/0/images.bin` can contain poses for images not present in
  `train/images`; the converter filters COLMAP registered images to files that
  actually exist in `train/images`.
- `test_poses.csv` stores `tx,ty,tz` as world-space camera position according to
  the released README. The adapter treats quaternion columns as OpenCV/COLMAP
  camera rotation and converts to Nerfstudio/OpenGL camera-to-world matrices.
- Rendered predictions are PNGs, while public ground-truth test images are JPGs;
  use `--match-by-stem` for local public-set evaluation.
- When SSIM and LPIPS dependencies are installed, `evaluate` also reports BTC's
  aggregate score: `0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * psnr_norm`, where
  `psnr_norm = clamp(PSNR / --psnr-max, 0, 1)`.
