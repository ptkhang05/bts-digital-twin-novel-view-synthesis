# BTS Digital Twin Novel View Synthesis Baseline

This repository provides a practical baseline pipeline for Viettel AI Race 2026
VAR (`var-2026`), "BTS Digital Twin (Novel View Synthesis)". It prepares
COLMAP/NeRF-style posed drone-image scenes for Nerfstudio, trains a 3D Gaussian
Splatting model with `splatfacto`, renders RGB target views as PNG image
sequences, evaluates predictions against holdout ground truth when available,
and packages multi-scene predictions into a ZIP submission.

The general public problem statement says each scene contains 100-300 RGB images
with camera intrinsics/poses and 20-50 target views. The round 1 brief says
150-300 train images and 40-70 target views, but the released `private_set1`
contains a scene with 103 train images and 26 target views. To avoid rejecting
official BTC-provided data, this project validates phase1 with the observed
safe envelope: 100-300 train images and 20-70 target views. The released phase1
data uses COLMAP sparse reconstructions under `train/sparse/0` and target poses
under `test/test_poses.csv`. Some BTC round text also refers to
`test/test_pose.csv`; the loader accepts both names and prefers
`test_poses.csv` when both exist. Public phase metadata lists `FILE_ZIP` submissions
on `GPU` workers. The BTC PDF briefs show PNG examples, but the round 1 brief
also says filenames must follow `image_name` in `test/test_poses.csv`. The
phase1 CSV currently uses `.JPG` names, so submission packaging preserves exact
target filenames.

Source: https://competition.viettel.vn/contests/var-2026

Local text snapshots from the BTC pages are kept in `docs/btc-main.md` and
`docs/btc-round1.md` so VM clones have the same readable reference without
committing the original PDF files.

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
  - `test/test_poses.csv` or `test/test_pose.csv`
- Nerfstudio/NeRF-style `train_cameras.json` or `transforms.json` plus `images/`.
- COLMAP sparse reconstruction in `sparse/0`, `sparse`, or `colmap/sparse/0`
  plus `images/`.

Target views use the same JSON camera schema as `transforms.json`.

When a raw scene contains `target_cameras.json`, `prepare` validates and copies
it to the processed scene. For VAI phase1 scenes, `prepare` converts
`test/test_poses.csv` or `test/test_pose.csv` into `target_cameras.json`. Use `--strict-contest` with
`--contest-phase` to enforce a known BTC rule set:

- `phase1`: 100-300 training images, 20-70 target cameras. This matches the
  observed public/private phase1 data currently present in `VAI_NVS_DATA`.
- `overview`: 100-300 training images, 20-50 target cameras.

## Commands

```powershell
python -m bts_nvs.prepare_dataset --root VAI_NVS_DATA/phase1/public_set --out processed/public_set --copy-mode hardlink --strict-contest
python -m bts_nvs.prepare --scene raw_scene --out processed_scene --strict-contest
python -m bts_nvs.train --scene processed_scene --preset fast
python -m bts_nvs.train --scene processed_scene --preset fast --disable-pose-normalization -- --viewer.quit-on-train-completion True
python -m bts_nvs.train --scene processed_scene --preset quality-aa --disable-pose-normalization -- --max-num-iterations 30000 --viewer.quit-on-train-completion True
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets processed_scene/target_cameras.json --out submission/scene_id --strict-contest
python -m bts_nvs.evaluate --pred submission/scene_id --gt VAI_NVS_DATA/phase1/public_set/scene_id/test/images --match-by-stem --psnr-max 50
python -m bts_nvs.score_submission --data-root VAI_NVS_DATA/phase1/public_set --submission submission/public_variant --match-by-stem --psnr-max 50 --out metrics/public_variant.json
python -m bts_nvs.package --submission submission --out submission.zip
python -m bts_nvs.validate_submission --data-root VAI_NVS_DATA/phase1/private_set1 --submission submission.zip
```

If no trained Nerfstudio checkpoint is available yet, create a low-cost
submission smoke test with temporal blending between adjacent drone frames:

```powershell
python -m bts_nvs.nearest_view --root VAI_NVS_DATA/phase1/private_set1 --out submission/temporal_blend_private_set1 --selection-mode temporal-blend --blend-weight-policy linear --jpeg-quality 95
python -m bts_nvs.package --submission submission/temporal_blend_private_set1 --out submission_round1.zip
python -m bts_nvs.validate_submission --data-root VAI_NVS_DATA/phase1/private_set1 --submission submission_round1.zip
```

For phase1, this writes exact `image_name` filenames such as
`DJI_..._V.JPG`. This avoids a zero-score failure mode where scenes match but
per-pose images are treated as missing because `.JPG` target names were changed
to `.png`. The `temporal-blend` mode uses the frame number in target filenames
to blend the closest preceding and following train frames. It is a stronger
no-training fallback than copying one nearest view, but it is still not a
replacement for per-scene 3DGS reconstruction.

Non-dry-run training captures the external `ns-train` output to
`<processed_scene>/training.log` by default. Override it with `--log-file` when
you want logs outside the processed scene directory.

For a dry run that prints the external command without running Nerfstudio:

```powershell
python -m bts_nvs.train --scene processed_scene --dry-run
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets target_cameras.json --out submission/scene_id --dry-run
```

## Submission Layout

The package command writes supported image files into the archive:

```text
submission.zip
  scene_001/
    <target_image_name>
    <another_target_image_name>
  scene_002/
    <target_image_name>
```

For VAI phase1, rendered filenames follow the `image_name` column in
`test/test_poses.csv` exactly. The general problem statement also shows
numbered PNG filenames such as `0001.png`; treat that as illustrative unless a
round-specific adapter says otherwise.

Before uploading, validate the folder or ZIP against the BTC CSVs:

```powershell
python -m bts_nvs.validate_submission --data-root VAI_NVS_DATA/phase1/private_set1 --submission submission_round1.zip
```

The validator checks scene folders, exact target filenames, missing or extra
images, image dimensions from `test_poses.csv`, readable image files, RGB mode,
and rejects ZIP members that are not directly under `scene_id/image_name`.

## VAI Phase1 Notes

- `train/sparse/0/images.bin` can contain poses for images not present in
  `train/images`; the converter filters COLMAP registered images to files that
  actually exist in `train/images`.
- `test_poses.csv` labels `tx,ty,tz` as camera translation. Earlier versions of
  this repo treated those values as a world-space camera center, but a public
  probe on `hcm0031` showed the COLMAP/OpenCV `tvec` interpretation was much
  better for Nerfstudio rendering. The adapter now converts `qw,qx,qy,qz` plus
  `tx,ty,tz` as an OpenCV world-to-camera pose into Nerfstudio/OpenGL
  camera-to-world matrices.
- Nerfstudio's default dataparser recenters, orients, and rescales poses. When
  rendering BTC target poses that are already in the same COLMAP coordinate
  frame, train with `--disable-pose-normalization` so target camera paths stay
  in the same frame as the trained model.
- Rendered predictions from Nerfstudio are PNGs by default, while phase1 target
  filenames are often `.JPG`. Preserve exact target names for submission, or the
  evaluator may mark images missing even when scene directories match.
- The low-cost fallback supports `--selection-mode nearest-pose`,
  `--selection-mode temporal-nearest`, and `--selection-mode temporal-blend`.
  `temporal-blend` defaults to `--blend-weight-policy linear`, which is the
  strongest private-set fallback observed so far. `midpoint` had slightly
  higher local public-set PSNR, but the private leaderboard score was lower
  because SSIM/LPIPS regressed.
  On the released public set, temporal blend improved internal PSNR from about
  9.22 dB for pose-nearest copying to about 10.95 dB for linear blending.
  This is local validation, not a guaranteed private-set score.
- When SSIM and LPIPS dependencies are installed, `evaluate` also reports BTC's
  aggregate score: `0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * psnr_norm`, where
  `psnr_norm = clamp(PSNR / --psnr-max, 0, 1)`. The current default is
  `--psnr-max 50`, which matches the observed phase1 leaderboard scale.
- Use `score_submission` on `public_set` variants before spending GPU time on a
  full private render. It reports both aggregate metrics and per-scene metrics,
  which makes regressions easier to localize than BTC's private aggregate score.
- Treat 30,000 iterations as the current phase1 baseline for `splatfacto-big`.
  The private 60,000-iteration run scored `57.85830`, below the otherwise
  equivalent 30,000-iteration run at `58.30090`; all three reported metrics
  regressed. Nerfstudio's Splatfacto optimizer schedules are also configured
  around 30,000 steps. The CLI now warns when an explicit iteration budget
  exceeds that value.
- `--preset quality-aa` is an experimental `splatfacto-big` variant using
  gsplat's antialiased rasterizer. Nerfstudio documents this mode as compensating
  tiny splats when render resolution differs from capture resolution. Score it
  across all five public scenes before using it for private scenes; it is not a
  confirmed leaderboard improvement yet.
