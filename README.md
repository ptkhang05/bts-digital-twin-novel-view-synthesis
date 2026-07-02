# BTS Digital Twin Novel View Synthesis Baseline

This repository provides a practical baseline pipeline for posed drone-image
novel view synthesis of BTS scenes. It prepares COLMAP/NeRF-style scene data for
Nerfstudio, trains a 3D Gaussian Splatting model with `splatfacto`, renders RGB
target views as PNG image sequences, and evaluates predictions against holdout
ground truth when available.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Install Nerfstudio separately in the environment used for training/rendering.
The local tests do not require Nerfstudio because this package treats
`ns-train` and `ns-render` as external commands.

## Scene Layout

Supported raw scene inputs:

- Nerfstudio/NeRF-style `train_cameras.json` or `transforms.json` plus `images/`.
- COLMAP sparse reconstruction in `sparse/0`, `sparse`, or `colmap/sparse/0`
  plus `images/`.

Target views use the same JSON camera schema as `transforms.json`.

## Commands

```powershell
python -m bts_nvs.prepare --scene raw_scene --out processed_scene
python -m bts_nvs.train --scene processed_scene --preset fast
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets target_cameras.json --out submission/scene_id
python -m bts_nvs.evaluate --pred submission/scene_id --gt holdout_gt
```

For a dry run that prints the external command without running Nerfstudio:

```powershell
python -m bts_nvs.train --scene processed_scene --dry-run
python -m bts_nvs.render --checkpoint outputs/.../config.yml --targets target_cameras.json --out submission/scene_id --dry-run
```
