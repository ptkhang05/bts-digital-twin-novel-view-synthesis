from __future__ import annotations

import json
from pathlib import Path


def test_round2_manifest_keeps_verified_dataset_contract() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / "vai_nvs_round2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    scenes = manifest["scenes"]
    assert manifest["dataset_id"] == "VAI_NVS_DATA_ROUND2"
    assert manifest["scene_count"] == len(scenes) == 7
    assert manifest["train_image_count"] == 1_653
    assert manifest["target_image_count"] == 386
    assert sum(scene["colmap_extra_registered_image_count"] for scene in scenes) == 298
    assert sum("SIMPLE_RADIAL" in scene["camera_models"] for scene in scenes) == 5
