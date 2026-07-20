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


def test_latest_btc_requirements_manifest_keeps_received_archive_contract() -> None:
    manifest_path = Path(__file__).resolve().parents[1] / "manifests" / "btc_requirements_2026-07-20.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["overall_sha256"] == "4a5ec5e93ffc0ef7a8dba3815bd5bfea266c16184ea6e592d6d7a261f76b08a8"
    assert manifest["file_count"] == len(manifest["files"]) == 2
    assert manifest["total_bytes"] == sum(entry["size"] for entry in manifest["files"]) == 14_489
    assert manifest["source_archive"] == {
        "path": "Yêu cầu BTC BTS - Markdown.zip",
        "size": 6_072,
        "sha256": "09fa39f08dac690149810be564de82587f454cd4e6817dcc009d5a6b3e789d67",
        "crc_check": "pass",
        "duplicate_members": 0,
        "unsafe_members": 0,
    }
