from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "demo-app" / "scripts" / "prepare_cadica_demo_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_cadica_demo_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
prepare_demo_data = MODULE.prepare_demo_data


def _write_png(path: Path, color: tuple[int, int, int] = (16, 16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=color).save(path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_cadica_demo_data_writes_manifest_and_empty_negative_labels(tmp_path: Path) -> None:
    cadica_root = tmp_path / "CADICA"
    split_manifest = cadica_root / "splits" / "patient_level_80_10_10_seed42" / "manifest.json"
    output_root = tmp_path / "demo_patients"

    manifest = {
        "splits": {
            "train": {"patients": [], "selected_videos": []},
            "val": {"patients": [], "selected_videos": []},
            "test": {"patients": ["p2"], "selected_videos": ["p2_v1", "p2_v2"]},
        }
    }
    _write_text(split_manifest, json.dumps(manifest))

    _write_text(cadica_root / "selectedVideos" / "p2" / "v1" / "p2_v1_selectedFrames.txt", "p2_v1_00001\np2_v1_00002\n")
    _write_png(cadica_root / "selectedVideos" / "p2" / "v1" / "input" / "p2_v1_00001.png")
    _write_png(cadica_root / "selectedVideos" / "p2" / "v1" / "input" / "p2_v1_00002.png")
    _write_text(cadica_root / "selectedVideos" / "p2" / "v1" / "groundtruth" / "p2_v1_00001.txt", "10 20 30 40 lesion\n")

    _write_text(cadica_root / "selectedVideos" / "p2" / "v2" / "p2_v2_selectedFrames.txt", "p2_v2_00007\n")
    _write_png(cadica_root / "selectedVideos" / "p2" / "v2" / "input" / "p2_v2_00007.png")

    summary = prepare_demo_data(
        cadica_root=cadica_root,
        split_manifest=split_manifest,
        output_root=output_root,
        split="test",
        copy_images=True,
        force=False,
    )

    assert summary["sequence_count"] == 2
    assert summary["frame_count"] == 3
    assert summary["positive_frame_count"] == 1
    assert summary["negative_frame_count"] == 2

    payload = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in payload["patients"]] == ["cadica_p2_v1", "cadica_p2_v2"]
    assert all(row["datasetId"] == "cadica" for row in payload["patients"])
    assert all(row["labelType"] == "bbox" for row in payload["patients"])

    positive_label = output_root / "cadica_p2_v1" / "labels" / "p2_v1_00001.txt"
    negative_label = output_root / "cadica_p2_v1" / "labels" / "p2_v1_00002.txt"
    unlabeled_video_label = output_root / "cadica_p2_v2" / "labels" / "p2_v2_00007.txt"

    assert positive_label.read_text(encoding="utf-8").strip() == "0 0.250000 0.500000 0.300000 0.500000"
    assert negative_label.read_text(encoding="utf-8") == ""
    assert unlabeled_video_label.read_text(encoding="utf-8") == ""
