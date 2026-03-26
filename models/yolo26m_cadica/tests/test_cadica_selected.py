from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo26m_cadica.scripts.cadica_selected_utils import (
    bbox_xywh_to_yolo,
    build_expected_split_index,
    iter_frame_samples,
    normalize_split_manifest,
    prepare_dataset,
)
from models.yolo26m_cadica.scripts.export_cadica_split_manifest import export_split_manifest


PNG_512 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x02\x00"
    b"\x00\x00\x02\x00"
    b"\x08\x02\x00\x00\x00"
    b"\xf4\x78\xd4\xfa"
    b"\x00\x00\x00\x0cIDATx\x9cc`\x18\x05\xa3\x60\x14\x0cw\x00\x00\x08\x00\x01"
    b"\x64\x5f\x17\xdc"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def write_png(path: Path) -> None:
    path.write_bytes(PNG_512)


class CadicaSelectedTests(unittest.TestCase):
    def make_fake_cadica(self, root: Path) -> tuple[Path, Path]:
        cadica_root = root / "datasets" / "cadica" / "CADICA"
        selected = cadica_root / "selectedVideos"
        manifest_path = cadica_root / "splits" / "patient_level_80_10_10_seed42" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        layout = {
            "train": {"p1_v1": {"selected": ["p1_v1_00001", "p1_v1_00003"], "gt": {"p1_v1_00003": "10 20 30 40 p0_20"}}},
            "val": {"p2_v1": {"selected": ["p2_v1_00002"], "gt": {}}},
            "test": {"p3_v2": {"selected": ["p3_v2_00001"], "gt": {"p3_v2_00001": "100 120 20 10 p0_20"}}},
        }

        for split, videos in layout.items():
            for video_key, metadata in videos.items():
                patient_id, video_id = video_key.split("_", 1)
                video_dir = selected / patient_id / video_id
                input_dir = video_dir / "input"
                gt_dir = video_dir / "groundtruth"
                input_dir.mkdir(parents=True, exist_ok=True)
                for frame_number in range(1, 4):
                    write_png(input_dir / f"{patient_id}_{video_id}_{frame_number:05d}.png")
                selected_file = video_dir / f"{patient_id}_{video_id}_selectedFrames.txt"
                selected_file.write_text("\n".join(metadata["selected"]) + "\n", encoding="utf-8")
                if metadata["gt"]:
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    for frame_stem, line in metadata["gt"].items():
                        (gt_dir / f"{frame_stem}.txt").write_text(line + "\n", encoding="utf-8")

        manifest = {
            "dataset_root": str(cadica_root),
            "strategy": "patient-level deterministic shuffle split to avoid leakage",
            "seed": 42,
            "ratios_requested": {"train": 0.8, "val": 0.1, "test": 0.1},
            "patient_counts": {"train": 1, "val": 1, "test": 1},
            "splits": {
                "train": {"patients": ["p1"], "selected_videos": ["p1_v1"]},
                "val": {"patients": ["p2"], "selected_videos": ["p2_v1"]},
                "test": {"patients": ["p3"], "selected_videos": ["p3_v2"]},
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cadica_root, manifest_path

    def test_bbox_conversion(self) -> None:
        x_center, y_center, width, height = bbox_xywh_to_yolo(10, 20, 30, 40, 512, 512)
        self.assertAlmostEqual(x_center, 25 / 512)
        self.assertAlmostEqual(y_center, 40 / 512)
        self.assertAlmostEqual(width, 30 / 512)
        self.assertAlmostEqual(height, 40 / 512)

    def test_selected_frame_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cadica_root, manifest_path = self.make_fake_cadica(Path(tmp_dir))
            samples = list(iter_frame_samples(cadica_root, manifest_path))
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                sorted(sample.prepared_stem for sample in samples),
                [
                    "cadica_p1_v1_00001",
                    "cadica_p1_v1_00003",
                    "cadica_p2_v1_00002",
                    "cadica_p3_v2_00001",
                ],
            )

    def test_prepare_dataset_handles_lesion_and_nonlesion_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cadica_root, manifest_path = self.make_fake_cadica(root)
            output_root = root / "prepared"
            summary = prepare_dataset(
                cadica_root=cadica_root,
                split_manifest=manifest_path,
                output_root=output_root,
                copy_images=True,
            )

            self.assertEqual(summary["splits"]["train"]["image_count"], 2)
            self.assertEqual(summary["splits"]["train"]["positive_image_count"], 1)
            self.assertEqual(summary["splits"]["train"]["negative_image_count"], 1)
            self.assertEqual(
                (output_root / "labels" / "train" / "cadica_p1_v1_00001.txt").read_text(encoding="utf-8"),
                "",
            )
            lesion_label = (output_root / "labels" / "train" / "cadica_p1_v1_00003.txt").read_text(encoding="utf-8")
            self.assertTrue(lesion_label.startswith("0 "))

    def test_expected_index_matches_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cadica_root, manifest_path = self.make_fake_cadica(Path(tmp_dir))
            expected = build_expected_split_index(cadica_root, manifest_path)
            self.assertEqual(expected["train"]["image_count"], 2)
            self.assertEqual(expected["val"]["image_count"], 1)
            self.assertEqual(expected["test"]["image_count"], 1)
            self.assertEqual(expected["train"]["selected_videos"], ["p1_v1"])
            self.assertEqual(expected["val"]["selected_videos"], ["p2_v1"])
            self.assertEqual(expected["test"]["selected_videos"], ["p3_v2"])

    def test_patient_only_manifest_derives_selected_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cadica_root, manifest_path = self.make_fake_cadica(Path(tmp_dir))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for split in payload["splits"].values():
                split.pop("selected_videos")

            normalized = normalize_split_manifest(payload, cadica_root=cadica_root)
            self.assertEqual(normalized["splits"]["train"]["selected_videos"], ["p1_v1"])
            self.assertEqual(normalized["splits"]["val"]["selected_videos"], ["p2_v1"])
            self.assertEqual(normalized["splits"]["test"]["selected_videos"], ["p3_v2"])

    def test_export_split_manifest_writes_dataset_root_and_selected_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cadica_root, manifest_path = self.make_fake_cadica(Path(tmp_dir))
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for split in payload["splits"].values():
                split.pop("selected_videos")
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            output_path = Path(tmp_dir) / "exported" / "manifest.json"
            exported = export_split_manifest(cadica_root, manifest_path, output_path)

            self.assertEqual(exported["dataset_root"], str(cadica_root.resolve()))
            self.assertEqual(exported["splits"]["train"]["selected_videos"], ["p1_v1"])
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
