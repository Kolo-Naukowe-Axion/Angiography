from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.yolo26x.scripts.rewrite_data_yaml import rewrite_data_yaml
from models.yolo26x.scripts.verify_prepared_dataset import verify_prepared_dataset


class VastDatasetScriptTests(unittest.TestCase):
    def test_rewrite_data_yaml_updates_dataset_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            data_yaml = dataset_root / "data.yaml"
            data_yaml.write_text(
                "path: /old/path\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )

            rewrite_data_yaml(dataset_root)

            self.assertEqual(
                data_yaml.read_text(encoding="utf-8").splitlines()[0],
                f"path: {dataset_root.resolve()}",
            )

    def test_verify_prepared_dataset_accepts_matching_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            for split in ("train", "val", "test"):
                (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
                (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
                image_path = dataset_root / "images" / split / f"{split}_001.png"
                label_path = dataset_root / "labels" / split / f"{split}_001.txt"
                image_path.write_bytes(b"fake-image")
                label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            (dataset_root / "data.yaml").write_text(
                f"path: {dataset_root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
            (dataset_root / "summary.json").write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {"image_count": 1},
                            "val": {"image_count": 1},
                            "test": {"image_count": 1},
                        }
                    }
                ),
                encoding="utf-8",
            )

            verify_prepared_dataset(dataset_root)


if __name__ == "__main__":
    unittest.main()
