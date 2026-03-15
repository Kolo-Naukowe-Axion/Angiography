from __future__ import annotations

import unittest
from pathlib import Path

from models.yolo26x import train


class Yolo26xTrainDefaultsTests(unittest.TestCase):
    def test_default_paths_point_to_yolo26x_workflow(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.assertEqual(train.REPO_ROOT, repo_root)
        self.assertEqual(
            train.DEFAULT_DATA,
            repo_root / "datasets" / "cadica" / "derived" / "yolo26_selected_seed42" / "data.yaml",
        )
        self.assertEqual(train.DEFAULT_PROJECT, repo_root / "models" / "yolo26x" / "runs")
        self.assertEqual(train.DEFAULT_RUN_NAME, "cadica_selected_seed42_4090")

    def test_parser_defaults_match_cli_contract(self) -> None:
        parser = train.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.model, "yolo26x.pt")
        self.assertEqual(args.epochs, 300)
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.patience, 50)
        self.assertIsNone(args.device)
        self.assertIsNone(args.resume)
        self.assertIsNone(args.amp)
        self.assertIsNone(args.batch)
        self.assertIsNone(args.workers)
        self.assertEqual(args.data, train.DEFAULT_DATA)
        self.assertEqual(args.project, train.DEFAULT_PROJECT)
        self.assertEqual(args.name, train.DEFAULT_RUN_NAME)

    def test_device_profiles(self) -> None:
        self.assertEqual(train.default_batch_for_device("0"), 16)
        self.assertEqual(train.default_workers_for_device("0"), 12)
        self.assertTrue(train.default_cache_for_device("0"))
        self.assertEqual(train.default_batch_for_device("mps"), 4)
        self.assertEqual(train.default_workers_for_device("mps"), 2)
        self.assertFalse(train.default_cache_for_device("mps"))
        self.assertEqual(train.default_batch_for_device("cpu"), 4)
        self.assertEqual(train.default_workers_for_device("cpu"), 2)
        self.assertFalse(train.default_cache_for_device("cpu"))


if __name__ == "__main__":
    unittest.main()
