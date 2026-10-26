"""
Unit and Integration Tests for Week 1 & Week 2 Implementation
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from classical_pipeline.localization import find_optic_disk, find_fovea
from evaluation.metrics import pixel_metrics, localization_distance
from dl_pipeline.data_utils import (
    create_synthetic_drive_sample,
    discover_dataset_samples,
)
from dl_pipeline.dataset import RetinalVesselDataset, split_samples


class TestWeek1MetricsAndLocalization(unittest.TestCase):
    def test_localization_stubs_raise_not_implemented(self):
        dummy_vessel = np.zeros((100, 100), dtype=np.uint8)
        dummy_green = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaises(NotImplementedError):
            find_optic_disk(dummy_vessel, dummy_green)

        with self.assertRaises(NotImplementedError):
            find_fovea(dummy_vessel, (50, 50))

    def test_pixel_metrics_perfect_score(self):
        gt = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)

        res = pixel_metrics(pred, gt)
        self.assertAlmostEqual(res["accuracy"], 1.0)
        self.assertAlmostEqual(res["sensitivity"], 1.0)
        self.assertAlmostEqual(res["specificity"], 1.0)

    def test_pixel_metrics_all_negative(self):
        gt = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        pred = np.array([[0, 0], [0, 0]], dtype=np.uint8)

        res = pixel_metrics(pred, gt)
        self.assertAlmostEqual(res["accuracy"], 1.0)
        self.assertAlmostEqual(res["specificity"], 1.0)
        self.assertAlmostEqual(res["sensitivity"], 0.0)

    def test_pixel_metrics_probabilities_thresholding(self):
        gt = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        pred = np.array([[0.8, 0.2], [0.1, 0.4]], dtype=np.float32)

        res = pixel_metrics(pred, gt, threshold=0.5)
        self.assertAlmostEqual(res["accuracy"], 1.0)
        self.assertAlmostEqual(res["sensitivity"], 1.0)
        self.assertAlmostEqual(res["specificity"], 1.0)

    def test_localization_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        dist = localization_distance(p1, p2)
        self.assertAlmostEqual(dist, 5.0)
        self.assertAlmostEqual(localization_distance((10.5, 20.0), (10.5, 20.0)), 0.0)


class TestWeek2DatasetAndUtils(unittest.TestCase):
    def test_split_samples_ratios(self):
        dummy_samples = [(f"img_{i}.png", f"mask_{i}.png") for i in range(20)]
        train_split = split_samples(dummy_samples, split="train", seed=42)
        val_split = split_samples(dummy_samples, split="val", seed=42)
        test_split = split_samples(dummy_samples, split="test", seed=42)

        self.assertEqual(len(train_split), 14)
        self.assertEqual(len(val_split), 3)
        self.assertEqual(len(test_split), 3)

        all_combined = set(train_split) | set(val_split) | set(test_split)
        self.assertEqual(len(all_combined), 20)
        self.assertTrue(set(train_split).isdisjoint(set(val_split)))
        self.assertTrue(set(train_split).isdisjoint(set(test_split)))
        self.assertTrue(set(val_split).isdisjoint(set(test_split)))

    def test_dataset_item_shape_and_range(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "DRIVE"
            create_synthetic_drive_sample(root)

            ds = RetinalVesselDataset(root_dir=root, split="all", img_size=512)
            self.assertGreater(len(ds), 0)

            sample = ds[0]
            self.assertIn("image", sample)
            self.assertIn("mask", sample)
            self.assertIsInstance(sample["image"], torch.Tensor)
            self.assertIsInstance(sample["mask"], torch.Tensor)

            self.assertEqual(sample["image"].shape, (3, 512, 512))
            self.assertEqual(sample["mask"].shape, (1, 512, 512))

            self.assertGreaterEqual(float(sample["image"].min()), 0.0)
            self.assertLessEqual(float(sample["image"].max()), 1.0)
            self.assertTrue((sample["mask"] == 0).logical_or(sample["mask"] == 1).all())

    def test_discover_dataset_samples_drive_scoping(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            drive_dir = tmp_path / "DRIVE"
            create_synthetic_drive_sample(drive_dir)

            other_dir = tmp_path / "OTHER" / "images"
            other_dir.mkdir(parents=True, exist_ok=True)
            (other_dir / "other_01.tif").write_bytes(b"dummy")

            discovered = discover_dataset_samples(tmp_path, dataset_name="DRIVE")
            self.assertGreater(len(discovered), 0)
            for img_p, mask_p in discovered:
                self.assertTrue(Path(img_p).is_relative_to(drive_dir))



if __name__ == "__main__":
    unittest.main()
