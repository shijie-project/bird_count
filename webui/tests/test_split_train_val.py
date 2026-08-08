import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "annotations"))

import split_train_val as splitter


class SplitTrainValTests(unittest.TestCase):
    def test_axis_key_supports_regular_and_dense_scene_names(self):
        self.assertEqual(splitter.axis_key("020_axis4_camera.jpg"), "axis4")
        self.assertEqual(splitter.axis_key("13_dense_scene_3_axis-CAMERA_clean.png"), "axis3")
        self.assertEqual(splitter.axis_key("03_local_dense_2_axis-CAMERA_clean.png"), "axis2")
        self.assertIsNone(splitter.axis_key("001.jpg"))

    def test_val_contains_every_axis_then_random_fill(self):
        images = [
            {"id": 1, "file_name": "a_axis1_one.jpg"},
            {"id": 2, "file_name": "b_axis1_two.jpg"},
            {"id": 3, "file_name": "c_axis2_one.jpg"},
            {"id": 4, "file_name": "d_axis2_two.jpg"},
            {"id": 5, "file_name": "e_axis3_one.jpg"},
            {"id": 6, "file_name": "f_axis4_one.jpg"},
            {"id": 7, "file_name": "legacy_001.jpg"},
            {"id": 8, "file_name": "legacy_002.jpg"},
            {"id": 9, "file_name": "legacy_003.jpg"},
            {"id": 10, "file_name": "legacy_004.jpg"},
        ]

        train, val, axes = splitter.split_images(images, val_ratio=0.5, seed=42)

        self.assertEqual(len(val), 5)
        self.assertEqual(len(train), 5)
        self.assertEqual(axes, ["axis1", "axis2", "axis3", "axis4"])
        self.assertEqual({splitter.axis_key(item["file_name"]) for item in val} - {None}, set(axes))
        self.assertEqual({item["id"] for item in train} | {item["id"] for item in val}, set(range(1, 11)))

    def test_axis_coverage_overrides_too_small_ratio(self):
        images = [{"id": axis, "file_name": f"sample_axis{axis}_one.jpg"} for axis in range(1, 5)]

        train, val, _ = splitter.split_images(images, val_ratio=0.1, seed=0)

        self.assertEqual(len(val), 4)
        self.assertEqual(train, [])

    def test_split_is_reproducible(self):
        images = [{"id": index, "file_name": f"sample_axis{index % 4 + 1}_{index}.jpg"} for index in range(30)]

        first = splitter.split_images(images, val_ratio=0.2, seed=9)
        second = splitter.split_images(images, val_ratio=0.2, seed=9)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
