import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "ops"))

import density_regions

from webui.runs import Run
from webui.schema import get_schema


class DensityRegionErrorTests(unittest.TestCase):
    def test_schema_exposes_blob_error_controls_without_fixed_grid(self):
        schema = get_schema("density_regions")
        self.assertEqual(schema["page"], "test")
        self.assertEqual([group["title"] for group in schema["groups"]], ["data", "model", "regions", "output"])
        self.assertEqual(schema["label"], "Blob density error")
        options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
        self.assertEqual(options["data_path"]["default"], "../data/annotated")
        self.assertEqual(options["split"]["default"], "val")
        self.assertEqual(options["num_workers"]["default"], 2)
        self.assertEqual(options["limit"]["default"], 0)
        self.assertNotIn("paths", options)
        self.assertNotIn("annotations_json", options)
        self.assertNotIn("recursive", options)
        self.assertEqual(options["label_min_error"]["default"], 0.5)
        self.assertEqual(options["good_error"]["default"], 1.0)
        self.assertEqual(options["point_snap"]["default"], 2)
        self.assertEqual(options["top_regions"]["default"], 100)
        self.assertEqual(options["no_density_map"]["default"], False)
        self.assertEqual(options["seed"]["default"], 42)
        self.assertIsNone(options["output_dir"]["default"])
        self.assertNotIn("<split>", options["output_dir"]["help"])
        self.assertFalse(options["output_dir"]["required"])
        self.assertNotIn("send_to_label_studio", options)
        self.assertNotIn("project_id", options)
        self.assertNotIn("grid", options)

    def test_points_are_counted_in_blob_and_background_separately(self):
        labels = np.asarray([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
        regions = [
            {"id": 1, "count": 1.5},
            {"id": 2, "count": 0.25},
        ]
        points = np.asarray([[5, 5], [15, 5], [15, 15], [25, 5]], dtype=np.float32)

        outside = density_regions.attach_blob_errors(labels, regions, points, width=30, height=20)

        self.assertEqual(outside, 1)
        self.assertEqual(regions[0]["gt_count"], 2)
        self.assertAlmostEqual(regions[0]["error"], -0.5)
        self.assertEqual(regions[1]["gt_count"], 1)
        self.assertAlmostEqual(regions[1]["error"], -0.75)

    def test_point_can_snap_across_small_blob_boundary(self):
        labels = np.asarray([[1, 0, 0]], dtype=np.int32)
        regions = [{"id": 1, "count": 1.0}]
        points = np.asarray([[15, 5]], dtype=np.float32)

        outside = density_regions.attach_blob_errors(labels, regions, points, width=30, height=10, point_snap=1)

        self.assertEqual(outside, 0)
        self.assertEqual(regions[0]["gt_count"], 1)
        self.assertEqual(regions[0]["error"], 0.0)

    def test_annotation_loader_matches_ls_upload_prefix(self):
        payload = {
            "images": [{"id": 7, "file_name": "a1b2c3d4-Chicken_01.JPG"}],
            "annotations": [{"image_id": 7, "points": [[1, 2]]}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = density_regions.load_annotation_points(path)

        self.assertEqual(result["chicken_01"].tolist(), [[1.0, 2.0]])

    def test_image_key_preserves_dots_inside_extensionless_annotation_id(self):
        annotation_id = "a1b2c3d4-camera.mkv_20260521_110227.753"
        image_name = "camera.mkv_20260521_110227.753.jpg"

        self.assertEqual(density_regions._image_key(annotation_id), density_regions._image_key(image_name))
        self.assertEqual(density_regions._image_key("a1b2c3d4-mask_f0.50_clean_mask"), "mask_f0.50_clean_mask")

    def test_run_parser_builds_blob_rows_and_summary(self):
        run = Run("density_regions", [], {})
        run._parse("Writing blob error overlays to: C:/out")
        run._parse("  bird.jpg: GT 8.0 Pred 10.0 BlobMAE 0.500 Worst 2.000 GTOutside 1")
        run._parse("  BLOB bird.jpg b3: GT 1 Pred 3.000 Err +2.000 Err% +200.0%")
        run._parse("BLOB SUMMARY | Images 1 | Blobs 4 | MAE 0.5000 | RMSE 0.7500 | Bias +0.1000 | GTOutside 1")

        self.assertEqual(run.result["overlay_suffix"], "_regions.png")
        self.assertEqual(run.result["images"][0]["worst_blob"], 2.0)
        self.assertEqual(run.result["regions"][0]["err"], 2.0)
        self.assertEqual(run.result["technical"]["GT outside blobs"], "1")


if __name__ == "__main__":
    unittest.main()
