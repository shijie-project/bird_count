import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "ops"))

import regional_density_error as regional

from webui.runs import Run
from webui.schema import get_schema


class RegionalDensityErrorTests(unittest.TestCase):
    def test_op_is_exposed_on_test_page(self):
        schema = get_schema("regional_density_error")
        self.assertEqual(schema["page"], "test")
        options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
        self.assertEqual(options["grid"]["default"], "4x6")
        self.assertIn("<checkpoint-dir>/regional_density_error/<split>", options["output_dir"]["help"])

    def test_fixed_grid_preserves_prediction_and_gt_totals(self):
        density = np.ones((4, 6), dtype=np.float32)
        points = np.asarray([[5, 5], [55, 35], [30, 20]], dtype=np.float32)
        regions = regional.regional_errors(
            density,
            points,
            2,
            3,
            valid_width=60,
            valid_height=40,
        )

        self.assertEqual(len(regions), 6)
        self.assertAlmostEqual(sum(item["pred"] for item in regions), 24.0)
        self.assertEqual(sum(item["gt"] for item in regions), 3)
        self.assertEqual(regions[0]["gt"], 1)
        self.assertEqual(regions[-1]["gt"], 1)

    def test_run_parser_builds_image_region_and_summary_results(self):
        run = Run("regional_density_error", [], {})
        run._parse("Writing regional error overlays to: C:/out")
        run._parse("  bird: GT 8.0 Pred 10.0 RegionMAE 0.500 Worst 2.000")
        run._parse("  REGION bird r2c3: GT 1 Pred 3.000 Err +2.000 Err% +200.0%")
        run._parse("REGIONAL SUMMARY | Images 1 | Regions 24 | MAE 0.5000 | RMSE 0.7500 | Bias +0.1000")

        self.assertEqual(run.result["overlay_suffix"], "_regional_error.png")
        self.assertEqual(run.result["images"][0]["worst_region"], 2.0)
        self.assertEqual(run.result["regions"][0]["name"], "bird r2c3")
        self.assertEqual(run.result["regions"][0]["err"], 2.0)
        self.assertEqual(run.result["technical"]["Regional MAE"], "0.5000")


if __name__ == "__main__":
    unittest.main()
