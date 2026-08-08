import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "ops"))

import density_to_ls_labels as density_ls

from webui.schema import get_schema


class DensityToLabelStudioTests(unittest.TestCase):
    CONFIG = """<View><KeyPointLabels name="dot-1" toName="img-1">
      <Label value="chicken" background="red"/>
    </KeyPointLabels><Image name="img-1" value="$img"/></View>"""

    def test_op_is_on_label_studio_page(self):
        schema = get_schema("density_to_ls_labels")
        self.assertEqual(schema["page"], "label_studio")
        options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
        self.assertEqual(options["label"]["default"], "chicken-pred")
        self.assertTrue(options["src"]["required"])

    def test_config_adds_prediction_label_without_removing_manual_label(self):
        xml, from_name, to_name, changed = density_ls.ensure_prediction_label(self.CONFIG)
        self.assertTrue(changed)
        self.assertEqual((from_name, to_name), ("dot-1", "img-1"))
        self.assertIn('value="chicken"', xml)
        self.assertIn('value="chicken-pred"', xml)

        xml_again, _from, _to, changed_again = density_ls.ensure_prediction_label(xml)
        self.assertFalse(changed_again)
        self.assertEqual(xml_again.count('value="chicken-pred"'), 1)

    def test_centroid_becomes_chicken_pred_keypoint(self):
        record = {
            "file_name": "bird.jpg",
            "width": 200,
            "height": 100,
            "regions": [{"id": 3, "count": 1.2, "centroid": [50, 25], "bbox": [40, 20, 20, 10]}],
        }
        result = density_ls.build_results(record, from_name="dot-1", to_name="img-1")[0]
        self.assertEqual(result["value"]["keypointlabels"], ["chicken-pred"])
        self.assertEqual(result["value"]["x"], 25.0)
        self.assertEqual(result["value"]["y"], 25.0)
        self.assertEqual(result["origin"], "prediction")

    def test_existing_task_gets_prediction_and_config_label(self):
        calls = []

        class Projects:
            def get(self, *, id):
                return types.SimpleNamespace(title="Birds", label_config=DensityToLabelStudioTests.CONFIG)

            def update(self, **kwargs):
                calls.append(("config", kwargs))

        task = {"id": 7, "data": {"img": "/data/local-files/?d=images%2Fbird.jpg"}, "predictions": []}
        client = types.SimpleNamespace(
            projects=Projects(),
            tasks=types.SimpleNamespace(list=lambda **_kwargs: [task]),
            predictions=types.SimpleNamespace(
                create=lambda **kwargs: calls.append(("create", kwargs)),
                update=lambda **kwargs: calls.append(("update", kwargs)),
            ),
        )
        records = [{"file_name": "bird.jpg", "width": 100, "height": 100, "regions": []}]
        summary = density_ls.apply_density_labels(client, 4, records)

        self.assertEqual(summary["matched_tasks"], 1)
        self.assertEqual(summary["predictions_created"], 1)
        self.assertEqual(calls[0][0], "config")
        self.assertEqual(calls[1][0], "create")
        self.assertEqual(calls[1][1]["task"], 7)


if __name__ == "__main__":
    unittest.main()
