import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "annotations"))

import regions_to_label_studio as regions_ls

from webui.schema import get_schema


class RegionsToLabelStudioTests(unittest.TestCase):
    def test_webui_exposes_live_import_as_a_checkbox(self):
        for key in ("density_regions", "regions_to_label_studio"):
            schema = get_schema(key)
            options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
            self.assertIn("send_to_label_studio", options)
            self.assertEqual(options["send_to_label_studio"]["kind"], "bool")
            self.assertIn("project_id", options)

    def test_convert_writes_prediction_tasks_and_returns_them(self):
        manifest = {
            "params": {"mode": "blobs"},
            "images": [
                {
                    "file_name": "one.jpg",
                    "width": 200,
                    "height": 100,
                    "regions": [
                        {"id": 1, "count": 3.2, "bbox": [20, 10, 80, 40]},
                        {"id": 2, "count": 0.8, "bbox": [0, 0, 10, 10]},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / "regions.json"
            dst = Path(temp_dir) / "regions_ls.json"
            src.write_text(json.dumps(manifest), encoding="utf-8")

            tasks = regions_ls.convert(src, dst, image_prefix="images/", min_count=1.5, project_id=9)

            self.assertEqual(tasks, json.loads(dst.read_text(encoding="utf-8")))
            self.assertEqual(tasks[0]["data"]["img"], "/data/local-files/?d=images%2Fone.jpg")
            results = tasks[0]["predictions"][0]["result"]
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["value"]["rectanglelabels"], ["3"])

    def test_live_import_updates_config_then_commits_tasks(self):
        calls = []

        class Projects:
            def get(self, *, id):
                calls.append(("get", id))
                return types.SimpleNamespace(title="Birds")

            def update(self, *, id, label_config, show_collab_predictions):
                calls.append(("update", id, label_config, show_collab_predictions))

            def import_tasks(self, *, id, request, commit_to_project, return_task_ids):
                calls.append(("import", id, request, commit_to_project, return_task_ids))
                return {"task_ids": [101]}

        class Client:
            def __init__(self, *, base_url, api_key):
                calls.append(("client", base_url, api_key))
                self.projects = Projects()

        fake_sdk = types.SimpleNamespace(LabelStudio=Client)
        tasks = [{"data": {"img": "image.jpg"}, "predictions": []}]
        with (
            patch.dict(sys.modules, {"label_studio_sdk": fake_sdk}),
            patch.dict(os.environ, {"LABEL_STUDIO_API_KEY": "secret"}),
        ):
            response = regions_ls.import_to_label_studio(tasks, project_id=7, url="localhost:8080/")

        self.assertEqual(response, {"task_ids": [101]})
        self.assertEqual(calls[0], ("client", "http://localhost:8080", "secret"))
        self.assertEqual(calls[1], ("get", 7))
        self.assertEqual(calls[2][0:2], ("update", 7))
        self.assertIn("<RectangleLabels", calls[2][2])
        self.assertEqual(calls[3], ("import", 7, tasks, True, True))


if __name__ == "__main__":
    unittest.main()
