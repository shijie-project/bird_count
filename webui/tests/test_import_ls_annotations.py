import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui" / "ops"))

import import_ls_annotations as importer

from webui.schema import get_schema, list_entrypoints


class ImportExternalAnnotationsTests(unittest.TestCase):
    def test_schema_exposes_one_live_ls_operation(self):
        pages = {item["key"]: item["page"] for item in list_entrypoints()}
        self.assertEqual(pages["ls_dedupe"], "label_studio")
        self.assertEqual(pages["ls_import_annotations"], "label_studio")
        schema = get_schema("ls_import_annotations")
        self.assertEqual(schema["page"], "label_studio")
        options = {option["dest"]: option for group in schema["groups"] for option in group["options"]}
        self.assertTrue(options["src"]["required"])
        self.assertIn("images_dir", options)
        self.assertIn("project_id", options)
        self.assertEqual(options["fuzzy_threshold"]["default"], importer.DEFAULT_FUZZY_THRESHOLD)
        self.assertIn("existing_only", options)
        self.assertEqual(options["keypoint_label"]["default"], "chicken")

    def test_rewrites_foreign_path_and_removes_database_ids(self):
        source = [
            {
                "id": 91,
                "data": {"image": "/data/upload/8/61701da2-Bird One.jpg", "camera": "north"},
                "annotations": [
                    {
                        "id": 801,
                        "task": 91,
                        "project": 12,
                        "completed_by": 44,
                        "lead_time": 3.5,
                        "result": [{"id": "point-a", "type": "keypointlabels", "value": {"x": 20, "y": 30}}],
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "Bird One.jpg").write_bytes(b"image")
            tasks, rejected = importer.prepare_tasks(
                source,
                images_dir=temp_dir,
                image_prefix="annotated/images/all/",
            )

        self.assertEqual(rejected, [])
        self.assertEqual(tasks[0]["data"]["img"], "/data/local-files/?d=annotated%2Fimages%2Fall%2FBird%20One.jpg")
        self.assertEqual(tasks[0]["data"]["camera"], "north")
        self.assertNotIn("image", tasks[0]["data"])
        annotation = tasks[0]["annotations"][0]
        self.assertEqual(annotation["lead_time"], 3.5)
        self.assertNotIn("id", annotation)
        self.assertNotIn("task", annotation)
        self.assertNotIn("project", annotation)
        self.assertNotIn("completed_by", annotation)
        self.assertEqual(annotation["result"][0]["id"], "point-a")
        self.assertEqual(annotation["result"][0]["value"]["keypointlabels"], ["chicken"])

    def test_source_label_case_is_normalized_to_destination_label(self):
        annotation = {
            "result": [
                {
                    "type": "keypointlabels",
                    "value": {"x": 20, "y": 30, "keypointlabels": ["Chicken"]},
                }
            ]
        }
        cleaned = importer.clean_annotation(annotation, "chicken")
        self.assertEqual(cleaned["result"][0]["value"]["keypointlabels"], ["chicken"])
        self.assertEqual(annotation["result"][0]["value"]["keypointlabels"], ["Chicken"])

    def test_destination_config_remaps_keypoint_control_names(self):
        config = """<View><KeyPointLabels name="dot-1" toName="img-1">
          <Label value="chicken"/>
        </KeyPointLabels><Image name="img-1" value="$img"/></View>"""
        self.assertEqual(
            importer._destination_keypoint_config(config, "chicken"),
            ("dot-1", "img-1", "chicken"),
        )
        result = importer._normalize_keypoint_results(
            [{"type": "keypointlabels", "from_name": "kp-1", "to_name": "old", "value": {}}],
            "chicken",
            from_name="dot-1",
            to_name="img-1",
        )[0]
        self.assertEqual(result["from_name"], "dot-1")
        self.assertEqual(result["to_name"], "img-1")
        self.assertEqual(result["value"]["keypointlabels"], ["chicken"])

    def test_missing_image_aborts_before_import_by_default(self):
        source = [{"data": {"img": "/foreign/missing.jpg"}, "annotations": []}]
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "present.jpg").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "nothing was imported"):
                importer.prepare_tasks(source, images_dir=temp_dir, image_prefix="images/")

    def test_fuzzy_fallback_ignores_extension_and_separators(self):
        source = [{"data": {"img": "/foreign/Camera01_Frame-001.jpeg"}, "annotations": []}]
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "camera01-frame_001.jpg").write_bytes(b"image")
            Path(temp_dir, "unrelated.jpg").write_bytes(b"image")
            tasks, rejected = importer.prepare_tasks(source, images_dir=temp_dir, image_prefix="images/")

        self.assertEqual(rejected, [])
        self.assertEqual(tasks[0]["data"]["img"], "/data/local-files/?d=images%2Fcamera01-frame_001.jpg")

    def test_fuzzy_fallback_rejects_an_ambiguous_best_name(self):
        source = [{"data": {"img": "/foreign/bird-left.jpg"}, "annotations": []}]
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "bird-left-001.jpg").write_bytes(b"image")
            Path(temp_dir, "bird-left-002.jpg").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "ambiguous fuzzy match"):
                importer.prepare_tasks(source, images_dir=temp_dir, image_prefix="images/")

    def test_existing_task_gets_a_new_annotation_without_creating_a_task(self):
        calls = []

        class Projects:
            def get(self, *, id):
                return types.SimpleNamespace(title="Birds", label_config="")

            def import_tasks(self, **kwargs):
                calls.append(("import", kwargs))
                return {"task_ids": [99]}

        class Tasks:
            def list(self, **kwargs):
                calls.append(("list", kwargs))
                return [{"id": 10, "data": {"img": "/data/upload/7/61701da2-bird.jpg"}, "annotations": []}]

        class Annotations:
            def create(self, **kwargs):
                calls.append(("annotation", kwargs))
                return {"id": 50}

        client = types.SimpleNamespace(projects=Projects(), tasks=Tasks(), annotations=Annotations())
        tasks = [{"data": {"img": "/data/local-files/?d=images%2Fbird.jpg"}, "annotations": [{"result": []}]}]
        summary = importer.apply_to_project(client, 7, tasks)

        self.assertEqual(summary["matched_tasks"], 1)
        self.assertEqual(summary["annotations_added"], 1)
        self.assertEqual(summary["tasks_created"], 0)
        annotation_call = next(payload for kind, payload in calls if kind == "annotation")
        self.assertEqual(annotation_call["id"], 10)
        self.assertEqual(annotation_call["task"], 10)
        self.assertEqual(annotation_call["project"], 7)
        self.assertFalse(any(kind == "import" for kind, _payload in calls))

    def test_missing_task_is_imported_unless_existing_only(self):
        calls = []

        class Projects:
            def get(self, *, id):
                return types.SimpleNamespace(title="Birds", label_config="")

            def import_tasks(self, **kwargs):
                calls.append(kwargs)
                return {"task_ids": [20]}

        client = types.SimpleNamespace(
            projects=Projects(),
            tasks=types.SimpleNamespace(list=lambda **_kwargs: []),
            annotations=types.SimpleNamespace(create=lambda **_kwargs: None),
        )
        tasks = [{"data": {"img": "/data/local-files/?d=images%2Fnew.jpg"}, "annotations": [{"result": []}]}]

        summary = importer.apply_to_project(client, 7, tasks)
        self.assertEqual(summary["tasks_created"], 1)
        self.assertEqual(calls[0]["request"], tasks)

        calls.clear()
        summary = importer.apply_to_project(client, 7, tasks, existing_only=True)
        self.assertEqual(summary["missing_skipped"], 1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
