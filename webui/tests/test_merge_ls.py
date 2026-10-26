import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "annotations"))

import merge_ls


class MergeLabelStudioTests(unittest.TestCase):
    def test_report_lists_every_image_added_as_an_empty_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir)
            for name in ("already-labelled.jpg", "empty-a.jpg", "empty-b.png"):
                (images_dir / name).write_bytes(b"image fixture")

            tasks = [
                {
                    "id": 1,
                    "data": {"img": "/data/local-files/?d=annotated/images/all/already-labelled.jpg"},
                    "annotations": [],
                }
            ]
            result = merge_ls.add_missing_images(
                tasks,
                str(images_dir),
                tasks[0],
                "annotated\\images\\all\\",
            )
            out_tasks, n_images, n_covered, added_names, n_unmatched, n_twins, ambiguous = result

            self.assertEqual(n_images, 3)
            self.assertEqual(n_covered, 1)
            self.assertEqual(added_names, ["empty-a.jpg", "empty-b.png"])
            self.assertEqual((n_unmatched, n_twins, ambiguous), (0, 0, set()))

            stream = io.StringIO()
            roster = (str(images_dir), n_images, n_covered, added_names, n_unmatched, n_twins, ambiguous)
            with redirect_stdout(stream):
                merge_ls._report([("input.json", 1, 0)], "output.json", out_tasks, 0, False, roster)

            report = stream.getvalue()
            self.assertIn("1 already in input exports, 2 added empty", report)
            self.assertIn("Added empty tasks from --images-dir", report)
            self.assertIn("- empty-a.jpg", report)
            self.assertIn("- empty-b.png", report)


if __name__ == "__main__":
    unittest.main()
