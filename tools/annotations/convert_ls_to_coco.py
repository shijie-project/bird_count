"""Convert a Label Studio keypoint export (all_ls.json) into the COCO-ish
training format used by BirdDataset (see bird_count/datasets/bird.py).

Source (Label Studio):
    [
      {
        "data": {"img": "/data/local-files/?d=images%5C...%5Cfoo.jpg"},
        "annotations": [
          {"result": [
              {"original_width": 1280, "original_height": 720,
               "value": {"x": <percent>, "y": <percent>, "keypointlabels": ["chicken"]}},
              ...
          ]}
        ]
      },
      ...
    ]

Target (all.json):
    {
      "images":      [{"id": 1, "file_name": "foo.jpg", "width": 1280, "height": 720}, ...],
      "annotations": [{"image_id": "foo", "points": [[x_px, y_px], ...]}, ...]
    }

Notes:
- Label Studio stores x/y as PERCENTAGES (0-100) of the image; we convert to
  absolute pixels using each item's original_width/original_height.
- image_id is the image filename stem (no extension) so BirdDataset can match it
  to the image files it lists from images/<split>/.
"""

import argparse
import json
import os
import re
from urllib.parse import unquote


def ls_img_to_filename(img_field: str) -> str:
    """'/data/local-files/?d=images%5C...%5Cfoo.jpg' -> 'foo.jpg'."""
    # strip the query prefix if present
    if "?d=" in img_field:
        img_field = img_field.split("?d=", 1)[1]
    decoded = unquote(img_field)  # %5C -> backslash, %20 -> space, etc.
    decoded = decoded.replace("\\", "/")  # normalise separators
    return os.path.basename(decoded)


# Label Studio prefixes files uploaded through its UI with a short hex hash:
# "/data/upload/2/61701da2-020_axis4_....jpg". The same image referenced through
# LOCAL_FILES_DOCUMENT_ROOT has no such prefix, so the two spellings must fold
# together before anything can be matched against a filename on disk.
_UPLOAD_HASH_RE = re.compile(r"^[0-9a-f]{8}-")


def image_key(file_name: str) -> str:
    """Canonical identity of an image file name, for matching across sources.

    Drops a Label Studio upload hash and case-folds, so
    "61701da2-020_axis4.jpg", "020_axis4.jpg" and "020_AXIS4.jpg" are one image.
    Callers should still prefer an exact file-name match first — some files on
    disk legitimately carry a hash prefix as part of their real name.
    """
    return _UPLOAD_HASH_RE.sub("", file_name, count=1).lower()


def pick_annotation(task: dict) -> list:
    """Return the keypoint result list for a task.

    Uses the first annotation that has results and was not cancelled.
    """
    for ann in task.get("annotations", []):
        if ann.get("was_cancelled"):
            continue
        result = ann.get("result", [])
        if result:
            return result
    # fall back to the first annotation's result (may be empty)
    anns = task.get("annotations", [])
    return anns[0].get("result", []) if anns else []


def task_points(task: dict) -> tuple[list, tuple]:
    """Absolute-pixel keypoints of a task, plus the (width, height) they came from.

    Shared with merge_ls.py so the point counts it reports are exactly the ones
    that will end up in the training JSON.
    """
    width = height = None
    points = []
    for item in pick_annotation(task):
        if item.get("type") and item["type"] != "keypointlabels":
            continue
        val = item.get("value", {})
        if "x" not in val or "y" not in val:
            continue
        ow = item.get("original_width")
        oh = item.get("original_height")
        if ow is None or oh is None:
            continue
        width, height = ow, oh
        points.append([val["x"] / 100.0 * ow, val["y"] / 100.0 * oh])
    return points, (width, height)


def convert(tasks: list) -> dict:
    images = []
    annotations = []

    next_id = 1
    for task in tasks:
        img_field = task.get("data", {}).get("img", "")
        file_name = ls_img_to_filename(img_field)
        stem = os.path.splitext(file_name)[0]

        points, (width, height) = task_points(task)

        # Skip images without any annotation; only keep annotated ones.
        if not points:
            continue

        images.append(
            {
                "id": next_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        annotations.append({"image_id": stem, "points": points})
        next_id += 1

    return {"images": images, "annotations": annotations}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Label Studio export json")
    parser.add_argument("output", help="output json in training format")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        tasks = json.load(f)

    coco = convert(tasks)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)

    n_img = len(coco["images"])
    n_pts = sum(len(a["points"]) for a in coco["annotations"])
    skipped = len(tasks) - n_img
    print(
        f"Wrote {args.output}: {n_img} annotated images, {n_pts} points total, {skipped} unannotated image(s) skipped."
    )


if __name__ == "__main__":
    main()
