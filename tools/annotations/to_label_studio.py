"""Convert point annotations to Label Studio keypoint-task JSON.

Input format (COCO-ish):
    {
      "images":      [{"id": int, "file_name": str, "width": int, "height": int}, ...],
      "annotations": [{"image_id": str, "points": [[x, y], ...]}, ...]
    }

Output: a Label Studio task array, importable via the Import UI or
`/api/projects/<id>/import`.

Usage:
    python -m tools.annotations.to_label_studio \\
        --src train.json \\
        --dst train_ls.json \\
        --label chicken \\
        --image-prefix "images\\all\\"

The default `--image-prefix` is `images\\all\\` so the generated
`data.img` URL matches what Label Studio writes natively on Windows
(backslash separators, no `..` traversal). This assumes LS is started
with `LOCAL_FILES_DOCUMENT_ROOT=<repo>/code/data`.

Below is the annotation template, do not delete this:
<View>
  <KeyPointLabels name="dot-1" toName="img-1" strokeWidth="5">
    <Label value="chicken" background="red" selected="true"/>
  </KeyPointLabels>
  <Image name="img-1" value="$img" zoomControl="true" smoothing="false" />
</View>
"""

from __future__ import annotations

import argparse
import json
import random
import string
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote


# Label Studio's per-result `id` uses URL-safe characters; 10 chars at 64-symbol
# alphabet ≈ 60 bits of entropy — far more than enough for per-task uniqueness.
_ID_ALPHABET = string.ascii_letters + string.digits + "-_"
_ID_LEN = 10

# LS's UI-rendered keypoint marker width, expressed as a percentage of the
# canvas (~0.376%). Match LS's default so re-imports look identical to
# fresh annotations.
DEFAULT_KP_WIDTH = 0.37650602409638556

# Match LS's date format exactly: ISO 8601 with microseconds + literal 'Z'.
_LS_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Default location of images relative to LS's LOCAL_FILES_DOCUMENT_ROOT.
# Backslash separators match what Label Studio on Windows writes natively,
# producing URL-encoded paths like `images%5Call%5C006.jpg` (%5C = '\\').
# Override per-platform with --image-prefix if LS runs on Linux/macOS or if
# the document root sits elsewhere.
DEFAULT_IMAGE_PREFIX = "images\\all\\"


def random_short_id(length: int = _ID_LEN) -> str:
    """LS-style 10-char base-64-ish id (not security-grade — just uniqueness)."""
    return "".join(random.choices(_ID_ALPHABET, k=length))


def _normalize_prefix(prefix: str) -> str:
    """Guarantee a trailing separator so `prefix + filename` never bleeds together.

    If the prefix already contains a backslash, append `\\`; otherwise append `/`.
    This way a user passing `images\\all` gets `images\\all\\`, and a user passing
    `images/all` gets `images/all/` — no mixed-separator URLs.
    """
    if not prefix:
        return ""
    if prefix.endswith(("/", "\\")):
        return prefix
    sep = "\\" if "\\" in prefix else "/"
    return prefix + sep


def _build_result(x_pct: float, y_pct: float, img_w: int, img_h: int, label: str, kp_width: float) -> dict:
    """One keypoint entry that goes inside annotations[].result."""
    return {
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": x_pct,
            "y": y_pct,
            "width": kp_width,
            "keypointlabels": [label],
        },
        "id": random_short_id(),
        "from_name": "dot-1",
        "to_name": "img-1",
        "type": "keypointlabels",
        "origin": "manual",
    }


def _build_annotation(
    annotation_id: int, task_id: int, results: list[dict], now: str, *, user_id: int, project_id: int
) -> dict:
    return {
        "id": annotation_id,
        "completed_by": user_id,
        "result": results,
        "was_cancelled": False,
        "ground_truth": False,
        "created_at": now,
        "updated_at": now,
        "draft_created_at": now,
        "lead_time": 0.0,
        "prediction": {},
        "result_count": len(results),
        "unique_id": str(uuid.uuid4()),
        "import_id": None,
        "last_action": None,
        "bulk_created": False,
        "task": task_id,
        "project": project_id,
        "updated_by": user_id,
        "parent_prediction": None,
        "parent_annotation": None,
        "last_created_by": None,
    }


def _build_task(task_id: int, annotation: dict, data_img: str, now: str, *, user_id: int, project_id: int) -> dict:
    return {
        "id": task_id,
        "annotations": [annotation],
        "drafts": [],
        "predictions": [],
        "data": {"img": data_img},
        "meta": {},
        "created_at": now,
        "updated_at": now,
        "allow_skip": True,
        "inner_id": task_id,
        "total_annotations": 1,
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "project": project_id,
        "updated_by": user_id,
        "comment_authors": [],
    }


def _local_files_url(image_prefix: str, file_name: str) -> str:
    """Build LS's `/data/local-files/?d=<urlencoded path>` URL."""
    return "/data/local-files/?d=" + quote(image_prefix + file_name, safe="")


def convert(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    label: str = "chicken",
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    kp_width: float = DEFAULT_KP_WIDTH,
    project_id: int = 4,
    user_id: int = 1,
    indent: int | None = None,
    progress_every: int = 500,
) -> None:
    """Read the source annotations, build LS tasks, write the output JSON.

    `indent=None` produces compact (single-line) output — faster + smaller for
    big imports. Pass `indent=2` if you want a human-readable file.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with src_path.open(encoding="utf-8") as f:
        src = json.load(f)

    # Map "file_name without extension" -> image record. `annotations[].image_id`
    # is the stem (e.g. "001"), while `images[].file_name` is "001.jpg" etc.
    images_by_stem = {Path(img["file_name"]).stem: img for img in src["images"]}

    now = datetime.now(UTC).strftime(_LS_TIME_FMT)
    image_prefix = _normalize_prefix(image_prefix)

    tasks: list[dict] = []
    skipped = 0
    total_anns = len(src["annotations"])
    next_annotation_id = 1  # distinct id-space from task_id (was a real bug)

    for task_idx, ann in enumerate(src["annotations"], start=1):
        key = str(ann["image_id"])
        img = images_by_stem.get(key)
        if img is None:
            print(f"[warn] annotation image_id={key!r} has no matching image, skipped")
            skipped += 1
            continue

        W, H = img["width"], img["height"]
        # Convert absolute pixel coords -> percent (Label Studio stores 0–100).
        results = [
            _build_result(
                x_pct=x / W * 100.0,
                y_pct=y / H * 100.0,
                img_w=W,
                img_h=H,
                label=label,
                kp_width=kp_width,
            )
            for x, y in ann["points"]
        ]

        annotation = _build_annotation(
            annotation_id=next_annotation_id,
            task_id=task_idx,
            results=results,
            now=now,
            user_id=user_id,
            project_id=project_id,
        )
        next_annotation_id += 1

        task = _build_task(
            task_id=task_idx,
            annotation=annotation,
            data_img=_local_files_url(image_prefix, img["file_name"]),
            now=now,
            user_id=user_id,
            project_id=project_id,
        )
        tasks.append(task)

        if progress_every and task_idx % progress_every == 0:
            print(f"  ... {task_idx}/{total_anns} tasks built")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=indent)

    total_pts = sum(t["annotations"][0]["result_count"] for t in tasks)
    print(f"[ok] wrote {len(tasks)} tasks, {total_pts} keypoints -> {dst_path}")
    if skipped:
        print(f"[ok] skipped {skipped} annotations (no matching image)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert point annotations to Label Studio keypoint JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", required=True, help="source annotation JSON")
    ap.add_argument("--dst", required=True, help="output Label Studio JSON")
    ap.add_argument("--label", default="chicken", help="keypoint label name")
    ap.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help=(
            "path (relative to LS's LOCAL_FILES_DOCUMENT_ROOT) prepended to file_name "
            "in data.img. Default uses '\\\\' separators to match LS-on-Windows native "
            "URLs; on Linux/macOS pass e.g. 'images/all/'. Trailing separator auto-added."
        ),
    )
    ap.add_argument(
        "--kp-width",
        type=float,
        default=DEFAULT_KP_WIDTH,
        help="visual marker size in LS UI (percentage of width)",
    )
    ap.add_argument("--project-id", type=int, default=4)
    ap.add_argument("--user-id", type=int, default=1)
    ap.add_argument(
        "--indent",
        type=int,
        default=None,
        help="JSON output indent (omit for compact single-line; 2 for human-readable)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print progress every N tasks (0 = silent)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        src_path=args.src,
        dst_path=args.dst,
        label=args.label,
        image_prefix=args.image_prefix,
        kp_width=args.kp_width,
        project_id=args.project_id,
        user_id=args.user_id,
        indent=args.indent,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
