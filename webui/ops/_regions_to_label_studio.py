"""Turn density regions into a Label Studio pre-annotation layer.

Internal helper for `density_regions.py`. Reads its `regions.json` and emits Label
Studio tasks carrying a *prediction* made of one rectangle per region, labeled
with how many chickens the model thinks are inside it. You then place keypoints
by hand on top, with the predicted counts visible on the canvas as you go.

The rectangles are predictions, not annotations, so they stay out of your way:
they render as a separate layer and your keypoints are what gets saved. Even if
the project has "use predictions to prelabel" enabled and the rectangles land in
your annotation, `convert_ls_to_coco.py` keeps only `keypointlabels` results, so
nothing leaks into the training JSON.

Label Studio has no way to draw an arbitrary number on a region, so the count is
carried three ways: the region's *label* is a count bucket ("3", "10-19", ...)
and shows on the canvas, the exact value sits in the region's `meta.text` in the
side panel, and full precision stays in `regions.json`.

The WebUI intentionally does not expose this module as a separate operation.
Tick `--send-to-label-studio` on Density regions to convert and import in the
same run.

The labeling config MUST contain both controls — the `KeyPointLabels` you already
use and the `RectangleLabels` these predictions target — or the import is
rejected with "No control tag found". Print it with `--print-config` rather than
copying it by hand: the bucket labels are generated from the same table this
module uses, so they cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dotenv


ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(ROOT / ".env")


# `tools/annotations.py` shadows the `tools/annotations/` directory, so import
# the existing task helpers from their directory explicitly.
sys.path.insert(0, str(ROOT / "tools" / "annotations"))
from to_label_studio import DEFAULT_IMAGE_PREFIX, local_files_url, ls_timestamp, random_short_id  # noqa: E402


# Control names, matching the config `--print-config` prints. `img-1` / `dot-1`
# are the names the existing keypoint template already uses.
IMAGE_NAME = "img-1"
KEYPOINT_NAME = "dot-1"
RECTANGLE_NAME = "region-1"

MODEL_VERSION = "density-regions"

# Count buckets. A region's label has to come from a fixed set declared in the
# labeling config, so the continuous count is bucketed: exact for the small
# counts you can verify at a glance, ranges once "how many exactly" stops being
# answerable by eye anyway. `(upper_bound, label, color)`, first match wins;
# the last entry is the catch-all.
COUNT_BUCKETS: tuple[tuple[float, str, str], ...] = (
    (1.5, "1", "#3498db"),
    (2.5, "2", "#1abc9c"),
    (3.5, "3", "#2ecc71"),
    (4.5, "4", "#f1c40f"),
    (5.5, "5", "#e67e22"),
    (9.5, "6-9", "#e74c3c"),
    (19.5, "10-19", "#c0392b"),
    (float("inf"), "20+", "#8e44ad"),
)


def bucket_for(count: float) -> str:
    """Label for a region holding `count` chickens."""
    for upper, label, _color in COUNT_BUCKETS:
        if count < upper:
            return label
    return COUNT_BUCKETS[-1][1]  # unreachable: the table ends at infinity


def labeling_config(keypoint_label: str = "chicken") -> str:
    """The project's Labeling Interface XML, with bucket labels generated here.

    Keeps the keypoint control first so it stays the default tool — you want a
    click to drop a point, not to start dragging a box.
    """
    bucket_labels = "\n".join(
        f'      <Label value="{label}" background="{color}"/>' for _u, label, color in COUNT_BUCKETS
    )
    return f"""<View>
  <KeyPointLabels name="{KEYPOINT_NAME}" toName="{IMAGE_NAME}" strokeWidth="5">
    <Label value="{keypoint_label}" background="red" selected="true"/>
  </KeyPointLabels>
  <RectangleLabels name="{RECTANGLE_NAME}" toName="{IMAGE_NAME}" opacity="0.1" strokeWidth="2" canRotate="false">
{bucket_labels}
  </RectangleLabels>
  <Image name="{IMAGE_NAME}" value="$img" zoomControl="true" smoothing="false"/>
</View>
"""


def _build_region_result(region: dict, img_w: int, img_h: int) -> dict:
    """One rectangle result. `bbox` is [x, y, w, h] in source pixels; LS wants
    percentages of the image, with the origin at the top-left corner."""
    x, y, w, h = region["bbox"]
    count = region["count"]
    return {
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": x / img_w * 100.0,
            "y": y / img_h * 100.0,
            "width": w / img_w * 100.0,
            "height": h / img_h * 100.0,
            "rotation": 0,
            "rectanglelabels": [bucket_for(count)],
        },
        "id": random_short_id(),
        "from_name": RECTANGLE_NAME,
        "to_name": IMAGE_NAME,
        "type": "rectanglelabels",
        "origin": "prediction",
        # Shown in the region list; the only place the un-bucketed count appears
        # inside Label Studio.
        "meta": {"text": [f"~{count:.1f} chickens (region #{region['id']})"]},
    }


def _build_task(task_id: int, data_img: str, results: list[dict], now: str, *, project_id: int) -> dict:
    return {
        "id": task_id,
        "annotations": [],
        "drafts": [],
        "predictions": [
            {
                "model_version": MODEL_VERSION,
                "created_ago": "0 minutes",
                "result": results,
                "created_at": now,
                "updated_at": now,
                "task": task_id,
            }
        ],
        "data": {"img": data_img},
        "meta": {},
        "created_at": now,
        "updated_at": now,
        "allow_skip": True,
        "inner_id": task_id,
        "total_annotations": 0,
        "cancelled_annotations": 0,
        "total_predictions": 1,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "project": project_id,
        "comment_authors": [],
    }


def convert(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    min_count: float = 1.5,
    project_id: int = 4,
    indent: int | None = None,
) -> list[dict]:
    """Read a `regions.json` manifest, build LS tasks, write the output JSON.

    `min_count` drops the small regions. In blob mode most regions are single
    birds reading ~1.0 and boxing every one of them hides the image under a
    hundred rectangles; the default keeps only regions where knowing the count
    actually helps. Pass 0 to keep everything (what you want for `--grid`
    manifests, where every tile matters).
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with src_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    if "images" not in manifest:
        raise SystemExit(f"{src_path} is not a density-regions manifest (no 'images' key).")

    now = ls_timestamp()
    tasks: list[dict] = []
    kept = dropped = 0

    for task_id, record in enumerate(manifest["images"], start=1):
        W, H = record["width"], record["height"]
        results = []
        for region in record["regions"]:
            if region["count"] < min_count:
                dropped += 1
                continue
            results.append(_build_region_result(region, W, H))
        kept += len(results)

        tasks.append(
            _build_task(
                task_id=task_id,
                data_img=local_files_url(record["file_name"], image_prefix),
                results=results,
                now=now,
                project_id=project_id,
            )
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=indent)

    params = manifest.get("params", {})
    print(f"[ok] source mode: {params.get('mode', 'blobs')}, grid={params.get('grid')}, merge={params.get('merge')}")
    print(f"[ok] wrote {len(tasks)} tasks, {kept} region boxes -> {dst_path}")
    if dropped:
        print(f"[ok] skipped {dropped} region(s) under --min-count {min_count:g}")
    print("[!] the project's labeling config must include both controls; see --print-config")
    return tasks


def import_to_label_studio(
    tasks: list[dict],
    *,
    project_id: int,
    keypoint_label: str = "chicken",
    url: str | None = None,
    update_config: bool = True,
) -> object:
    """Import generated tasks into a live Label Studio project.

    The token deliberately comes only from ``LABEL_STUDIO_API_KEY``. Keeping it
    out of argparse means it never appears in the WebUI form, command preview,
    process list, or run log.
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LABEL_STUDIO_API_KEY is not set in .env")
    target = (url or os.getenv("LABEL_STUDIO_URL") or f"http://localhost:{os.getenv('LS_PORT', '8080')}").strip()
    if not target.startswith(("http://", "https://")):
        target = "http://" + target
    target = target.rstrip("/")

    try:
        from label_studio_sdk import LabelStudio
    except ImportError as exc:
        raise RuntimeError("label-studio-sdk is required to import tasks") from exc

    client = LabelStudio(base_url=target, api_key=api_key)
    # Fetch first for a useful error when the id or credentials are wrong.
    project = client.projects.get(id=project_id)
    if update_config:
        client.projects.update(
            id=project_id,
            label_config=labeling_config(keypoint_label),
            show_collab_predictions=True,
        )
        print(f"[ok] updated labeling interface for project {project_id}")

    response = client.projects.import_tasks(
        id=project_id,
        request=tasks,
        commit_to_project=True,
        return_task_ids=True,
    )
    title = getattr(project, "title", "")
    print(f"[ok] imported {len(tasks)} task(s) into Label Studio project {project_id}{f' ({title})' if title else ''}")
    print(f"[ok] open Label Studio: {target}/projects/{project_id}/data")
    return response


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert density regions into a Label Studio prediction layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--print-config",
        action="store_true",
        help="print the labeling-interface XML these predictions need, then exit",
    )
    ap.add_argument("--keypoint-label", default="chicken", help="keypoint label name used by --print-config")
    ap.add_argument("--src", help="regions.json written by webui/ops/density_regions.py")
    ap.add_argument("--dst", help="output Label Studio JSON")
    ap.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help="path (relative to LS's LOCAL_FILES_DOCUMENT_ROOT) prepended to file_name in data.img",
    )
    ap.add_argument(
        "--min-count",
        type=float,
        default=1.5,
        help="skip regions predicting fewer chickens than this; 0 keeps every region",
    )
    ap.add_argument("--project-id", type=int, default=4)
    ap.add_argument(
        "--send-to-label-studio",
        action="store_true",
        help="import the generated tasks into the selected live Label Studio project",
    )
    ap.add_argument(
        "--label-studio-url",
        default=os.getenv("LABEL_STUDIO_URL") or f"http://localhost:{os.getenv('LS_PORT', '8080')}",
        help="Label Studio base URL; the API key is read only from .env",
    )
    ap.add_argument(
        "--keep-label-config",
        action="store_true",
        help="do not install the required keypoint + region-box labeling interface before import",
    )
    ap.add_argument(
        "--indent",
        type=int,
        default=None,
        help="JSON output indent (omit for compact single-line; 2 for human-readable)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(labeling_config(args.keypoint_label), end="")
        return
    if not args.src or not args.dst:
        raise SystemExit("--src and --dst are required (or pass --print-config).")
    tasks = convert(
        src_path=args.src,
        dst_path=args.dst,
        image_prefix=args.image_prefix,
        min_count=args.min_count,
        project_id=args.project_id,
        indent=args.indent,
    )
    if args.send_to_label_studio:
        try:
            import_to_label_studio(
                tasks,
                project_id=args.project_id,
                keypoint_label=args.keypoint_label,
                url=args.label_studio_url,
                update_config=not args.keep_label_config,
            )
        except Exception as exc:
            raise SystemExit(f"Label Studio import failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
