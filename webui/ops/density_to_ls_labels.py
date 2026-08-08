"""Add density-region centroids to existing Label Studio tasks as predictions.

The source is ``regions.json`` written by ``density_regions.py``. Each retained
region becomes one keypoint in a prediction layer, labeled ``chicken-pred`` by
default. Human points keep their existing ``chicken`` label and annotations.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from _common import connect, get_field, target_parser
from convert_ls_to_coco import image_key
from import_ls_annotations import _choose_existing, _existing_task_index


DEFAULT_LABEL = "chicken-pred"
DEFAULT_MANUAL_LABEL = "chicken"
PREDICTION_COLOR = "#2563eb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert density regions to chicken-pred keypoints on existing Label Studio image tasks",
        parents=[target_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("density labels")
    group.add_argument("--src", required=True, help="regions.json written by Density regions")
    group.add_argument(
        "--min-count",
        type=float,
        default=0.5,
        help="skip density regions whose integrated predicted count is below this value",
    )
    group.add_argument("--label", default=DEFAULT_LABEL, help="Label Studio label for every generated point")
    group.add_argument(
        "--manual-label",
        default=DEFAULT_MANUAL_LABEL,
        help="existing human keypoint label used to identify the destination KeyPointLabels control",
    )
    group.add_argument(
        "--check-only",
        action="store_true",
        help="report task matches and generated point counts without changing Label Studio",
    )
    return parser


def load_regions(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    images = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(images, list):
        raise ValueError("expected regions.json with an 'images' list")
    return images


def ensure_prediction_label(
    label_config: str,
    *,
    prediction_label: str = DEFAULT_LABEL,
    manual_label: str = DEFAULT_MANUAL_LABEL,
) -> tuple[str, str, str, bool]:
    """Return updated XML, keypoint control name, image control name, changed."""
    try:
        root = ET.fromstring(label_config)
    except ET.ParseError as exc:
        raise ValueError(f"project label config is invalid XML: {exc}") from exc

    controls = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "KeyPointLabels"]
    with_prediction = [
        control for control in controls if any(child.attrib.get("value") == prediction_label for child in control)
    ]
    with_manual = [
        control for control in controls if any(child.attrib.get("value") == manual_label for child in control)
    ]
    candidates = with_prediction or with_manual or (controls if len(controls) == 1 else [])
    if len(candidates) != 1:
        raise ValueError(
            f"could not identify one KeyPointLabels control (looked for {manual_label!r}); "
            "the project must have one compatible keypoint control"
        )
    control = candidates[0]
    from_name = control.attrib.get("name", "").strip()
    to_name = control.attrib.get("toName", "").strip()
    if not from_name or not to_name:
        raise ValueError("destination KeyPointLabels control is missing name or toName")

    changed = not with_prediction
    if changed:
        ET.SubElement(control, "Label", value=prediction_label, background=PREDICTION_COLOR)
    return ET.tostring(root, encoding="unicode"), from_name, to_name, changed


def build_results(
    record: dict,
    *,
    from_name: str,
    to_name: str,
    label: str = DEFAULT_LABEL,
    min_count: float = 0.5,
) -> list[dict]:
    width = float(record["width"])
    height = float(record["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions for {record.get('file_name', '<unknown>')}")
    results = []
    for index, region in enumerate(record.get("regions") or [], start=1):
        count = float(region.get("count", 0.0))
        if count < min_count:
            continue
        centroid = region.get("centroid")
        if not isinstance(centroid, list) or len(centroid) != 2:
            x, y, w, h = region["bbox"]
            centroid = [x + w / 2.0, y + h / 2.0]
        x, y = map(float, centroid)
        results.append(
            {
                "original_width": int(width),
                "original_height": int(height),
                "image_rotation": 0,
                "value": {
                    "x": min(100.0, max(0.0, x / width * 100.0)),
                    "y": min(100.0, max(0.0, y / height * 100.0)),
                    "width": 0.5,
                    "keypointlabels": [label],
                },
                "id": f"density-{region.get('id', index)}",
                "from_name": from_name,
                "to_name": to_name,
                "type": "keypointlabels",
                "origin": "prediction",
                "meta": {"text": [f"density count ~{count:.2f}"]},
            }
        )
    return results


def _prediction_summary(prediction: Any) -> tuple[int, str]:
    return int(get_field(prediction, "id", 0) or 0), str(get_field(prediction, "model_version", "") or "")


def apply_density_labels(
    client,
    project_id: int,
    records: list[dict],
    *,
    label: str = DEFAULT_LABEL,
    manual_label: str = DEFAULT_MANUAL_LABEL,
    min_count: float = 0.5,
    check_only: bool = False,
) -> dict:
    project = client.projects.get(id=project_id)
    config, from_name, to_name, config_changed = ensure_prediction_label(
        str(get_field(project, "label_config", "") or ""),
        prediction_label=label,
        manual_label=manual_label,
    )
    tasks = list(client.tasks.list(project=project_id, fields="all"))
    exact, canonical = _existing_task_index(tasks)
    matched = missing = created = updated = points = duplicate_tasks = duplicate_sources = 0
    seen_task_ids: set[int] = set()
    model_version = f"density-keypoints:{label}"

    # Label Studio validates result labels when a prediction is created, so the
    # config must contain chicken-pred before the first prediction API call.
    if config_changed and not check_only:
        client.projects.update(id=project_id, label_config=config, show_collab_predictions=True)
        print(f"[ok] added label {label!r} to keypoint control {from_name!r}")

    for record in records:
        name = str(record.get("file_name", ""))
        candidates = exact.get(name.casefold(), []) or canonical.get(image_key(name), [])
        if not candidates:
            missing += 1
            print(f"[warn] {name}: no existing Label Studio task; skipped")
            continue
        if len(candidates) > 1:
            duplicate_tasks += 1
        task = _choose_existing(candidates)
        task_id = int(get_field(task, "id"))
        if task_id in seen_task_ids:
            duplicate_sources += 1
            print(f"[warn] {name}: another regions.json row already maps to task #{task_id}; skipped")
            continue
        seen_task_ids.add(task_id)
        matched += 1
        results = build_results(
            record,
            from_name=from_name,
            to_name=to_name,
            label=label,
            min_count=min_count,
        )
        points += len(results)
        prior = [p for p in (get_field(task, "predictions", []) or []) if _prediction_summary(p)[1] == model_version]
        if check_only:
            print(f"[check] {name}: task #{task_id}, {len(results)} {label} point(s)")
        elif prior:
            prediction_id = min(_prediction_summary(p)[0] for p in prior)
            client.predictions.update(
                id=prediction_id,
                task=task_id,
                result=copy.deepcopy(results),
                model_version=model_version,
            )
            updated += 1
            print(f"[ok] {name}: updated density prediction #{prediction_id} on task #{task_id}")
        else:
            client.predictions.create(task=task_id, result=copy.deepcopy(results), model_version=model_version)
            created += 1
            print(f"[ok] {name}: added {len(results)} {label} point(s) to task #{task_id}")

    summary = {
        "matched_tasks": matched,
        "missing_tasks": missing,
        "points": points,
        "predictions_created": created,
        "predictions_updated": updated,
        "duplicate_task_matches": duplicate_tasks,
        "duplicate_source_matches": duplicate_sources,
        "config_needed": config_changed,
        "config_changed": config_changed and not check_only,
    }
    print(f"[ok] project {project_id} ({get_field(project, 'title', '')}): {summary}")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    if args.min_count < 0:
        raise SystemExit("--min-count must be >= 0")
    if not args.label.strip() or not args.manual_label.strip():
        raise SystemExit("--label and --manual-label cannot be empty")
    try:
        records = load_regions(args.src)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read density regions: {exc}") from exc
    client = connect(args)
    try:
        apply_density_labels(
            client,
            args.project_id,
            records,
            label=args.label.strip(),
            manual_label=args.manual_label.strip(),
            min_count=args.min_count,
            check_only=args.check_only,
        )
    except Exception as exc:
        raise SystemExit(f"Density-to-label import failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
