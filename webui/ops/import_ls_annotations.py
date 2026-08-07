"""Import annotations from an external Label Studio JSON export.

The source export may refer to uploads or local-file paths from another
machine. Those paths are never reused. Each task is matched to an image in
``--images-dir`` by file name (including Label Studio's optional eight-hex
upload-prefix normalization), then rewritten to this server's local-files URL.

External task and annotation IDs are deliberately discarded. Annotation
``result`` payloads are imported as completed annotations into the selected
project. Keypoint control names and labels are remapped to the destination
project's labeling configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from _common import connect, get_field, ls_img_to_filename, target_parser, task_label


ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = ROOT / "tools" / "annotations"
sys.path.insert(0, str(ANNOTATIONS_DIR))

from convert_ls_to_coco import image_key  # noqa: E402
from to_label_studio import DEFAULT_IMAGE_PREFIX, local_files_url  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_LOCAL_ROOT = Path(os.getenv("LS_LOCAL_FILES_ROOT") or ROOT.parent / "data")
DEFAULT_IMAGES_DIR = DEFAULT_LOCAL_ROOT / Path(DEFAULT_IMAGE_PREFIX.replace("\\", "/"))
SOURCE_IMAGE_KEYS = ("img", "image", "url")
DEFAULT_FUZZY_THRESHOLD = 0.82
DEFAULT_FUZZY_MARGIN = 0.03
DEFAULT_KEYPOINT_LABEL = "chicken"


def load_export(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        payload = payload["tasks"]
    if not isinstance(payload, list):
        raise ValueError("expected a Label Studio task list (or an object containing a 'tasks' list)")
    if not all(isinstance(task, dict) for task in payload):
        raise ValueError("every item in the Label Studio export must be a task object")
    return payload


def _source_image(task: dict) -> tuple[str, str]:
    data = task.get("data")
    if not isinstance(data, dict):
        return "", ""
    for key in SOURCE_IMAGE_KEYS:
        value = data.get(key)
        if value:
            return key, str(value)
    return "", ""


def index_images(
    images_dir: str | Path, recursive: bool = False
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    root = Path(images_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"images directory not found: {root}")
    candidates = root.rglob("*") if recursive else root.iterdir()
    images = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"no supported images found in: {root}")

    exact: dict[str, list[Path]] = {}
    canonical: dict[str, list[Path]] = {}
    for path in images:
        exact.setdefault(path.name.casefold(), []).append(path)
        canonical.setdefault(image_key(path.name), []).append(path)
    return exact, canonical


def _fuzzy_key(file_name: str) -> str:
    """Comparable stem, ignoring separators, case, extension and small ordinal prefixes."""
    stem = Path(image_key(file_name)).stem
    normalized = unicodedata.normalize("NFKC", stem).casefold()
    tokens = [
        "".join(char for char in token if char.isalnum())
        for token in normalized.replace("-", " ").replace("_", " ").split()
    ]
    tokens = [token for token in tokens if token]
    if len(tokens) > 1 and tokens[0].isdigit() and len(tokens[0]) <= 4:
        tokens = tokens[1:]
    return " ".join(tokens)


def _name_similarity(source_name: str, candidate_name: str) -> float:
    source = _fuzzy_key(source_name)
    candidate = _fuzzy_key(candidate_name)
    if not source or not candidate:
        return 0.0
    compact = SequenceMatcher(None, source.replace(" ", ""), candidate.replace(" ", "")).ratio()
    token_order_free = SequenceMatcher(
        None, " ".join(sorted(source.split())), " ".join(sorted(candidate.split()))
    ).ratio()
    return max(compact, token_order_free)


def match_image(
    source_name: str,
    exact: dict[str, list[Path]],
    canonical: dict[str, list[Path]],
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    fuzzy_margin: float = DEFAULT_FUZZY_MARGIN,
) -> tuple[Path | None, str]:
    direct = exact.get(source_name.casefold(), [])
    if len(direct) == 1:
        return direct[0], ""
    if len(direct) > 1:
        return None, "ambiguous exact filename"

    normalized = canonical.get(image_key(source_name), [])
    if len(normalized) == 1:
        return normalized[0], ""
    if len(normalized) > 1:
        return None, "ambiguous filename after removing the LS upload hash"

    if fuzzy_threshold <= 0:
        return None, "image not found (fuzzy matching disabled)"
    candidates = sorted(
        {path for paths in canonical.values() for path in paths}, key=lambda path: str(path).casefold()
    )
    ranked = sorted(
        ((_name_similarity(source_name, path.name), path) for path in candidates),
        key=lambda item: (-item[0], str(item[1]).casefold()),
    )
    best_score, best = ranked[0]
    runner_up_score, runner_up = ranked[1] if len(ranked) > 1 else (0.0, None)
    if best_score < fuzzy_threshold:
        return None, f"image not found; closest is {best.name!r} (score {best_score:.3f})"
    if runner_up is not None and best_score - runner_up_score < fuzzy_margin:
        return None, (
            f"ambiguous fuzzy match: {best.name!r} ({best_score:.3f}) vs {runner_up.name!r} ({runner_up_score:.3f})"
        )
    return best, f"fuzzy match score {best_score:.3f}"


def _normalize_keypoint_results(
    results: list,
    keypoint_label: str,
    *,
    from_name: str | None = None,
    to_name: str | None = None,
) -> list:
    normalized = copy.deepcopy(results)
    for item in normalized:
        if not isinstance(item, dict) or item.get("type") != "keypointlabels":
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            value = {}
            item["value"] = value
        value["keypointlabels"] = [keypoint_label]
        if from_name:
            item["from_name"] = from_name
        if to_name:
            item["to_name"] = to_name
    return normalized


def _destination_keypoint_config(label_config: str, preferred_label: str) -> tuple[str, str, str] | None:
    """Return the target project's single compatible keypoint control."""
    if not label_config.strip():
        return None
    try:
        root = ET.fromstring(label_config)
    except ET.ParseError as exc:
        raise ValueError(f"destination project has invalid label config XML: {exc}") from exc

    controls = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "KeyPointLabels"]
    compatible = []
    for control in controls:
        labels = [
            child.attrib.get("value", "")
            for child in control
            if child.tag.rsplit("}", 1)[-1] == "Label" and child.attrib.get("value")
        ]
        if preferred_label in labels:
            compatible.append((control, preferred_label))
        elif len(labels) == 1:
            compatible.append((control, labels[0]))
    if len(compatible) != 1:
        detail = "none" if not compatible else "multiple"
        raise ValueError(
            f"destination project has {detail} compatible KeyPointLabels controls; "
            "use a project with one keypoint control containing --keypoint-label"
        )
    control, label = compatible[0]
    from_name = control.attrib.get("name", "").strip()
    to_name = control.attrib.get("toName", "").strip()
    if not from_name or not to_name:
        raise ValueError("destination KeyPointLabels control is missing name or toName")
    return from_name, to_name, label


def clean_annotation(annotation: Any, keypoint_label: str = DEFAULT_KEYPOINT_LABEL) -> dict:
    """Keep portable content, normalize point labels, and drop foreign database IDs."""
    if not isinstance(annotation, dict):
        return {"result": []}
    cleaned = {
        "result": _normalize_keypoint_results(annotation.get("result") or [], keypoint_label),
        "was_cancelled": bool(annotation.get("was_cancelled", False)),
        "ground_truth": bool(annotation.get("ground_truth", False)),
    }
    if annotation.get("lead_time") is not None:
        cleaned["lead_time"] = annotation["lead_time"]
    return cleaned


def prepare_tasks(
    source_tasks: list[dict],
    *,
    images_dir: str | Path,
    image_prefix: str,
    recursive: bool = False,
    allow_missing: bool = False,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    fuzzy_margin: float = DEFAULT_FUZZY_MARGIN,
    keypoint_label: str = DEFAULT_KEYPOINT_LABEL,
) -> tuple[list[dict], list[dict]]:
    """Rewrite image paths and return (importable tasks, rejected rows)."""
    images_root = Path(images_dir).expanduser().resolve()
    exact, canonical = index_images(images_dir, recursive)
    prepared: list[dict] = []
    rejected: list[dict] = []

    for index, source in enumerate(source_tasks, start=1):
        source_key, source_value = _source_image(source)
        source_name = ls_img_to_filename(source_value) if source_value else ""
        matched, reason = (
            match_image(
                source_name,
                exact,
                canonical,
                fuzzy_threshold=fuzzy_threshold,
                fuzzy_margin=fuzzy_margin,
            )
            if source_name
            else (None, "no image field")
        )
        if matched is None:
            rejected.append({"index": index, "image": source_name or source_value or "<missing>", "reason": reason})
            continue
        if reason:
            print(f"[fuzzy] {source_name} -> {matched.name} ({reason})")

        data = copy.deepcopy(source.get("data")) if isinstance(source.get("data"), dict) else {}
        if source_key and source_key != "img":
            data.pop(source_key, None)
        separator = "\\" if "\\" in image_prefix else "/"
        relative_name = separator.join(matched.relative_to(images_root).parts)
        data["img"] = local_files_url(relative_name, image_prefix)
        source_annotations = source.get("annotations")
        if not isinstance(source_annotations, list):
            source_annotations = []
        annotations = [clean_annotation(item, keypoint_label) for item in source_annotations]
        prepared.append({"data": data, "annotations": annotations})

    if rejected and not allow_missing:
        preview = "\n".join(f"  #{row['index']} {row['image']}: {row['reason']}" for row in rejected[:12])
        more = f"\n  ... and {len(rejected) - 12} more" if len(rejected) > 12 else ""
        raise ValueError(
            f"{len(rejected)} task(s) could not be matched; nothing was imported. "
            f"Fix --images-dir / --recursive, or tick --allow-missing to skip them:\n{preview}{more}"
        )
    return prepared, rejected


def _existing_task_index(tasks: list[Any]) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    exact: dict[str, list[Any]] = {}
    canonical: dict[str, list[Any]] = {}
    for task in tasks:
        name = task_label(task)
        exact.setdefault(name.casefold(), []).append(task)
        canonical.setdefault(image_key(name), []).append(task)
    return exact, canonical


def _choose_existing(candidates: list[Any]) -> Any:
    """Use the oldest task if a previous bad import already created duplicates."""
    return min(candidates, key=lambda task: int(get_field(task, "id", 0) or 0))


def _create_annotation(client, task_id: int, project_id: int, annotation: dict) -> object:
    kwargs = {
        "result": annotation.get("result") or [],
        "was_cancelled": bool(annotation.get("was_cancelled", False)),
        "ground_truth": bool(annotation.get("ground_truth", False)),
        "project": project_id,
        "task": task_id,
    }
    if annotation.get("lead_time") is not None:
        kwargs["lead_time"] = annotation["lead_time"]
    return client.annotations.create(id=task_id, **kwargs)


def apply_to_project(
    client,
    project_id: int,
    tasks: list[dict],
    *,
    existing_only: bool = False,
    keypoint_label: str = DEFAULT_KEYPOINT_LABEL,
) -> dict:
    """Add annotations to matching tasks; import new tasks only when absent."""
    project = client.projects.get(id=project_id)
    target_config = _destination_keypoint_config(str(get_field(project, "label_config", "") or ""), keypoint_label)
    if target_config:
        from_name, to_name, label = target_config
        tasks = copy.deepcopy(tasks)
        for source in tasks:
            for annotation in source.get("annotations") or []:
                annotation["result"] = _normalize_keypoint_results(
                    annotation.get("result") or [],
                    label,
                    from_name=from_name,
                    to_name=to_name,
                )
        print(f"[ok] mapped keypoints to destination control {from_name!r} -> {to_name!r}, label {label!r}")
    existing_tasks = list(client.tasks.list(project=project_id, fields="all"))
    exact, canonical = _existing_task_index(existing_tasks)
    missing = []
    matched = annotations_added = duplicate_task_matches = empty_annotations = 0

    for source in tasks:
        name = ls_img_to_filename(str(source.get("data", {}).get("img", "")))
        candidates = exact.get(name.casefold(), []) or canonical.get(image_key(name), [])
        if not candidates:
            missing.append(source)
            continue
        matched += 1
        if len(candidates) > 1:
            duplicate_task_matches += 1
            ids = sorted(int(get_field(task, "id", 0) or 0) for task in candidates)
            print(f"[warn] {name}: existing duplicate tasks {ids}; using oldest task #{ids[0]}")
        target = _choose_existing(candidates)
        task_id = int(get_field(target, "id"))
        annotations = source.get("annotations") or []
        if not annotations:
            empty_annotations += 1
            print(f"[ok] {name}: matched existing task #{task_id}, but source has no annotation")
            continue
        for annotation in annotations:
            _create_annotation(client, task_id, project_id, annotation)
            annotations_added += 1
        print(f"[ok] {name}: added {len(annotations)} annotation(s) to existing task #{task_id}")

    created = skipped_missing = 0
    response = None
    if missing and existing_only:
        skipped_missing = len(missing)
        print(f"[warn] skipped {skipped_missing} image(s) with no existing task (--existing-only)")
    elif missing:
        response = client.projects.import_tasks(
            id=project_id,
            request=missing,
            commit_to_project=True,
            return_task_ids=True,
        )
        created = len(missing)
        print(f"[ok] created {created} new task(s) because no existing image task matched")

    summary = {
        "matched_tasks": matched,
        "annotations_added": annotations_added,
        "tasks_created": created,
        "missing_skipped": skipped_missing,
        "duplicate_task_matches": duplicate_task_matches,
        "empty_annotations": empty_annotations,
        "import_response": response,
    }
    print(f"[ok] project {project_id} ({get_field(project, 'title', '')}): {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an external LS JSON export, remapping its images to this server's local files",
        parents=[target_parser()],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("source and image mapping")
    group.add_argument("--src", required=True, help="external Label Studio JSON export (choose/upload it in WebUI)")
    group.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="folder containing the corresponding images on this machine",
    )
    group.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help="path from Label Studio's LOCAL_FILES_DOCUMENT_ROOT to --images-dir",
    )
    group.add_argument("--recursive", action="store_true", help="search for matching images below --images-dir")
    group.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=DEFAULT_FUZZY_THRESHOLD,
        help="minimum 0..1 filename similarity used only after exact matching fails; 0 disables fuzzy matching",
    )
    group.add_argument(
        "--fuzzy-margin",
        type=float,
        default=DEFAULT_FUZZY_MARGIN,
        help="best fuzzy score must beat the second-best by at least this amount",
    )
    group.add_argument(
        "--keypoint-label",
        default=DEFAULT_KEYPOINT_LABEL,
        help="force every imported keypoint result to this destination-project label (case-sensitive)",
    )
    group.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip unmatched/ambiguous images; by default any mismatch cancels the entire import",
    )
    group.add_argument(
        "--existing-only",
        action="store_true",
        help="only add annotations to matching existing tasks; never create a new task",
    )
    group.add_argument("--check-only", action="store_true", help="validate and report the mapping without importing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.fuzzy_threshold <= 1:
        raise SystemExit("--fuzzy-threshold must be between 0 and 1")
    if not 0 <= args.fuzzy_margin <= 1:
        raise SystemExit("--fuzzy-margin must be between 0 and 1")
    if not args.keypoint_label.strip():
        raise SystemExit("--keypoint-label cannot be empty")
    try:
        source_tasks = load_export(args.src)
        tasks, rejected = prepare_tasks(
            source_tasks,
            images_dir=args.images_dir,
            image_prefix=args.image_prefix,
            recursive=args.recursive,
            allow_missing=args.allow_missing,
            fuzzy_threshold=args.fuzzy_threshold,
            fuzzy_margin=args.fuzzy_margin,
            keypoint_label=args.keypoint_label.strip(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Import preparation failed: {exc}") from exc

    annotations = sum(len(task["annotations"]) for task in tasks)
    results = sum(len(annotation.get("result", [])) for task in tasks for annotation in task["annotations"])
    print(f"Source: {args.src}")
    print(f"Matched: {len(tasks)}/{len(source_tasks)} task(s), {annotations} annotation(s), {results} result(s)")
    for task in tasks[:10]:
        print(f"  -> {task['data']['img']}")
    if len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more")
    if rejected:
        print(f"Skipped: {len(rejected)} unmatched or ambiguous task(s)")
        for row in rejected:
            print(f"  #{row['index']} {row['image']}: {row['reason']}")
    if not tasks:
        raise SystemExit("No matched tasks to import.")
    if args.check_only:
        print("[ok] check only; nothing imported")
        return

    client = connect(args)
    try:
        apply_to_project(
            client,
            args.project_id,
            tasks,
            existing_only=args.existing_only,
            keypoint_label=args.keypoint_label.strip(),
        )
    except Exception as exc:
        raise SystemExit(f"Label Studio import failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    main()
