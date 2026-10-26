"""Operations on a live Label Studio project.

Anything that deletes or overwrites work in Label Studio — which has no undo —
reports what it would do unless `apply` is set.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.training.labeling.ls_format import (
    add_label,
    config_xml,
    file_name,
    image_key,
    keypoint,
    keypoint_control,
    local_files_url,
    relabel,
    task_file,
    task_image,
)


if TYPE_CHECKING:
    from birdcount.training.labeling.client import LabelStudio

Log = Callable[[str], None]


# -- one annotation per task --------------------------------------------------------------

KEEP_STRATEGIES = ("most-points", "newest", "oldest")


def choose_keeper(annotations: list[dict], strategy: str) -> dict:
    """The annotation to keep. A real one always beats a cancelled (skipped) one.

    most-points keeps the most complete labelling — the default, because a stray
    empty re-submit is usually the newest.
    """
    pool = [a for a in annotations if not a.get("was_cancelled")] or annotations

    def stamp(a: dict) -> tuple:
        return (str(a.get("updated_at") or ""), a.get("id") or 0)

    if strategy == "most-points":
        return max(pool, key=lambda a: (len(a.get("result") or []), *stamp(a)))
    if strategy == "newest":
        return max(pool, key=stamp)
    if strategy == "oldest":
        return min(pool, key=stamp)
    raise ValueError(f"unknown strategy {strategy!r}")


def dedupe_annotations(ls: LabelStudio, project_id: int, *, keep: str, apply: bool, log: Log = print) -> dict:
    """Keep one annotation per task, delete the rest (only with `apply`).

    LS adds an annotation each time a task is submitted again, the exports read
    only the first, and the UI's counts do not show the extras.
    """
    tasks = ls.tasks(project_id)
    counts = {"tasks": len(tasks), "single": 0, "none": 0, "duplicated": 0, "extra": 0, "deleted": 0, "failed": 0}
    for task in tasks:
        annotations = task.get("annotations") or []
        if len(annotations) < 2:
            counts["single" if annotations else "none"] += 1
            continue
        keeper = choose_keeper(annotations, keep)
        drops = [a for a in annotations if a is not keeper]
        counts["duplicated"] += 1
        counts["extra"] += len(drops)
        listed = ", ".join(
            f"#{a['id']}({len(a.get('result') or [])}p{', cancelled' if a.get('was_cancelled') else ''})"
            for a in annotations
        )
        log(f"  {task_file(task) or 'task #' + str(task['id'])}: {listed} -> keep #{keeper['id']}")
        for a in drops if apply else ():
            try:
                ls.delete_annotation(a["id"])
                counts["deleted"] += 1
            except Exception as e:  # one bad id must not stop the pass
                counts["failed"] += 1
                log(f"    ! could not delete #{a['id']}: {e}")
    return counts


# -- importing an external export ------------------------------------------------------------

FUZZY_THRESHOLD = 0.82
FUZZY_MARGIN = 0.03


class ImageIndex:
    """The images of a folder, findable by the names another machine gave them."""

    def __init__(self, root: Path, *, recursive: bool = False):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"image folder not found: {self.root}")
        found = self.root.rglob("*") if recursive else self.root.iterdir()
        self.paths = sorted(p for p in found if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"no images in {self.root}")
        self.exact: dict[str, list[Path]] = {}
        self.by_key: dict[str, list[Path]] = {}
        for p in self.paths:
            self.exact.setdefault(p.name.casefold(), []).append(p)
            self.by_key.setdefault(image_key(p.name), []).append(p)

    def match(
        self, name: str, *, threshold: float = FUZZY_THRESHOLD, margin: float = FUZZY_MARGIN
    ) -> tuple[Path | None, str]:
        """(file, note) — exact name, then without the LS upload hash, then a
        conservative closest-name match; (None, reason) when unsure."""
        for candidates, what in (
            (self.exact.get(name.casefold(), []), "exact"),
            (self.by_key.get(image_key(name), []), "hash-free"),
        ):
            if len(candidates) == 1:
                return candidates[0], ""
            if candidates:
                return None, f"ambiguous {what} name match"
        if threshold <= 0:
            return None, "not found (fuzzy matching off)"
        ranked = sorted(
            ((_similarity(name, p.name), p) for p in self.paths), key=lambda t: (-t[0], str(t[1]).casefold())
        )
        (best, path), runner_up = ranked[0], ranked[1][0] if len(ranked) > 1 else 0.0
        if best < threshold:
            return None, f"not found; closest {path.name!r} scores {best:.3f}"
        if best - runner_up < margin:
            return None, f"ambiguous: {path.name!r} {best:.3f} vs runner-up {runner_up:.3f}"
        return path, f"fuzzy match {best:.3f}"


def _fuzzy_key(name: str) -> str:
    """Comparable stem: no upload hash, extension, case, separators or short ordinal prefix."""
    stem = Path(image_key(name)).stem
    tokens = unicodedata.normalize("NFKC", stem).casefold().replace("-", " ").replace("_", " ").split()
    tokens = [t for t in ("".join(c for c in tok if c.isalnum()) for tok in tokens) if t]
    if len(tokens) > 1 and tokens[0].isdigit() and len(tokens[0]) <= 4:
        tokens = tokens[1:]
    return " ".join(tokens)


def _similarity(a: str, b: str) -> float:
    a, b = _fuzzy_key(a), _fuzzy_key(b)
    if not a or not b:
        return 0.0
    compact = SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()
    unordered = SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
    return max(compact, unordered)


def load_export(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        payload = payload["tasks"]
    if not isinstance(payload, list) or not all(isinstance(t, dict) for t in payload):
        raise ValueError(f"{path.name}: expected a Label Studio task list")
    return payload


def prepare_import(
    source: list[dict],
    index: ImageIndex,
    prefix: str,
    *,
    label: str,
    threshold: float = FUZZY_THRESHOLD,
    margin: float = FUZZY_MARGIN,
    log: Log = print,
) -> tuple[list[dict], list[tuple[int, str, str]]]:
    """Tasks pointing at this machine's images, and (index, image, reason) for the rest.

    The other machine's paths, task ids and annotation ids are dropped; only
    the annotation content travels, with every keypoint relabelled `label`.
    """
    prepared, rejected = [], []
    separator = "\\" if "\\" in prefix else "/"
    for i, task in enumerate(source, start=1):
        ref = task_image(task)
        name = file_name(ref) if ref else ""
        path, note = index.match(name, threshold=threshold, margin=margin) if name else (None, "no image")
        if path is None:
            rejected.append((i, name or "<none>", note))
            continue
        if note:
            log(f"  [fuzzy] {name} -> {path.name} ({note})")
        data = {k: v for k, v in (task.get("data") or {}).items() if v != ref}
        data["img"] = local_files_url(separator.join(path.relative_to(index.root).parts), prefix)
        annotations = []
        for ann in task.get("annotations") or []:
            clean = {
                "result": relabel(ann.get("result") or [], label),
                "was_cancelled": bool(ann.get("was_cancelled")),
                "ground_truth": bool(ann.get("ground_truth")),
            }
            if ann.get("lead_time") is not None:
                clean["lead_time"] = ann["lead_time"]
            annotations.append(clean)
        prepared.append({"data": data, "annotations": annotations})
    return prepared, rejected


def _task_index(tasks: list[dict], name_of: Callable[[str], str] = lambda n: n) -> tuple[dict, dict]:
    exact: dict[str, list[dict]] = {}
    by_key: dict[str, list[dict]] = {}
    for task in tasks:
        name = name_of(task_file(task))
        exact.setdefault(name.casefold(), []).append(task)
        by_key.setdefault(image_key(name), []).append(task)
    return exact, by_key


def _find_task(name: str, exact: dict, by_key: dict) -> list[dict]:
    return exact.get(name.casefold()) or by_key.get(image_key(name)) or []


def _oldest(tasks: list[dict]) -> dict:
    """With duplicate tasks for one image (an earlier bad import), use the first."""
    return min(tasks, key=lambda t: t["id"])


def import_annotations(
    ls: LabelStudio, project_id: int, tasks: list[dict], *, label: str, existing_only: bool = False, log: Log = print
) -> dict:
    """Add each task's annotations to the project's task for the same image;
    import the tasks no existing one matches (unless `existing_only`)."""
    project = ls.project(project_id)
    if (project.get("label_config") or "").strip():
        _, control, labels = keypoint_control(project["label_config"], label)
        target = label if label in labels else labels[0]
        names = {"from_name": control.attrib["name"], "to_name": control.attrib["toName"]}
        for task in tasks:
            for ann in task["annotations"]:
                ann["result"] = relabel(ann["result"], target, **names)
        log(f"  keypoints go to control {names['from_name']!r} as {target!r}")

    exact, by_key = _task_index(ls.tasks(project_id))
    counts = {"matched": 0, "annotations_added": 0, "tasks_created": 0, "skipped": 0, "duplicate_tasks": 0}
    missing = []
    for task in tasks:
        name = task_file(task)
        candidates = _find_task(name, exact, by_key)
        if not candidates:
            missing.append(task)
            continue
        counts["matched"] += 1
        if len(candidates) > 1:
            counts["duplicate_tasks"] += 1
            log(f"  [warn] {name}: {len(candidates)} tasks for this image; using the oldest")
        target = _oldest(candidates)
        for ann in task["annotations"]:
            ls.create_annotation(target["id"], {**ann, "project": project_id})
            counts["annotations_added"] += 1
    if missing and existing_only:
        counts["skipped"] = len(missing)
    elif missing:
        ls.import_tasks(project_id, missing)
        counts["tasks_created"] = len(missing)
    return counts


# -- density pre-labels ----------------------------------------------------------------------

PREDICTION_LABEL = "chicken-pred"
PREDICTION_COLOR = "#2563eb"


@dataclass(frozen=True)
class Prelabel:
    name: str  # image stem
    result: list[dict]


def prelabels(blobs: dict, *, from_name: str, to_name: str, label: str, min_count: float) -> list[Prelabel]:
    """One keypoint per warm density blob of each image in an eval `blobs.json`."""
    frame_w, frame_h = blobs["params"]["frame_size"]
    out = []
    for image in blobs["images"]:
        size = image["size"]
        sx, sy = size[0] / frame_w, size[1] / frame_h  # blob coordinates are pixels of the model frame
        result = [
            keypoint(
                r["centroid"][0] * sx,
                r["centroid"][1] * sy,
                size,
                label=label,
                from_name=from_name,
                to_name=to_name,
                origin="prediction",
                width=0.5,
                id=f"density-{r['id']}",
                meta={"text": [f"density count ~{r['count']:.2f}"]},
            )
            for r in image["regions"]
            if r["count"] >= min_count
        ]
        out.append(Prelabel(image["name"], result))
    return out


def push_prelabels(
    ls: LabelStudio,
    project_id: int,
    blobs: dict,
    *,
    label: str = PREDICTION_LABEL,
    manual_label: str = "chicken",
    min_count: float = 0.5,
    check_only: bool = False,
    log: Log = print,
) -> dict:
    """Density blobs as a prediction layer on the project's existing tasks.

    Each image's prediction is created once and updated on later runs (keyed by
    model_version), never duplicated. The label is added to the project's
    keypoint control first, since LS validates prediction labels.
    """
    project = ls.project(project_id)
    root, control, labels = keypoint_control(project.get("label_config") or "", manual_label)
    config_changed = label not in labels
    if config_changed and not check_only:
        add_label(control, label, background=PREDICTION_COLOR)
        ls.update_project(project_id, label_config=config_xml(root), show_collab_predictions=True)
        log(f"  added label {label!r} to control {control.attrib['name']!r}")

    model_version = f"density-keypoints:{label}"
    exact, by_key = _task_index(ls.tasks(project_id), name_of=lambda n: Path(n).stem)
    counts = {"matched": 0, "missing": 0, "repeated": 0, "points": 0, "created": 0, "updated": 0}
    counts["config_changed"] = config_changed
    used: set[int] = set()
    items = prelabels(
        blobs, from_name=control.attrib["name"], to_name=control.attrib["toName"], label=label, min_count=min_count
    )
    for item in items:
        candidates = _find_task(item.name, exact, by_key)
        task = _oldest(candidates) if candidates else None
        if task is None:
            counts["missing"] += 1
            log(f"  [warn] {item.name}: no task for this image; skipped")
            continue
        if task["id"] in used:
            counts["repeated"] += 1
            log(f"  [warn] {item.name}: task #{task['id']} already took another image's blobs; skipped")
            continue
        used.add(task["id"])
        counts["matched"] += 1
        counts["points"] += len(item.result)
        prior = [p for p in task.get("predictions") or [] if p.get("model_version") == model_version]
        if check_only:
            continue
        if prior:
            ls.update_prediction(min(p["id"] for p in prior), task["id"], item.result, model_version)
            counts["updated"] += 1
        else:
            ls.create_prediction(task["id"], item.result, model_version)
            counts["created"] += 1
    return counts
