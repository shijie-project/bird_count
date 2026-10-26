"""The file pipeline from Label Studio exports to the training split, and back.

    LS exports --merge--> one export --convert--> annotations/all.json --split--> train.json / val.json
                                                         |                         + images/train|val/
                                                         +--to_tasks--> LS tasks (re-import)

`all.json`, `train.json` and `val.json` share one format, read by
`training.data.annotations`:

    {"images":      [{"id": 1, "file_name": "001.jpg", "width": 1920, "height": 1080}, ...],
     "annotations": [{"image_id": "001", "points": [[x_px, y_px], ...]}, ...]}

`image_id` is the file stem; points are pixels of the original image.

numpy-free; the image copies use shutil.
"""

from __future__ import annotations

import random
import re
import shutil
import string
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from birdcount.core.camera_identity import extract_mac
from birdcount.core.paths import IMAGE_SUFFIXES
from birdcount.training.labeling.ls_format import (
    count_points,
    image_key,
    keypoint,
    local_files_url,
    task_file,
    task_points,
)


# LS's own result ids: 10 URL-safe characters. Only unique within a task.
_ID_CHARS = string.ascii_letters + string.digits + "-_"


def _result_id() -> str:
    return "".join(random.choices(_ID_CHARS, k=10))


# -- export -> dataset ---------------------------------------------------------------


def export_to_dataset(tasks: list[dict]) -> dict:
    """Annotated tasks as a dataset; tasks without points are left out."""
    images, annotations = [], []
    for task in tasks:
        name = task_file(task)
        points, (width, height) = task_points(task)
        if not points:
            continue
        images.append({"id": len(images) + 1, "file_name": name, "width": width, "height": height})
        annotations.append({"image_id": Path(name).stem, "points": points})
    return {"images": images, "annotations": annotations}


def dataset_to_tasks(dataset: dict, prefix: str, *, label: str = "chicken") -> tuple[list[dict], list[str]]:
    """LS tasks for a dataset's annotated images, and the annotation ids with no image.

    Only what LS needs to import: it assigns task / annotation ids and times.
    """
    images = {Path(im["file_name"]).stem: im for im in dataset.get("images", [])}
    tasks, orphans = [], []
    for ann in dataset.get("annotations", []):
        image = images.get(str(ann["image_id"]))
        if image is None:
            orphans.append(str(ann["image_id"]))
            continue
        size = (image["width"], image["height"])
        result = [keypoint(x, y, size, label=label, id=_result_id()) for x, y in ann["points"]]
        tasks.append(
            {"data": {"img": local_files_url(image["file_name"], prefix)}, "annotations": [{"result": result}]}
        )
    return tasks, orphans


# -- merging exports -------------------------------------------------------------------


@dataclass
class MergeReport:
    inputs: list[tuple[str, int, int]] = field(default_factory=list)  # (name, tasks, points)
    duplicates: int = 0  # repeated images collapsed (later inputs win)
    kept_labelled: int = 0  # of those, a labelled copy kept over a later empty one


def merge_exports(exports: list[tuple[str, list[dict]]], *, keep_all: bool = False) -> tuple[list[dict], MergeReport]:
    """Concatenate exports, one task per image (`image_key`); later exports win.

    Except that a task WITH points is never replaced by one without: a re-export
    often repeats an uploaded image with its labels stripped.
    """
    report = MergeReport()
    merged: list[dict] = []
    by_key: dict[str, tuple[dict, int]] = {}
    for name, tasks in exports:
        report.inputs.append((name, len(tasks), count_points(tasks)))
        for task in tasks:
            if keep_all:
                merged.append(task)
                continue
            key = image_key(task_file(task))
            n = len(task_points(task)[0])
            previous = by_key.get(key)
            if previous is not None:
                report.duplicates += 1
                if not n and previous[1]:
                    report.kept_labelled += 1
                    continue
            by_key[key] = (task, n)
    return (merged if keep_all else [task for task, _ in by_key.values()]), report


@dataclass
class Roster:
    images: int  # in the folder
    covered: int  # already in the tasks
    added: list[str]  # got an empty task
    twins: int  # a second spelling of an image already in the output
    unmatched: int  # tasks whose image is not in the folder (kept)
    ambiguous: list[str]  # task names matching several files; left unmatched


def list_images(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def add_missing_images(tasks: list[dict], names: list[str], prefix: str) -> Roster:
    """Append an empty task (in place) for every image in `names` the tasks do not cover.

    A task covers a file by exact name, else by `image_key` when that is unique.
    """
    exact, by_key = set(names), defaultdict(list)
    for name in names:
        by_key[image_key(name)].append(name)
    covered, ambiguous, unmatched = set(), [], 0
    for task in tasks:
        name = task_file(task)
        if name in exact:
            covered.add(name)
            continue
        candidates = by_key.get(image_key(name), []) if name else []
        if len(candidates) == 1:
            covered.add(candidates[0])
        else:
            unmatched += 1
            if len(candidates) > 1:
                ambiguous.append(name)

    seen = {image_key(task_file(t)) for t in tasks}
    added, twins = [], 0
    for name in names:
        if name in covered:
            continue
        if image_key(name) in seen:  # "foo.png" beside its LS-uploaded "c08fbd05-foo.png"
            twins += 1
            continue
        tasks.append({"data": {"img": local_files_url(name, prefix)}, "annotations": []})
        seen.add(image_key(name))
        added.append(name)
    return Roster(len(names), len(covered), added, twins, unmatched, sorted(set(ambiguous)))


# -- train / val split -------------------------------------------------------------------

# A frame's recording session: the source video's start (+ segment id) in the
# name, e.g. ..._20260325_164628_F92A_frame926... or MAC_20260808_065008...
_SESSION = (re.compile(r"(\d{8}_\d{6})_[0-9A-F]{4}"), re.compile(r"[0-9A-F]{12}_(\d{8}_\d{6})"))
# Old frames named 20210404_08_73.jpg: grouped by date + hour.
_OLD_SESSION = re.compile(r"(\d{8}_\d{2})_")
# MAC-less batches whose camera was identified by eye against the recordings.
CAMERA_BY_PREFIX = {"20260311_": "B8A44FD51C3C"}
UNKNOWN_CAMERA = "unknown"


def camera_of(file_name: str) -> str:
    mac = extract_mac(file_name)
    if mac:
        return mac
    for prefix, camera in CAMERA_BY_PREFIX.items():
        if file_name.startswith(prefix):
            return camera
    return UNKNOWN_CAMERA


def session_of(file_name: str) -> str:
    if camera_of(file_name) != UNKNOWN_CAMERA:
        for pattern in _SESSION:
            if match := pattern.search(file_name):
                return match.group(1)
    match = _OLD_SESSION.match(file_name)
    return match.group(1) if match else file_name


def split_images(images: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Per camera, whole recording sessions to val until ~`val_ratio` of its images.

    Frames of one session are near-duplicates, so a session never straddles the
    split (val would score memorised scenes). Every camera with more than one
    session gets at least one in val. Deterministic for a seed.

    Returns (train, val, {camera: (n_train, n_val)}).
    """
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for image in images:
        name = str(image.get("file_name", ""))
        groups[camera_of(name)][session_of(name)].append(image)

    rng = random.Random(seed)
    train, val, per_camera = [], [], {}
    for camera in sorted(groups):
        sessions = list(groups[camera].values())
        rng.shuffle(sessions)
        quota = max(1, round(sum(len(s) for s in sessions) * val_ratio))
        chosen: list[dict] = []
        for session in sessions:
            if len(chosen) + len(session) <= quota:
                chosen += session
        if not chosen and len(sessions) > 1:  # every session exceeds the quota: still give val the smallest
            chosen = min(sessions, key=len)
        ids = {id(im) for im in chosen}
        kept = [im for session in sessions for im in session if id(im) not in ids]
        train += kept
        val += chosen
        per_camera[camera] = (len(kept), len(chosen))
    return train, val, per_camera


def split_dataset(dataset: dict, images: list[dict]) -> dict:
    """The part of `dataset` covering `images`, in image-id order."""
    images = sorted(images, key=lambda im: im["id"])
    by_stem = {str(a["image_id"]): a for a in dataset.get("annotations", [])}
    stems = [Path(im["file_name"]).stem for im in images]
    return {"images": images, "annotations": [by_stem[s] for s in stems if s in by_stem]}


def place_images(images: list[dict], src: Path, dst: Path, *, move: bool = False) -> list[str]:
    """Empty `dst`, then copy (or move) the images into it; returns names missing from `src`."""
    if dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    op = shutil.move if move else shutil.copy2
    missing = []
    for im in images:
        source = src / im["file_name"]
        if source.is_file():
            op(source, dst / im["file_name"])
        else:
            missing.append(im["file_name"])
    return missing
