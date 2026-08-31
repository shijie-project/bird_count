"""Merge several Label Studio keypoint exports (all_ls.json) into one file.

Each Label Studio export is a JSON list of tasks:
    [{"data": {"img": ...}, "annotations": [...]}, ...]

Merging is concatenation of those lists. Because different exports may re-label
the same image (e.g. two annotators, or a re-export after fixing a scene), tasks
are de-duplicated by image identity: later inputs win, so list the export you
trust most LAST. One exception, because it is always data loss: a task WITH
labels is never replaced by one without. Pass --keep-all to skip de-duplication
entirely.

Image identity ignores how Label Studio happens to spell the path — the URL is
unquoted (%5C and friends), separators normalised, the basename taken, and the
upload hash LS prepends to files added through its UI is dropped. That is what
lets "/data/upload/2/61701da2-020_axis4.jpg" and
"/data/local-files/?d=annotated%5Cimages%5Call%5C020_axis4.jpg" recognise each
other, and match "020_axis4.jpg" in --images-dir.

Reports tasks and ground-truth keypoints per input file and for the output, so
you can see exactly what each export contributed and what de-duplication cost.

With --images-dir the image folder becomes the roster: every image in it ends up
in the output. Images the inputs already label keep those labels; images nobody
labelled get an empty task, ready to annotate in Label Studio. Tasks whose image
is missing from the folder are still kept (and counted), never silently dropped.

Empty tasks are shaped after --template, so they carry exactly the keys your
Label Studio version writes. Without --template the first merged input task is
used as the shape; with no inputs at all, a minimal task is emitted.

Feed the merged file straight into convert_ls_to_coco.py.

Usage:
    python merge_ls.py a/all_ls.json b/all_ls.json -o all_ls.json
    python merge_ls.py exports/*.json -o all_ls.json          # shell glob
    python merge_ls.py exports/*.json -o all_ls.json --keep-all

    # every image in the folder, labelled where we have labels:
    python merge_ls.py a/all_ls.json -o all_ls.json \\
        --images-dir ../../../data/annotated/images/all \\
        --template template.json \\
        --image-prefix "annotated\\images\\all\\"
"""

import argparse
import json
import os

from convert_ls_to_coco import image_key, ls_img_to_filename, task_points
from to_label_studio import DEFAULT_IMAGE_PREFIX, local_files_url, ls_timestamp


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Task fields that describe *work done on* a task rather than the task itself.
# Only those already present in the template are reset, so we never invent keys
# a given Label Studio version does not use.
_BLANK_FIELDS = {
    "annotations": [],
    "drafts": [],
    "predictions": [],
    "meta": {},
    "total_annotations": 0,
    "cancelled_annotations": 0,
    "total_predictions": 0,
    "comment_count": 0,
    "unresolved_comment_count": 0,
    "last_comment_updated_at": None,
    "comment_authors": [],
}

_MINIMAL_TEMPLATE = {"data": {"img": ""}, "annotations": []}


def load_tasks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of Label Studio tasks, got {type(data).__name__}")
    return data


def load_template(path: str) -> dict:
    """Read --template, accepting either a single task object or an LS export list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path}: template export is empty")
        return data[0]
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a task object or a list of tasks, got {type(data).__name__}")
    return data


def list_images(images_dir: str) -> list:
    """Image file names directly inside `images_dir`, sorted, case-insensitive extensions."""
    if not os.path.isdir(images_dir):
        raise ValueError(f"--images-dir is not a directory: {images_dir}")
    names = [
        entry
        for entry in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, entry)) and os.path.splitext(entry)[1].lower() in IMAGE_EXTS
    ]
    return sorted(names)


def blank_task(template: dict, file_name: str, image_prefix: str, task_id: int, now: str) -> dict:
    """A copy of `template` pointing at `file_name` with all annotation work cleared."""
    task = json.loads(json.dumps(template))  # deep copy of plain JSON data

    for key, empty in _BLANK_FIELDS.items():
        if key in task:
            task[key] = json.loads(json.dumps(empty))
    task["annotations"] = []  # required even if the template lacked the key

    data = task.get("data")
    if not isinstance(data, dict):
        data = {}
    data["img"] = local_files_url(file_name, image_prefix)
    task["data"] = data

    for key in ("id", "inner_id"):
        if key in task:
            task[key] = task_id
    for key in ("created_at", "updated_at"):
        if key in task:
            task[key] = now

    return task


def resolve_to_folder(tasks: list, names: list) -> tuple:
    """Map each task onto the folder file it refers to.

    An exact file-name match always wins; only then do we fall back to the
    canonical key (see `image_key`), which folds away Label Studio's upload hash
    so "/data/upload/2/61701da2-020_axis4.jpg" finds "020_axis4.jpg" on disk. An
    ambiguous fallback (several files collapsing to one key) is left unmatched
    rather than guessed at.

    Returns (covered_names, n_unmatched, ambiguous_names).
    """
    exact = set(names)
    by_key = {}
    for name in names:
        by_key.setdefault(image_key(name), []).append(name)

    covered = set()
    unmatched = 0
    ambiguous = set()

    for task in tasks:
        file_name = ls_img_to_filename(task.get("data", {}).get("img", ""))
        if file_name in exact:
            covered.add(file_name)
            continue
        candidates = by_key.get(image_key(file_name), []) if file_name else []
        if len(candidates) == 1:
            covered.add(candidates[0])
        else:
            if len(candidates) > 1:
                ambiguous.add(file_name)
            unmatched += 1

    return covered, unmatched, ambiguous


def add_missing_images(tasks: list, images_dir: str, template: dict, image_prefix: str) -> tuple:
    """Append an empty task for every image in `images_dir` the tasks do not cover.

    Returns (tasks, n_images, n_covered, added_names, n_unmatched, n_twins, ambiguous).
    """
    names = list_images(images_dir)
    covered, n_unmatched, ambiguous = resolve_to_folder(tasks, names)

    # A folder can hold the same picture twice under two spellings — "foo.png"
    # next to the LS-uploaded "c08fbd05-foo.png". Only one of them matched a
    # task; adding the other would put the same image in the output twice.
    seen_keys = {image_key(ls_img_to_filename(t.get("data", {}).get("img", ""))) for t in tasks}

    numeric_ids = [t["id"] for t in tasks if isinstance(t.get("id"), int)]
    next_id = max(numeric_ids, default=0) + 1
    now = ls_timestamp()

    added_names = []
    n_twins = 0
    for name in names:
        if name in covered:
            continue
        key = image_key(name)
        if key in seen_keys:
            n_twins += 1
            continue
        tasks.append(blank_task(template, name, image_prefix, next_id, now))
        seen_keys.add(key)
        next_id += 1
        added_names.append(name)

    return tasks, len(names), len(covered), added_names, n_unmatched, n_twins, ambiguous


def count_points(tasks: list) -> int:
    """Ground-truth keypoints across `tasks`, counted exactly as the converter will."""
    return sum(len(task_points(task)[0]) for task in tasks)


def merge(paths: list, keep_all: bool):
    """Merge the exports at `paths`.

    Returns the merged tasks, one `(path, n_tasks, n_points)` row per input, how
    many duplicate images were collapsed, and how many times an earlier labelled
    copy was preferred over a later empty one. Per-input rows are reported rather
    than a single total because de-duplication drops repeats, so the output count
    is not the sum of the inputs.
    """
    merged = []  # order-preserving list of tasks (only used when keep_all)
    by_name = {}  # image key -> (task, n_points)
    order = []  # image keys in first-seen order, to keep output stable
    per_input = []
    n_dupes = 0
    n_kept_labelled = 0

    for path in paths:
        tasks = load_tasks(path)
        per_input.append((path, len(tasks), count_points(tasks)))
        for task in tasks:
            if keep_all:
                merged.append(task)
                continue
            # Key on the canonical image identity so the same picture exported
            # once as an upload and once via local-files still de-duplicates.
            name = image_key(ls_img_to_filename(task.get("data", {}).get("img", "")))
            n_points = len(task_points(task)[0])
            previous = by_name.get(name)

            if previous is None:
                order.append(name)
            else:
                n_dupes += 1
                # "Later wins" must never trade labels for an empty re-import:
                # a re-export via local-files often repeats an uploaded image
                # with its annotations stripped.
                if not n_points and previous[1]:
                    n_kept_labelled += 1
                    continue

            by_name[name] = (task, n_points)

    if keep_all:
        return merged, per_input, 0, 0
    return [by_name[name][0] for name in order], per_input, n_dupes, n_kept_labelled


NAME_COL_MAX = 58


def _elide(path: str) -> str:
    """Trim a long path from the left; the tail identifies the file.

    ASCII only: this prints to a Windows console whose codepage is often cp1252.
    """
    text = str(path)
    return text if len(text) <= NAME_COL_MAX else "..." + text[-(NAME_COL_MAX - 3) :]


def _report(
    per_input: list,
    out_path: str,
    out_tasks: list,
    n_dupes: int,
    keep_all: bool,
    roster=None,
    n_kept_labelled: int = 0,
) -> None:
    """Print an aligned per-file table of tasks and ground-truth points."""
    rows = [(_elide(path), n_tasks, n_points) for path, n_tasks, n_points in per_input]
    rows.append((_elide(out_path), len(out_tasks), count_points(out_tasks)))

    name_w = max(len(path) for path, _, _ in rows)
    task_w = max(len(f"{n:,}") for _, n, _ in rows)
    pts_w = max(len(f"{n:,}") for _, _, n in rows)

    def line(path, n_tasks, n_points):
        return f"  {path:<{name_w}}  {n_tasks:>{task_w},} tasks  {n_points:>{pts_w},} GT points"

    if len(rows) > 1:
        print("Inputs:")
        for row in rows[:-1]:
            print(line(*row))

    if roster:
        images_dir, n_images, n_covered, added_names, n_unmatched, n_twins, ambiguous = roster
        print("Images dir:")
        print(f"  {_elide(images_dir):<{name_w}}  {n_images:>{task_w},} images")
        print(f"    {n_covered:,} already in input exports, {len(added_names):,} added empty")
        if added_names:
            print("    Added empty tasks from --images-dir (missing from every input export):")
            for name in added_names:
                print(f"      - {name}")
        if n_twins:
            print(f"    {n_twins:,} skipped as a second copy of an image already in the output")
        if n_unmatched:
            print(f"    {n_unmatched:,} task(s) kept whose image is not in this folder")
        if ambiguous:
            print(f"    {len(ambiguous):,} task(s) matched several files by name; left unmatched:")
            for name in sorted(ambiguous)[:5]:
                print(f"      - {name}")

    print("Output:")
    print(line(*rows[-1]))

    if keep_all:
        print("  (--keep-all: no de-duplication)")
    elif n_dupes:
        print(f"  ({n_dupes} duplicate image(s) collapsed; later inputs win)")
        if n_kept_labelled:
            print(f"  ({n_kept_labelled} of them kept the labelled copy over a later empty one)")
    else:
        print("  (no duplicate images across inputs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # "*" not "+": with --images-dir and --template you can build a fresh,
    # fully unlabelled project straight from a folder of images.
    parser.add_argument("inputs", nargs="*", help="Label Studio export json files to merge (later ones win on dupes)")
    parser.add_argument("-o", "--output", default="all_ls.json", help="merged output json")
    parser.add_argument(
        "--keep-all", action="store_true", help="do not de-duplicate; keep every task even if images repeat"
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="add an empty task for every image in this folder that the inputs do not already label",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="task json (single object or export list) whose keys the added empty tasks copy; "
        "defaults to the first merged input task",
    )
    parser.add_argument(
        "--image-prefix",
        default=DEFAULT_IMAGE_PREFIX,
        help=f"path prefix inside LOCAL_FILES_DOCUMENT_ROOT for added tasks (default: {DEFAULT_IMAGE_PREFIX!r})",
    )
    args = parser.parse_args()

    if not args.inputs and not args.images_dir:
        raise SystemExit("Nothing to do: pass at least one input export, or --images-dir to start from images.")

    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        raise SystemExit("Input file(s) not found:\n" + "\n".join(f"  - {p}" for p in missing))

    tasks, per_input, n_dupes, n_kept_labelled = merge(args.inputs, args.keep_all)

    roster = None
    if args.images_dir:
        if args.template:
            template = load_template(args.template)
        else:
            template = tasks[0] if tasks else _MINIMAL_TEMPLATE
        try:
            tasks, *counts = add_missing_images(tasks, args.images_dir, template, args.image_prefix)
        except ValueError as exc:
            raise SystemExit(str(exc))
        roster = (args.images_dir, *counts)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    _report(per_input, args.output, tasks, n_dupes, args.keep_all, roster, n_kept_labelled)


if __name__ == "__main__":
    main()
