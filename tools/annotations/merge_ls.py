"""Merge several Label Studio keypoint exports (all_ls.json) into one file.

Each Label Studio export is a JSON list of tasks:
    [{"data": {"img": ...}, "annotations": [...]}, ...]

Merging is concatenation of those lists. Because different exports may re-label
the same image (e.g. two annotators, or a re-export after fixing a scene), tasks
are de-duplicated by image filename: later inputs win, so list the export you
trust most LAST. Pass --keep-all to skip de-duplication entirely.

Reports tasks and ground-truth keypoints per input file and for the output, so
you can see exactly what each export contributed and what de-duplication cost.

Feed the merged file straight into convert_ls_to_coco.py.

Usage:
    python merge_ls.py a/all_ls.json b/all_ls.json -o all_ls.json
    python merge_ls.py exports/*.json -o all_ls.json          # shell glob
    python merge_ls.py exports/*.json -o all_ls.json --keep-all
"""

import argparse
import json
import os

from convert_ls_to_coco import ls_img_to_filename, task_points


def load_tasks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of Label Studio tasks, got {type(data).__name__}")
    return data


def count_points(tasks: list) -> int:
    """Ground-truth keypoints across `tasks`, counted exactly as the converter will."""
    return sum(len(task_points(task)[0]) for task in tasks)


def merge(paths: list, keep_all: bool):
    """Merge the exports at `paths`.

    Returns the merged tasks, one `(path, n_tasks, n_points)` row per input, and
    how many duplicate images were collapsed. Per-input rows are reported rather
    than a single total because de-duplication keeps only the LAST copy of a
    repeated image, so the output count is not the sum of the inputs.
    """
    merged = []  # order-preserving list of tasks (only used when keep_all)
    by_name = {}  # file_name -> task, later inputs overwrite earlier ones
    order = []  # file_names in first-seen order, to keep output stable
    per_input = []
    n_dupes = 0

    for path in paths:
        tasks = load_tasks(path)
        per_input.append((path, len(tasks), count_points(tasks)))
        for task in tasks:
            if keep_all:
                merged.append(task)
                continue
            name = ls_img_to_filename(task.get("data", {}).get("img", ""))
            if name in by_name:
                n_dupes += 1
            else:
                order.append(name)
            by_name[name] = task

    if keep_all:
        return merged, per_input, 0
    return [by_name[name] for name in order], per_input, n_dupes


NAME_COL_MAX = 58


def _elide(path: str) -> str:
    """Trim a long path from the left; the tail identifies the file.

    ASCII only: this prints to a Windows console whose codepage is often cp1252.
    """
    text = str(path)
    return text if len(text) <= NAME_COL_MAX else "..." + text[-(NAME_COL_MAX - 3) :]


def _report(per_input: list, out_path: str, out_tasks: list, n_dupes: int, keep_all: bool) -> None:
    """Print an aligned per-file table of tasks and ground-truth points."""
    rows = [(_elide(path), n_tasks, n_points) for path, n_tasks, n_points in per_input]
    rows.append((_elide(out_path), len(out_tasks), count_points(out_tasks)))

    name_w = max(len(path) for path, _, _ in rows)
    task_w = max(len(f"{n:,}") for _, n, _ in rows)
    pts_w = max(len(f"{n:,}") for _, _, n in rows)

    def line(path, n_tasks, n_points):
        return f"  {path:<{name_w}}  {n_tasks:>{task_w},} tasks  {n_points:>{pts_w},} GT points"

    print("Inputs:")
    for row in rows[:-1]:
        print(line(*row))
    print("Output:")
    print(line(*rows[-1]))

    if keep_all:
        print("  (--keep-all: no de-duplication)")
    elif n_dupes:
        print(f"  ({n_dupes} duplicate image(s) collapsed; later inputs win)")
    else:
        print("  (no duplicate images across inputs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Label Studio export json files to merge (later ones win on dupes)")
    parser.add_argument("-o", "--output", default="all_ls.json", help="merged output json")
    parser.add_argument(
        "--keep-all", action="store_true", help="do not de-duplicate; keep every task even if images repeat"
    )
    args = parser.parse_args()

    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        raise SystemExit("Input file(s) not found:\n" + "\n".join(f"  - {p}" for p in missing))

    tasks, per_input, n_dupes = merge(args.inputs, args.keep_all)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    _report(per_input, args.output, tasks, n_dupes, args.keep_all)


if __name__ == "__main__":
    main()
