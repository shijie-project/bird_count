"""Merge several Label Studio keypoint exports (all_ls.json) into one file.

Each Label Studio export is a JSON list of tasks:
    [{"data": {"img": ...}, "annotations": [...]}, ...]

Merging is concatenation of those lists. Because different exports may re-label
the same image (e.g. two annotators, or a re-export after fixing a scene), tasks
are de-duplicated by image filename: later inputs win, so list the export you
trust most LAST. Pass --keep-all to skip de-duplication entirely.

Feed the merged file straight into convert_ls_to_coco.py.

Usage:
    python merge_ls.py a/all_ls.json b/all_ls.json -o all_ls.json
    python merge_ls.py exports/*.json -o all_ls.json          # shell glob
    python merge_ls.py exports/*.json -o all_ls.json --keep-all
"""

import argparse
import json
import os

from convert_ls_to_coco import ls_img_to_filename


def load_tasks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of Label Studio tasks, got {type(data).__name__}")
    return data


def merge(paths: list, keep_all: bool):
    merged = []  # order-preserving list of tasks (only used when keep_all)
    by_name = {}  # file_name -> task, later inputs overwrite earlier ones
    order = []  # file_names in first-seen order, to keep output stable
    n_in = 0
    n_dupes = 0

    for path in paths:
        tasks = load_tasks(path)
        n_in += len(tasks)
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
        return merged, n_in, 0
    return [by_name[name] for name in order], n_in, n_dupes


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

    tasks, n_in, n_dupes = merge(args.inputs, args.keep_all)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(args.inputs)} file(s): {n_in} tasks in -> {len(tasks)} tasks out", end="")
    if not args.keep_all:
        print(f" ({n_dupes} duplicate image(s) collapsed)")
    else:
        print(" (--keep-all: no de-duplication)")


if __name__ == "__main__":
    main()
