"""The scripts the web UI can drive, and how to read their CLI spec.

Forms are generated from each script's real `argparse` definition rather than a
hand-written copy, so adding a flag to a script adds a field to the UI with its
help text and default, and nothing here needs to change.

Extraction runs in a subprocess on purpose: importing train.py pulls in torch
and the whole trainer stack, which we do not want resident inside the web
server. Results are cached for the lifetime of the server.
"""

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Entrypoint:
    """One runnable script.

    `pythonpath` entries (relative to ROOT) are prepended to PYTHONPATH so a
    script that imports a sibling module still works while running from ROOT —
    keeping every path the user types relative to the project root, whichever
    tool is selected.
    """

    key: str
    script: str
    label: str
    page: str
    blurb: str = ""
    pythonpath: tuple[str, ...] = ()


_ANNOTATION_PATH = ("tools/annotations",)
_LS_OPS_PATH = ("tools/ls_ops",)
# A script under tools/ that imports the project's own packages (models,
# datasets, utils) needs ROOT on the path: running `python tools/x.py` puts
# tools/ on sys.path, not the root, and `import datasets` would then find the
# pip-installed HuggingFace package instead of ours. "tools" is there so the
# spec extractor can import the module and call its build_parser() directly.
_ROOT_PATH = (".", "tools")

ENTRYPOINTS: dict[str, Entrypoint] = {
    e.key: e
    for e in [
        # `page` is the tab the tool lives on; a tab with several tools gets a
        # picker (see the annotation pipeline below).
        Entrypoint("train", "train.py", "Train", "train", "Train the ShuffleNet density model."),
        Entrypoint("test", "test.py", "Test", "test", "Evaluate a checkpoint on a dataset split."),
        # Live-project operations (tools/ls_ops/): one script per operation.
        Entrypoint(
            "ls_dedupe",
            "tools/ls_ops/dedupe_annotations.py",
            "LS · dedupe annotations",
            "annotations",
            "Keep one annotation per task in a live Label Studio project and delete the duplicates. "
            "Dry run unless --apply is ticked.",
            _LS_OPS_PATH,
        ),
        # Pre-annotation: predict first, then hand the regions to Label Studio
        # so the counts are on screen while you place points.
        Entrypoint(
            "density_regions",
            "tools/density_regions.py",
            "Density regions",
            "annotations",
            "Split the predicted density map into regions and report how many chickens each one holds. "
            "Writes overlay PNGs plus a regions.json for the next step.",
            _ROOT_PATH,
        ),
        Entrypoint(
            "regions_to_label_studio",
            "tools/annotations/regions_to_label_studio.py",
            "Regions → LS pre-labels",
            "annotations",
            "Turn a regions.json into a Label Studio prediction layer: one box per region, labeled with its "
            "predicted count. Run with --print-config first to get the matching labeling interface.",
            _ANNOTATION_PATH,
        ),
        # The file pipeline (tools/annotations/), in the order you normally run it.
        Entrypoint(
            "merge_ls",
            "tools/annotations/merge_ls.py",
            "Merge LS exports",
            "annotations",
            "Concatenate several Label Studio exports into one, de-duplicating by image (later inputs win).",
            _ANNOTATION_PATH,
        ),
        Entrypoint(
            "convert_ls_to_coco",
            "tools/annotations/convert_ls_to_coco.py",
            "LS → training JSON",
            "annotations",
            "Convert a Label Studio keypoint export into the COCO-ish format BirdDataset reads.",
            _ANNOTATION_PATH,
        ),
        Entrypoint(
            "drop_masked_annotations",
            "tools/annotations/drop_masked_annotations.py",
            "Drop masked points",
            "annotations",
            "Remove keypoints that fall inside a blacked-out region of the image.",
            _ANNOTATION_PATH,
        ),
        Entrypoint(
            "split_train_val",
            "tools/annotations/split_train_val.py",
            "Split train / val",
            "annotations",
            "Split all.json + images/all into train and val at a given ratio.",
            _ANNOTATION_PATH,
        ),
        Entrypoint(
            "to_label_studio",
            "tools/annotations/to_label_studio.py",
            "Training JSON → LS",
            "annotations",
            "Turn point annotations back into Label Studio keypoint tasks for re-review.",
            _ANNOTATION_PATH,
        ),
    ]
}

_SENTINEL = "<<<ARGPARSE-JSON>>>"

# Runs inside the target interpreter. Scripts that expose build_parser() are
# introspected directly; for the rest we execute the file as __main__ with
# parse_args patched to hand back the parser and abort — every script builds its
# parser as the first thing main() does, so no work is performed.
_DUMP_SNIPPET = f"""
import argparse, json, runpy, sys
from pathlib import Path

target = sys.argv[1]
sys.argv = [target]


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def _load_parser(path):
    name = Path(path).stem
    try:
        module = __import__(name)
        if hasattr(module, "build_parser"):
            return module.build_parser()
    except Exception:
        pass

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, *a, **k: (_ for _ in ()).throw(_Captured(self))
    try:
        runpy.run_path(path, run_name="__main__")
    except _Captured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise SystemExit("script never called parse_args()")


parser = _load_parser(target)


def kind_of(action):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.choices:
        return "choice"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "str"


def is_multi(action):
    if isinstance(action.nargs, str):
        return action.nargs in ("+", "*")
    return isinstance(action.nargs, int) and action.nargs > 1


groups = []
for group in parser._action_groups:
    options = []
    for action in group._group_actions:
        if isinstance(action, argparse._HelpAction):
            continue
        positional = not action.option_strings
        options.append({{
            "dest": action.dest,
            "flag": max(action.option_strings, key=len) if action.option_strings else action.dest,
            "kind": kind_of(action),
            "default": action.default,
            "choices": list(action.choices) if action.choices else None,
            "multi": is_multi(action),
            "positional": positional,
            "required": bool(action.required) or (positional and action.nargs not in ("?", "*")),
            "help": (action.help or "").strip(),
        }})
    if options:
        groups.append({{"title": group.title or "options", "options": options}})

print({_SENTINEL!r} + json.dumps({{"description": parser.description, "groups": groups}}))
"""


# Extracting one spec means starting a Python that imports the script — for
# train.py and test.py that is the whole torch stack, ~2.6s each. The result only
# changes when the script does, so it is kept in memory *and* on disk, and the
# server warms every entrypoint in the background at startup: a tab switch should
# never wait on a subprocess.
CACHE_PATH = ROOT / "logs" / "webui" / "schema-cache.json"

_cache: dict[str, dict] = {}
_key_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_disk_cache: Optional[dict] = None
_disk_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _script_stamp(entry: Entrypoint) -> str:
    """Identity of the script on disk; any change invalidates the cached spec."""
    try:
        stat = (ROOT / entry.script).stat()
    except OSError:
        return ""
    return f"{int(stat.st_mtime)}:{stat.st_size}"


def _read_disk_cache() -> dict:
    global _disk_cache
    if _disk_cache is None:
        try:
            _disk_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # missing or corrupt: start over
            _disk_cache = {}
    return _disk_cache


def _write_disk_cache() -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_disk_cache), encoding="utf-8")
    except OSError:  # a cache that cannot be persisted still works in memory
        pass


def _env_for(entry: Entrypoint) -> dict:
    env = os.environ.copy()
    # The parent reads the child's stdout as UTF-8; without this the child would
    # encode with the console codepage (cp1252 on Windows) and mangle non-ASCII.
    env["PYTHONIOENCODING"] = "utf-8"
    if entry.pythonpath:
        extra = [str(ROOT / p) for p in entry.pythonpath]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([*extra, existing]) if existing else os.pathsep.join(extra)
    return env


def list_entrypoints() -> list[dict]:
    return [
        {"key": e.key, "label": e.label, "page": e.page, "blurb": e.blurb, "script": e.script}
        for e in ENTRYPOINTS.values()
    ]


def _extract(entry: Entrypoint) -> dict:
    """Run the script's own argparse in a subprocess and bring back its spec."""
    proc = subprocess.run(
        [sys.executable, "-c", _DUMP_SNIPPET, entry.script],
        cwd=ROOT,
        env=_env_for(entry),
        capture_output=True,
        text=True,
    )
    marker = proc.stdout.rfind(_SENTINEL)
    if proc.returncode != 0 or marker < 0:
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        raise RuntimeError(f"could not read the CLI spec of {entry.script}:\n{detail}")
    return json.loads(proc.stdout[marker + len(_SENTINEL) :])


def get_schema(key: str) -> dict:
    """Return the JSON-able argparse description for one entrypoint."""
    cached = _cache.get(key)
    if cached is not None:
        return cached
    entry = ENTRYPOINTS[key]  # KeyError -> 404 at the route

    with _lock_for(key):  # two requests for one script must not both extract it
        cached = _cache.get(key)
        if cached is not None:
            return cached

        stamp = _script_stamp(entry)
        with _disk_guard:
            saved = _read_disk_cache().get(key)
        if stamp and saved and saved.get("stamp") == stamp:
            spec = saved["spec"]
        else:
            spec = _extract(entry)
            with _disk_guard:
                _read_disk_cache()[key] = {"stamp": stamp, "spec": spec}
                _write_disk_cache()

        # Label, page and blurb come from the code above, not from the script, so
        # they are merged in on the way out: editing a blurb costs no re-extract.
        schema = {
            **spec,
            "key": key,
            "label": entry.label,
            "page": entry.page,
            "script": entry.script,
            "blurb": entry.blurb,
        }
        _cache[key] = schema
        return schema


def warm_cache() -> None:
    """Extract every spec up front. The server runs this in a background thread."""
    for key in ENTRYPOINTS:
        try:
            get_schema(key)
        except Exception:  # one broken script must not stop the rest warming
            pass


def _values_of(spec: dict, value) -> list[str]:
    """Normalize a form value into the argv fragments for one option."""
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
    elif spec["multi"]:
        # Multi-valued fields are edited as one entry per line, so a path
        # containing spaces survives the round trip.
        items = [line.strip() for line in str(value).splitlines()]
    else:
        items = [str(value).strip()]
    return [item for item in items if item]


def build_argv(key: str, values: dict) -> tuple[list[str], dict]:
    """Turn `{dest: value}` from the form into an argv list plus the run env.

    Only flags present in the schema are accepted, so nothing a browser sends
    can inject an arbitrary command. Empty values mean "leave at default" and
    are dropped; booleans become bare store_true flags. Positionals are emitted
    last so they cannot be swallowed by a preceding variadic option.
    """
    entry = ENTRYPOINTS[key]
    schema = get_schema(key)
    specs = [opt for group in schema["groups"] for opt in group["options"]]

    argv = [sys.executable, "-u", entry.script]
    positionals: list[str] = []
    missing: list[str] = []

    for spec in specs:
        value = values.get(spec["dest"])
        if spec["kind"] == "bool":
            if value:
                argv.append(spec["flag"])
            continue

        items = [] if value is None else _values_of(spec, value)
        if not items:
            if spec["required"]:
                missing.append(spec["flag"])
            continue
        if spec["positional"]:
            positionals.extend(items)
        else:
            argv.append(spec["flag"])
            argv.extend(items)

    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}")
    return argv + positionals, _env_for(entry)
