"""The tools the panel can start, their forms, and the command line a form becomes.

Each form is generated from the tool's real argparse parser (`build_parser()`),
so a flag added to a tool appears in the panel with its help and default.
Reading the parsers imports the tools — eval and video import torch, ~3 s —
so it runs once, in a subprocess, for every tool together, and is kept on
disk until a source file of the package changes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from birdcount.core.paths import PROJECT_ROOT


@dataclass(frozen=True)
class Tool:
    key: str
    module: str  # what `python -m` runs
    parser: str  # "module:function" returning the argparse parser
    label: str
    page: str
    blurb: str
    command: str | None = None  # subcommand of a multi-command CLI
    gpu: bool = False  # one GPU tool runs at a time
    run: str | None = None  # the run kind it writes (birdcount.training.runs)
    pickers: dict[str, str] = field(default_factory=dict)  # option dest -> picker
    hidden: tuple[str, ...] = ()  # options the panel does not offer
    passthrough: str = ""  # label of a free-text field passed after "--"


_EVAL = "birdcount.training.eval"
_VIDEO = "birdcount.training.video"
_LABELING = "birdcount.training.labeling"


def _labeling(command: str, label: str, page: str, blurb: str, **kw) -> Tool:
    return Tool(
        f"labeling-{command}", _LABELING, f"{_LABELING}.__main__:build_parser", label, page, blurb, command, **kw
    )


TOOLS: dict[str, Tool] = {
    t.key: t
    for t in [
        Tool(
            "train",
            "birdcount.training.train",
            "birdcount.training.train.config:build_parser",
            "Train",
            "train",
            "Train the density model. Leave --resume empty for a new run; pick a run folder to continue it.",
            gpu=True,
            run="train",
            pickers={"init": "checkpoint", "resume": "train_run", "data_root": "folder"},
        ),
        Tool(
            "sweep",
            "birdcount.training.train.sweep",
            "birdcount.training.train.sweep:build_parser",
            "Sweep",
            "train",
            "One training run per combination of the --grid values (one key=v1,v2,... per line), then a "
            "leaderboard. Re-running with the same --sweep-dir resumes it.",
            gpu=True,
            run="sweep",
            pickers={"sweep_dir": "sweep_run"},
            hidden=("summary",),
            passthrough="training flags for every run, e.g. --epochs 200",
        ),
        Tool(
            "eval",
            _EVAL,
            f"{_EVAL}.__main__:build_parser",
            "Evaluate",
            "eval",
            "Evaluate a checkpoint on a dataset split, exactly as the live runtime counts. --blobs and --grid "
            "add per-region errors.",
            gpu=True,
            run="eval",
            pickers={"ckpt": "checkpoint", "split": "split"},
        ),
        Tool(
            "video",
            _VIDEO,
            f"{_VIDEO}.__main__:build_parser",
            "Video timeline",
            "video",
            "Count birds over a video: the flock count and the busiest patch over time, the busiest moments, "
            "optional overlay frames and video.",
            gpu=True,
            run="video",
            pickers={"videos": "video", "ckpt": "checkpoint", "mask": "mask"},
        ),
        _labeling(
            "convert",
            "LS export → dataset",
            "data",
            "Turn a Label Studio export into the dataset json (annotated images only).",
            pickers={"export": "json"},
        ),
        _labeling(
            "merge",
            "Merge LS exports",
            "data",
            "Merge several exports into one, one task per image; later exports win.",
            pickers={"inputs": "json", "images_dir": "folder"},
        ),
        _labeling(
            "split",
            "Split train / val",
            "data",
            "Split annotations/all.json + images/all into train and val per camera, whole recording sessions "
            "on one side.",
        ),
        _labeling(
            "to-ls",
            "Dataset → LS tasks",
            "data",
            "Turn the dataset's points back into Label Studio tasks for review.",
            pickers={"src": "json"},
        ),
        _labeling(
            "drop-masked",
            "Drop masked points",
            "data",
            "Drop points that lie in blacked-out image areas.",
            pickers={"input": "json"},
        ),
        _labeling(
            "dedupe-images",
            "Find duplicate images",
            "data",
            "Find duplicate images; --apply moves them to a quarantine folder (nothing is deleted).",
            pickers={"folder": "folder", "ls_json": "json"},
        ),
        _labeling(
            "apply-mask",
            "Apply masks",
            "data",
            "Write copies of images with each camera's ignored region painted black.",
            pickers={"inputs": "folder", "mask": "mask"},
        ),
        _labeling(
            "pull",
            "Pull project",
            "label_studio",
            "Download the project's tasks and annotations as a Label Studio export.",
            pickers={"project_id": "ls_project"},
        ),
        _labeling(
            "dedupe-annotations",
            "Dedupe annotations",
            "label_studio",
            "Keep one annotation per task. Nothing changes without --apply.",
            pickers={"project_id": "ls_project"},
        ),
        _labeling(
            "import",
            "Import an export",
            "label_studio",
            "Add another machine's annotations to this project's matching images.",
            pickers={"project_id": "ls_project", "src": "json", "images_dir": "folder"},
        ),
        _labeling(
            "prelabel",
            "Pre-label from blobs",
            "label_studio",
            "Density blobs of an `eval --blobs` run as chicken-pred predictions on existing tasks; re-running "
            "updates them.",
            pickers={"project_id": "ls_project", "blobs": "blobs"},
        ),
    ]
}


# --- reading a parser ---------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _option(action: argparse.Action) -> dict:
    """One argparse action as a form field."""
    opt = {
        "dest": action.dest,
        "flag": max(action.option_strings, key=len) if action.option_strings else None,
        "positional": not action.option_strings,
        "default": _jsonable(action.default),
        # the form shows the default as the field's value; a "(default: x)" suffix would repeat it
        "help": re.sub(
            r"\s*\(default: [^)]*\)$",
            "",
            (action.help or "").replace("%(default)s", str(action.default)).replace("%%", "%"),
        ),
        "choices": [_jsonable(c) for c in action.choices] if action.choices else None,
        "multi": False,
        "repeat": False,
        "required": False,
    }
    if isinstance(action, argparse.BooleanOptionalAction):
        long = max((s for s in action.option_strings if not s.startswith("--no-")), key=len)
        return {**opt, "kind": "bool", "flag": long, "on": long, "off": "--no-" + long[2:]}
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        on = isinstance(action, argparse._StoreTrueAction)
        return {**opt, "kind": "bool", "on": opt["flag"] if on else None, "off": None if on else opt["flag"]}
    nargs = action.nargs
    opt["multi"] = nargs in ("+", "*") or (isinstance(nargs, int) and nargs > 1)
    if isinstance(action, argparse._AppendAction):
        opt["multi"] = opt["repeat"] = True
    opt["kind"] = (
        "choice" if action.choices else {int: "int", float: "float", Path: "path"}.get(action.type, "str")  # type: ignore[arg-type]
    )
    opt["required"] = bool(action.required) or (opt["positional"] and nargs not in ("?", "*"))
    return opt


def describe(parser: argparse.ArgumentParser, hidden: tuple[str, ...] = ()) -> dict:
    groups = []
    for group in parser._action_groups:
        options = [
            _option(a)
            for a in group._group_actions
            if not isinstance(a, (argparse._HelpAction, argparse._SubParsersAction)) and a.dest not in hidden
        ]
        if options:
            title = group.title or "options"
            groups.append({"title": "inputs" if title == "positional arguments" else title, "options": options})
    return {"description": parser.description or "", "groups": groups}


def load_parser(tool: Tool) -> argparse.ArgumentParser:
    module, _, function = tool.parser.partition(":")
    parser = getattr(importlib.import_module(module), function)()
    if tool.command is None:
        return parser
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices[tool.command]


def extract_all() -> dict[str, dict]:
    """Every tool's form spec; imports every tool, so run it in a subprocess."""
    return {key: describe(load_parser(tool), tool.hidden) for key, tool in TOOLS.items()}


# --- the cache ------------------------------------------------------------------------------


def source_stamp() -> str:
    """Changes whenever any source file of the package does."""
    package = PROJECT_ROOT / "birdcount"
    stamp = 0
    for path in sorted(package.rglob("*.py")):
        st = path.stat()
        stamp = zlib.crc32(f"{path.relative_to(package)}:{st.st_mtime_ns}:{st.st_size};".encode(), stamp)
    return f"{stamp:08x}"


class Specs:
    """Form specs of every tool, from the disk cache or one extraction subprocess."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._specs: dict[str, dict] | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def get(self, key: str) -> dict:
        if key not in TOOLS:
            raise KeyError(key)
        specs = self.load()
        return specs[key]

    def load(self) -> dict[str, dict]:
        with self._lock:
            if self._specs is None:
                self._specs = self._read_or_extract()
            return self._specs

    def _read_or_extract(self) -> dict[str, dict]:
        stamp = source_stamp()
        try:
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if cached.get("stamp") == stamp and set(cached["specs"]) == set(TOOLS):
                return cached["specs"]
        except (OSError, ValueError, KeyError):
            pass
        proc = subprocess.run(
            [sys.executable, "-m", __name__],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if proc.returncode:
            raise RuntimeError(f"could not read the tools' options:\n{(proc.stderr or proc.stdout)[-3000:]}")
        specs = json.loads(proc.stdout.strip().splitlines()[-1])
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps({"stamp": stamp, "specs": specs}), encoding="utf-8")
        except OSError:  # an unwritable cache still works for this session
            pass
        return specs


# --- form values -> command line --------------------------------------------------------------


def options(spec: dict) -> list[dict]:
    return [opt for group in spec["groups"] for opt in group["options"]]


def _items(opt: dict, value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(v).strip() for v in value]
    else:  # one entry per line, so paths with spaces survive; a single value has one line
        items = [line.strip() for line in str(value).splitlines()]
    return [item for item in items if item]


def _is_default(opt: dict, items: list[str]) -> bool:
    default = opt["default"]
    if default is None or default == []:
        return False
    defaults = [str(d) for d in default] if isinstance(default, list) else [str(default)]
    if opt["kind"] in ("int", "float"):
        try:
            return [float(i) for i in items] == [float(d) for d in defaults]
        except ValueError:
            return False
    return items == defaults


def _check(opt: dict, item: str) -> None:
    name = opt["flag"] or opt["dest"]
    if opt["kind"] == "int":
        try:
            int(item)
        except ValueError:
            raise ValueError(f"{name}: {item!r} is not a whole number") from None
    elif opt["kind"] == "float":
        try:
            float(item)
        except ValueError:
            raise ValueError(f"{name}: {item!r} is not a number") from None
    elif opt["kind"] == "choice" and item not in [str(c) for c in opt["choices"]]:
        raise ValueError(f"{name}: {item!r} is not one of {', '.join(map(str, opt['choices']))}")


def split_passthrough(text: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(text)
    # Windows paths keep their backslashes; quotes only group.
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(text, posix=False)]


PASSTHROUGH = "__passthrough"


def build_args(tool: Tool, spec: dict, values: dict) -> list[str]:
    """The arguments after `python -m <module>` for a form's values.

    Only options of the tool's own parser are accepted, and positionals go
    after "--", so no value can smuggle in a flag. Values equal to the default
    are left out, which keeps the command line to what was changed.
    """
    opts = options(spec)
    unknown = set(values) - {o["dest"] for o in opts} - {PASSTHROUGH}
    if unknown:
        raise ValueError(f"unknown option(s) for {tool.key}: {', '.join(sorted(unknown))}")
    flags, positionals, missing = [], [], []
    for opt in opts:
        value = values.get(opt["dest"])
        if opt["kind"] == "bool":
            if value is None or bool(value) == bool(opt["default"]):
                continue
            flag = opt["on"] if value else opt["off"]
            if flag:
                flags.append(flag)
            continue
        items = _items(opt, value)
        if not items:
            if opt["required"]:
                missing.append(opt["flag"] or opt["dest"])
            continue
        if not opt["multi"] and len(items) > 1:
            raise ValueError(f"{opt['flag'] or opt['dest']} takes one value")
        for item in items:
            _check(opt, item)
        if opt["positional"]:
            positionals += items
        elif _is_default(opt, items):
            continue
        elif opt["repeat"]:
            for item in items:
                flags += [opt["flag"], item]
        else:
            flags += [opt["flag"], *items]
    if missing:
        raise ValueError(f"missing required: {', '.join(missing)}")
    args = ([tool.command] if tool.command else []) + flags
    if positionals:
        args += ["--", *positionals]
    extra = str(values.get(PASSTHROUGH) or "").strip()
    if extra:
        if not tool.passthrough:
            raise ValueError(f"{tool.key} takes no extra arguments")
        args += ["--", *split_passthrough(extra)]
    return args


if __name__ == "__main__":
    print(json.dumps(extract_all()))
