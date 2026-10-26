"""Forms from argparse, and the command line a form becomes."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pytest

from birdcount.training.webui import tools
from birdcount.training.webui.tools import PASSTHROUGH, TOOLS, Specs, build_args, describe, load_parser


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="demo")
    p.add_argument("inputs", nargs="+", type=Path)
    g = p.add_argument_group("knobs")
    g.add_argument("--lr", type=float, default=1e-5, help="peak lr (default: 1e-05)")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--mode", choices=("a", "b"), default="a")
    g.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--no-cache", dest="cache", action="store_false")
    g.add_argument("--grid", action="append", default=[])
    g.add_argument("--sizes", type=int, nargs="+", default=[1280])
    return p


TOOL = tools.Tool("demo", "demo", "x:y", "Demo", "train", "")


def spec() -> dict:
    return describe(parser())


def test_options_become_typed_fields() -> None:
    by_dest = {o["dest"]: o for g in spec()["groups"] for o in g["options"]}
    assert [g["title"] for g in spec()["groups"]] == ["inputs", "knobs"]  # "options" held only --help
    assert by_dest["inputs"]["positional"] and by_dest["inputs"]["required"]
    assert by_dest["lr"]["kind"] == "float" and by_dest["lr"]["help"] == "peak lr"  # the default is the field's value
    assert by_dest["mode"]["kind"] == "choice" and by_dest["mode"]["choices"] == ["a", "b"]
    assert (by_dest["ema"]["on"], by_dest["ema"]["off"], by_dest["ema"]["default"]) == ("--ema", "--no-ema", True)
    assert (by_dest["apply"]["on"], by_dest["apply"]["off"]) == ("--apply", None)
    assert (by_dest["cache"]["on"], by_dest["cache"]["off"]) == (None, "--no-cache")
    assert by_dest["grid"]["repeat"] and by_dest["sizes"]["multi"] and by_dest["sizes"]["default"] == [1280]


def test_only_changed_values_reach_the_command_and_it_parses_back() -> None:
    values = {
        "inputs": "a.json\n--apply\nb c.json",  # a file named like a flag stays a file
        "lr": "1e-05",  # the default, spelled differently
        "epochs": "20",
        "mode": "a",
        "ema": False,
        "apply": True,
        "cache": False,
        "grid": "lr=1,2\nwot=3",
        "sizes": "1280",
    }
    args = build_args(TOOL, spec(), values)
    assert args == [
        "--epochs", "20", "--no-ema", "--apply", "--no-cache", "--grid", "lr=1,2", "--grid", "wot=3",
        "--", "a.json", "--apply", "b c.json",
    ]  # fmt: skip
    ns = parser().parse_args(args)
    assert ns.inputs == [Path("a.json"), Path("--apply"), Path("b c.json")]
    assert (ns.epochs, ns.ema, ns.apply, ns.cache, ns.grid, ns.lr) == (
        20,
        False,
        True,
        False,
        ["lr=1,2", "wot=3"],
        1e-5,
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"inputs": "a", "epochs": "ten"}, "not a whole number"),
        ({"inputs": "a", "lr": "fast"}, "not a number"),
        ({"inputs": "a", "mode": "c"}, "not one of"),
        ({"inputs": "a", "epochs": "1\n2"}, "takes one value"),
        ({}, "missing required"),
        ({"inputs": "a", "rm": "-rf"}, "unknown option"),
        ({"inputs": "a", PASSTHROUGH: "--x"}, "takes no extra"),
    ],
)
def test_bad_values_are_refused(values, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_args(TOOL, spec(), values)


def test_subcommands_and_passthrough_on_the_real_parsers() -> None:
    merge = TOOLS["labeling-merge"]
    args = build_args(merge, describe(load_parser(merge)), {"inputs": "x.json\ny.json", "output": "o.json"})
    assert args == ["merge", "--output", "o.json", "--", "x.json", "y.json"]
    root = load_parser(tools.Tool("l", "", f"{tools._LABELING}.__main__:build_parser", "", "", ""))
    assert root.parse_args(args).inputs == [Path("x.json"), Path("y.json")]

    sweep = TOOLS["sweep"]
    spec_ = describe(load_parser(sweep), sweep.hidden)
    assert "summary" not in {o["dest"] for g in spec_["groups"] for o in g["options"]}
    init = "C:\\w s\\a.pth" if os.name == "nt" else "/w s/a.pth"  # Windows paths keep their backslashes
    args = build_args(sweep, spec_, {"grid": "lr=1e-5,5e-5", PASSTHROUGH: f'--epochs 5 --init "{init}"'})
    assert args == ["--grid", "lr=1e-5,5e-5", "--", "--epochs", "5", "--init", init]


def test_specs_are_extracted_once_and_cached_on_disk(tmp_path, monkeypatch) -> None:
    fake = {key: {"description": "", "groups": []} for key in TOOLS}
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="noise\n" + json.dumps(fake), stderr="")

    monkeypatch.setattr(tools.subprocess, "run", run)
    cache = tmp_path / "cache.json"
    assert Specs(cache).get("train") == fake["train"]
    assert Specs(cache).get("eval") == fake["eval"] and len(calls) == 1  # a new panel reads the disk cache
    monkeypatch.setattr(tools, "source_stamp", lambda: "changed")
    Specs(cache).load()
    assert len(calls) == 2  # a source change re-extracts
    with pytest.raises(KeyError):
        Specs(cache).get("nope")


def test_extraction_reads_every_tool(tmp_path) -> None:
    specs = Specs(tmp_path / "cache.json").load()  # the real subprocess
    assert set(specs) == set(TOOLS)
    eval_opts = {o["dest"] for g in specs["eval"]["groups"] for o in g["options"]}
    assert {"split", "ckpt", "blobs", "grid", "output_dir"} <= eval_opts
