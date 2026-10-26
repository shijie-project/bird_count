"""The labelling commands end to end on a tiny DATA_ROOT, Label Studio faked."""

import json

import cv2
import httpx
import numpy as np

from birdcount.training.labeling import __main__ as cli
from birdcount.training.labeling import client
from birdcount.training.labeling.client import LabelStudio
from birdcount.training.labeling.ls_format import keypoint, local_files_url

from .conftest import REFRESH


MACS = ("B8A44FD51C3C", "B8A44FD52B7F")


def make_root(tmp_path, monkeypatch, fake=None):
    data = tmp_path / "data"
    (data / "images" / "all").mkdir(parents=True)
    (data / "images" / "masks").mkdir()
    for mac in MACS:
        mask = np.full((64, 96, 3), 255, np.uint8)
        mask[:, 48:] = 0
        cv2.imwrite(str(data / "images" / "masks" / f"{mac}.png"), mask)
    env = {
        "DATA_ROOT": data,
        "MASK_DIR": data / "images" / "masks",
        "LABEL_STUDIO_API_KEY": REFRESH,
        "LABEL_STUDIO_PROJECT_ID": "1",
        "LABEL_STUDIO_FILES_ROOT": "",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    if fake is not None:
        monkeypatch.setattr(
            client,
            "LabelStudio",
            lambda url, token: LabelStudio(url, token, transport=httpx.MockTransport(fake.handler)),
        )
    return data


def export(data, n_per_camera: int = 3) -> list[dict]:
    tasks = []
    for mac in MACS:
        for k in range(n_per_camera):
            name = f"axis1_{mac}_2026031{k}_101010_AB12_frame{k}.png"
            image = np.full((64, 96, 3), 90, np.uint8)
            image[:, 48:] = 0
            cv2.imwrite(str(data / "images" / "all" / name), image)
            points = [keypoint(10, 10, (96, 64), label="chicken"), keypoint(70, 10, (96, 64), label="chicken")]
            tasks.append(
                {"data": {"img": local_files_url(name, "images\\all\\")}, "annotations": [{"result": points}]}
            )
    return tasks


def test_file_pipeline(tmp_path, monkeypatch, capsys) -> None:
    data = make_root(tmp_path, monkeypatch)
    ex = tmp_path / "export.json"
    ex.write_text(json.dumps(export(data)), encoding="utf-8")
    all_json = data / "annotations" / "all.json"

    assert (
        cli.main(["merge", str(ex), "-o", str(tmp_path / "merged.json"), "--images-dir", str(data / "images" / "all")])
        == 0
    )
    assert cli.main(["convert", str(tmp_path / "merged.json"), "-o", str(all_json)]) == 0
    assert len(json.loads(all_json.read_text())["images"]) == 6

    assert cli.main(["drop-masked"]) == 0
    clean = json.loads((data / "annotations" / "all_clean.json").read_text())
    assert {len(a["points"]) for a in clean["annotations"]} == {1}  # the point in the black half is gone

    assert cli.main(["split", "--val-ratio", "0.34"]) == 0
    train = json.loads((data / "annotations" / "train.json").read_text())
    val = json.loads((data / "annotations" / "val.json").read_text())
    assert len(train["images"]) + len(val["images"]) == 6 and val["images"]
    assert sorted(p.name for p in (data / "images" / "val").iterdir()) == sorted(
        im["file_name"] for im in val["images"]
    )

    assert cli.main(["to-ls", "-o", str(tmp_path / "tasks.json")]) == 0
    tasks = json.loads((tmp_path / "tasks.json").read_text())
    assert tasks[0]["data"]["img"].startswith("/data/local-files/?d=images%5Call%5C")

    assert cli.main(["apply-mask", str(data / "images" / "all"), "--out", str(tmp_path / "masked")]) == 0
    assert len(list((tmp_path / "masked").iterdir())) == 6
    assert cli.main(["dedupe-images"]) == 0
    assert "Dry run" in capsys.readouterr().out


def test_live_commands(tmp_path, monkeypatch, fake, capsys) -> None:
    data = make_root(tmp_path, monkeypatch, fake)
    for task in export(data, 1):
        fake.add_task(task["data"]["img"], task["annotations"] * 2)
    out = tmp_path / "pulled.json"
    assert cli.main(["pull", "-o", str(out)]) == 0
    assert len(json.loads(out.read_text())) == 2
    assert cli.main(["dedupe-annotations"]) == 0
    assert "2 annotation(s) WOULD be deleted" in capsys.readouterr().out
    assert len(fake.annotations) == 4
    assert cli.main(["dedupe-annotations", "--apply"]) == 0
    assert len(fake.annotations) == 2

    monkeypatch.setenv("LABEL_STUDIO_PROJECT_ID", "")
    assert cli.main(["dedupe-annotations", "--project-id", "7"]) == 1  # no such project
    assert "HTTP 404" in capsys.readouterr().err


def test_json_is_written_exactly_as_json_dump_would() -> None:
    import io

    payloads = [
        {"images": [{"id": 1, "file_name": "ä.jpg"}], "annotations": [], "meta": {}, "n": [1, [2, {"a": "x\ny"}]]},
        [{"data": {"img": "/a"}, "annotations": [{"result": [{"value": {"x": 1.5}}]}]}, [], {}, 3, None],
        [],
        "plain",
    ]
    for payload in payloads:
        for indent in (None, 2, 4):
            f = io.StringIO()
            cli.write_json_items(f, payload, indent)
            assert f.getvalue() == json.dumps(payload, indent=indent, ensure_ascii=False)
