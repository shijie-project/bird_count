"""The panel's HTTP API on a real socket: guards, files, runs, forms, uploads, dataset."""

import http.client
import json
import threading

import pytest

from birdcount.core.paths import PROJECT_ROOT
from birdcount.training.webui import server as srv
from birdcount.training.webui import services
from birdcount.training.webui.settings import WebuiSettings
from birdcount.training.webui.tools import TOOLS, describe, load_parser


class InProcessSpecs:
    """Specs read in this process (the light tools only), instead of the extraction subprocess."""

    def get(self, key):
        if key not in TOOLS:
            raise KeyError(key)
        return describe(load_parser(TOOLS[key]), TOOLS[key].hidden)

    def load(self):
        return {}


@pytest.fixture
def panel(tmp_path, output_root, data_root):
    settings = WebuiSettings(
        _env_file=None,
        data_root=data_root,
        output_root=output_root,
        model_path=None,
        mask_dir=None,
        label_studio_api_key="",
        label_studio_public_url="",
    )
    p = srv.Panel(settings)
    p.specs = InProcessSpecs()
    server = srv.make_server(p, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    p.port = server.server_address[1]
    yield p
    server.shutdown()
    server.server_close()


def request(panel, method, path, body=None, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", panel.port, timeout=30)
    hdrs = {"Host": f"127.0.0.1:{panel.port}"}
    if body is not None:
        raw = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    conn.request(method, path, body=raw, headers=hdrs)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    try:
        return r.status, json.loads(data), r
    except ValueError:
        return r.status, data, r


def test_requests_from_elsewhere_are_refused(panel) -> None:
    assert request(panel, "GET", "/api/health")[0] == 200
    assert request(panel, "GET", "/api/health", headers={"Host": "evil.example:80"})[0] == 403  # DNS rebinding
    status, data, _ = request(panel, "POST", "/api/jobs/clear", {}, headers={"Origin": "https://evil.example"})
    assert status == 403 and "own page" in data["error"]
    assert request(panel, "POST", "/api/jobs/clear", {}, headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    # a cross-site form can only post text/plain: never accepted
    assert request(panel, "POST", "/api/shutdown", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    origin = {"Origin": f"http://127.0.0.1:{panel.port}"}
    assert request(panel, "POST", "/api/jobs/clear", {}, headers=origin)[1] == {"removed": 0}


def test_only_data_and_output_files_are_served(panel, data_root, output_root) -> None:
    ok = request(panel, "GET", f"/api/file?path={data_root / 'images' / 'val' / 'a.jpg'}")
    assert ok[0] == 200 and ok[2].getheader("Content-Type") == "image/jpeg"
    for path in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.example", PROJECT_ROOT / "birdcount" / "core" / "paths.py",
                 PROJECT_ROOT / "pyproject.toml", data_root.parent / "elsewhere.json"):  # fmt: skip
        assert request(panel, "GET", f"/api/file?path={path}")[0] == 403, path
    log = output_root / "train" / "20261001-100000-a" / "train.log"
    status, body, r = request(panel, "GET", f"/api/file?path={log}", headers={"Range": "bytes=2-6"})
    assert status == 206 and body == b"och 0" and r.getheader("Content-Range") == f"bytes 2-6/{log.stat().st_size}"


def test_runs_and_their_details(panel) -> None:
    items = request(panel, "GET", "/api/runs?page=train")[1]["items"]
    assert [(i["type"], i["kind"], i["ref"]) for i in items] == [
        ("run", "sweep", "train/sweep-20261003-100000"),
        ("run", "train", "train/20261002-100000-b"),
        ("run", "train", "train/20261001-100000-a"),
    ]
    d = request(panel, "GET", "/api/run?ref=train/20261001-100000-a&offset=0")[1]
    assert d["kind"] == "train" and len(d["records"]) == 4 and d["job"] is None
    again = request(panel, "GET", f"/api/run?ref=train/20261001-100000-a&offset={d['offset']}")[1]
    assert again["records"] == []
    assert request(panel, "GET", "/api/run?ref=train/sweep-20261003-100000/lr=1")[1]["kind"] == "train"
    for bad in ("../..", "train/nope", "C:/Windows"):
        assert request(panel, "GET", f"/api/run?ref={bad}")[0] == 404
    log = request(panel, "GET", "/api/log?ref=train/20261001-100000-a&offset=-1")[1]
    assert log["text"] == "epoch 0\nepoch 1\n"
    assert request(panel, "GET", "/api/runs?page=eval")[1]["items"][0]["counting_accuracy"] == 0.9


def test_the_form_preview_names_the_run_folder(panel, output_root) -> None:
    p = request(
        panel, "POST", "/api/preview", {"tool": "train", "values": {"name": "x", "epochs": "1000", "lr": "5e-5"}}
    )[1]
    assert p["command"].startswith("python -m birdcount.training.train --output-dir ")
    assert "--lr 5e-5" in p["command"] and "--epochs" not in p["command"]  # the default stays out
    assert p["run_dir"].startswith(str(output_root / "train")) and p["run_dir"].endswith("-x")
    resume = str(output_root / "train" / "20261002-100000-b")
    assert (
        request(panel, "POST", "/api/preview", {"tool": "train", "values": {"resume": resume}})[1]["run_dir"] == resume
    )
    merge = request(panel, "POST", "/api/preview", {"tool": "labeling-merge", "values": {"output": "o.json"}})[1]
    assert merge["command"] == "python -m birdcount.training.labeling merge --output o.json"
    assert (
        "missing required"
        in request(panel, "POST", "/api/preview", {"tool": "labeling-convert", "values": {}})[1]["error"]
    )
    assert request(panel, "POST", "/api/preview", {"tool": "rm", "values": {}})[0] == 404


def test_uploads(panel, output_root) -> None:
    status, data, _ = request(panel, "POST", "/api/upload?kind=json&name=my%20export.json", raw=b'[{"id": 1}]',
                              headers={"Content-Type": "application/octet-stream"})  # fmt: skip
    assert status == 200 and data["path"].endswith("-my_export.json")
    assert (PROJECT_ROOT / data["path"]).resolve().parent == (output_root / "webui" / "uploads").resolve()
    bad = request(
        panel,
        "POST",
        "/api/upload?kind=json&name=x.json",
        raw=b"{nope",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert bad[0] == 400 and len(list((output_root / "webui" / "uploads").iterdir())) == 1
    assert request(panel, "POST", "/api/upload?kind=json&name=evil.exe", raw=b"{}")[0] == 400
    assert request(panel, "POST", "/api/upload?kind=video&name=..%2F..%2Fx.mkv", raw=b"v")[1]["path"].endswith(
        "-x.mkv"
    )


def test_dataset_and_pickers(panel, data_root) -> None:
    assert request(panel, "GET", "/api/dataset")[1] == {"splits": [{"name": "val", "images": 2}]}
    assert len(request(panel, "GET", "/api/dataset/val")[1]["images"]) == 2
    assert request(panel, "GET", "/api/dataset/val/points?name=a")[1] == {"points": [[1.0, 2.0], [3.0, 4.0]]}
    assert request(panel, "GET", "/api/dataset/..")[0] == 404
    runs = request(panel, "GET", "/api/pickers/train_run")[1]["items"]
    assert [i["label"] for i in runs] == ["train/20261002-100000-b", "train/20261001-100000-a"]
    assert request(panel, "GET", "/api/pickers/split")[1]["items"][0]["value"] == "val"
    assert request(panel, "GET", "/api/pickers/nope")[0] == 404
    assert request(panel, "GET", "/api/pickers/ls_project")[1]["error"].startswith("LABEL_STUDIO_API_KEY")


def test_static_page(panel) -> None:
    status, body, r = request(panel, "GET", "/")
    assert status == 200 and b'type="module"' in body
    status, _, r = request(panel, "GET", "/static/js/app.js")
    assert status == 200 and r.getheader("Content-Type").startswith("text/javascript")
    assert (
        request(panel, "GET", "/static/js/app.js", headers={"If-Modified-Since": r.getheader("Last-Modified")})[0]
        == 304
    )
    assert request(panel, "GET", "/static/../server.py")[0] == 404


def test_label_studio_commands(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services.shutil, "which", lambda name: f"/bin/{name}")
    s = WebuiSettings(_env_file=None, data_root=tmp_path, label_studio_url="http://localhost:8091",
                      label_studio_public_url="https://name.ngrok-free.app/", label_studio_files_root=None)  # fmt: skip
    argv, env = services.label_studio_command(s, public=True)
    assert argv == ["/bin/label-studio", "start", "--data-dir", str(tmp_path), "-p", "8091", "--no-browser"]
    assert env["LABEL_STUDIO_HOST"] == env["CSRF_TRUSTED_ORIGINS"] == "https://name.ngrok-free.app"
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == str(tmp_path)
    assert services.ngrok_command(s)[0][:4] == ["/bin/ngrok", "http", "--url=name.ngrok-free.app", "8091"]
    bare = WebuiSettings(_env_file=None, label_studio_public_url="")
    with pytest.raises(RuntimeError, match="LABEL_STUDIO_PUBLIC_URL"):
        services.ngrok_command(bare)
    assert "LABEL_STUDIO_HOST" not in services.label_studio_command(bare, public=False)[1]
