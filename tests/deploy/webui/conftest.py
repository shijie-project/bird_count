"""A small farm on disk, as the runtime leaves it: config, topology, status, count log, evidence."""

import csv
import http.client
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from birdcount.deploy.alarm import load_config
from birdcount.deploy.topology import parse_topology
from birdcount.deploy.webui.server import Backend, make_server


PLACED, LOOSE = "B8A44FD51C3C", "B8A44F5B4394"
DAY = "2026-10-25"
T0 = int(datetime(2026, 10, 25, 7, 0, 0).timestamp())
ROW_FIELDS = ["time_local", "timestamp", "frames", "count_mean", "count_max", "count_min",
              "centroid_x", "centroid_y", "alarm_level", "motion_excluded_frames", "snapshot"]  # fmt: skip


def write_rows(path: Path, rows: list[dict], header: bool = True) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if header:
            w.writeheader()
        w.writerows(rows)


def row(t: int, count: float, level: int = 0, snapshot: str = "") -> dict:
    return {"time_local": f"{datetime.fromtimestamp(t):%Y-%m-%d %H:%M:%S}", "timestamp": t, "frames": 5,
            "count_mean": f"{count:.2f}", "count_max": f"{count + 2:.2f}", "count_min": f"{count - 2:.2f}",
            "centroid_x": "0.5", "centroid_y": "0.5", "alarm_level": level, "motion_excluded_frames": 0,
            "snapshot": snapshot}  # fmt: skip


@pytest.fixture
def farm(tmp_path):
    cfg_path = tmp_path / "alarm.json"
    cfg_path.write_text(json.dumps({
        "output_dir": "evidence",
        "cameras": [{"camera_id": f"axis1/{PLACED}", "threshold": 100}, {"camera_id": f"axis2/{LOOSE}", "threshold": 120}],
    }), encoding="utf-8")  # fmt: skip
    config = load_config(cfg_path)
    topology = parse_topology({"zones": [
        {"name": "Zone_01_Right_Front", "cameras": [{"ip": "10.0.0.1", "mac": PLACED}, "10.0.0.2"], "speakers": ["10.0.0.9"]},
        {"name": "Zone_05_Right_Outer", "cameras": ["10.0.0.3"]},
    ]})  # fmt: skip

    day = tmp_path / "count_log" / DAY
    shots = day / "snapshots" / PLACED
    shots.mkdir(parents=True)
    for name in ("070010_bin_40.jpg", "070100_high_110.jpg"):
        (shots / name).write_bytes(b"\xff\xd8 jpeg")
    rows = [row(T0 + i, 40 if i < 60 else 110, level=1 if i >= 70 else 0) for i in range(120)]
    rows[10]["snapshot"] = f"snapshots/{PLACED}/070010_bin_40.jpg"
    rows[60]["snapshot"] = f"snapshots/{PLACED}/070100_high_110.jpg"
    write_rows(day / f"{PLACED}.csv", rows)
    (day / "meta.json").write_text(json.dumps({"thresholds": {PLACED: 100.0}, "hours": "06:00-20:00",
                                               "model_path": "C:/m/best.pth"}), encoding="utf-8")  # fmt: skip

    evidence = config.output_dir
    event_dir = evidence / "events" / f"axis1_{PLACED}" / "20261024T200110Z_abcd1234"
    event_dir.mkdir(parents=True)
    (event_dir / f"level1_{T0 + 70}.jpg").write_bytes(b"\xff\xd8 jpeg")
    action = {"action_type": "level1", "camera_id": f"axis1/{PLACED}", "event_id": "axis1_x", "timestamp": T0 + 70.2,
              "count": 110.0, "threshold": 100.0, "level": 1, "snapshot_path": str(event_dir / f"level1_{T0 + 70}.jpg")}  # fmt: skip
    (evidence / "events.jsonl").write_text(json.dumps(action) + "\n", encoding="utf-8")

    status = tmp_path / "live" / "status.json"
    status.parent.mkdir()
    status.write_text(json.dumps({"written": time.time(), "started": time.time() - 60, "stopped": None,
                                  "runtime": {"sms": "dry run"},
                                  "cameras": [{"sid": 0, "mac": PLACED, "count": 111.0, "level": 1, "online": True}]}),
                      encoding="utf-8")  # fmt: skip
    backend = Backend(
        config, topology, status_path=status, count_log_dir=tmp_path / "count_log", evidence_dir=evidence
    )
    backend.tmp = tmp_path
    return backend


@pytest.fixture
def served(farm):
    server = make_server(farm, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    farm.port = server.server_address[1]
    yield farm
    server.shutdown()
    server.server_close()


def request(backend, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", backend.port, timeout=30)
    hdrs = {"Host": f"127.0.0.1:{backend.port}"}
    raw = None
    if body is not None:
        raw = json.dumps(body).encode()
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    conn.request(method, path, body=raw, headers=hdrs)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data
