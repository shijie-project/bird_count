"""(delivered frame, teacher density) pairs from recordings.

A *window* is `window_s` seconds of one recording. It is sampled at a stream
setting's fps, encoded the way that stream would be (H.264 at a bitrate cap and
GOP, optionally with everything outside the region mask black, as a camera
privacy mask would send it), decoded, and `per_window` frames of it are kept:

    pairs/<name>.png   the delivered frame times the region mask (BGR), as the model gets it
    pairs/<name>.npy   the teacher's masked density on the clean frame (float16, 1/8 resolution)
    index.csv          one row per pair: setting, the teacher's count and the teacher's
                       count on the delivered frame, and whether the frame is crowded

Encoders vary (NVENC and x264, two presets each) because the cameras' own
encoder cannot be run offline: a model robust to all of them should be robust
to it. Needs ffmpeg on PATH (NVENC settings need an NVIDIA GPU).
"""

from __future__ import annotations

import csv
import logging
import random
import re
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from birdcount.core.camera_identity import extract_mac


logger = logging.getLogger(__name__)

INDEX_FIELDS = (
    "file",
    "mac",
    "clip",
    "offset",
    "setting",
    "enc",
    "fps",
    "kbps_target",
    "kbps",
    "gop",
    "masked",
    "teacher",
    "teacher_on_delivered",
    "dense",
)
ENCODERS = ("nv_p1ll", "nv_p4", "x264_vf", "x264_med")
_DAY = re.compile(r"(20\d{2})(\d{2})(\d{2})_\d{6}")


@dataclass(frozen=True)
class StreamSetting:
    enc: str  # "clean" or one of ENCODERS
    fps: int
    kbps: int = 0
    gop: int = 0  # seconds between keyframes
    masked: bool = False  # black outside the region mask before encoding

    def __post_init__(self):
        if self.enc != "clean" and self.enc not in ENCODERS:
            raise ValueError(f"unknown encoder {self.enc!r}")
        if self.fps < 1 or (self.enc != "clean" and (self.kbps < 1 or self.gop < 1)):
            raise ValueError(f"incomplete stream setting {self}")

    @property
    def key(self) -> str:
        if self.enc == "clean":
            return f"clean {self.fps}fps"
        return f"{self.enc} {self.fps}fps {self.kbps}k g{self.gop}s{' masked' if self.masked else ''}"


# The held-out report's settings: the deployment's (3 fps, 400 kbps, a keyframe a
# minute) on two encoders, a harder 5 fps, and clean frames (nothing may drift there).
HOLDOUT_SETTINGS = (
    StreamSetting("nv_p1ll", 3, 400, 60),
    StreamSetting("x264_med", 5, 400, 60),
    StreamSetting("clean", 3),
)


def random_setting(rng: random.Random) -> StreamSetting:
    """A training stream: mostly what a thin link carries (2-5 fps, 250-500 kbps), sometimes clean."""
    if rng.random() < 0.10:
        return StreamSetting("clean", rng.choice([3, 4, 5]))
    return StreamSetting(
        enc=rng.choices(ENCODERS, [0.3, 0.2, 0.25, 0.25])[0],
        fps=rng.choices([2, 3, 4, 5], [0.1, 0.3, 0.25, 0.35])[0],
        kbps=rng.choice([250, 300, 350, 400, 450, 500]),
        gop=rng.choice([10, 30, 60]),
        masked=rng.random() < 0.15,
    )


def encoder_args(s: StreamSetting) -> list[str]:
    """ffmpeg output options for `s`: a bitrate cap with a one-second buffer, no B-frames (as the cameras)."""
    k, g = f"{s.kbps}k", str(s.gop * s.fps)
    if s.enc == "nv_p1ll":
        return ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "cbr", "-bf", "0", "-g", g,
                "-b:v", k, "-maxrate", k, "-bufsize", k]  # fmt: skip
    if s.enc == "nv_p4":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-bf", "0", "-g", g,
                "-b:v", k, "-maxrate", k, "-bufsize", f"{2 * s.kbps}k"]  # fmt: skip
    preset = "veryfast" if s.enc == "x264_vf" else "medium"
    return ["-c:v", "libx264", "-preset", preset, "-bf", "0", "-g", g, "-b:v", k, "-maxrate", k, "-bufsize", k]


# --------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    mac: str
    clip: Path
    offset: float  # seconds into the clip
    day: str = ""


def camera_days(paths: Iterable[str | Path]) -> set[tuple[str, str]]:
    """(MAC, YYYY-MM-DD) of every clip whose name or path carries both: the days to keep out of training."""
    out = set()
    for p in paths:
        mac, day = extract_mac(str(p)), _DAY.search(Path(p).name)
        if mac and day:
            out.add((mac, f"{day[1]}-{day[2]}-{day[3]}"))
    return out


def windows_from_replay(
    samples_dir: Path,
    thresholds: dict[str, float],
    *,
    exclude: set[tuple[str, str]] = frozenset(),
    rng: random.Random,
    dense_per_day: int = 3,
    random_per_day: int = 1,
    window_s: float = 60.0,
    dense_ratio: float = 0.6,
    exists: Callable[[Path], bool] = Path.exists,
) -> list[Window]:
    """Windows from a replay's samples (`samples/<MAC>/<day>.npz`: t, count, clip, clips).

    Per camera-day: up to `dense_per_day` windows whose mean count is at least
    `dense_ratio` x the camera's threshold, plus `random_per_day` with some birds.
    """
    out: list[Window] = []
    for f in sorted(Path(samples_dir).glob("*/*.npz")):
        mac, day = f.parent.name, f.stem
        if mac not in thresholds or (mac, day) in exclude:
            continue
        d = np.load(f, allow_pickle=True)
        t, count, clip_i, clips = d["t"], d["count"], d["clip"], d["clips"]
        cands: list[tuple[float, Path, float]] = []
        for ci in np.unique(clip_i):
            sel = np.flatnonzero(clip_i == ci)
            clip = Path(str(clips[ci]))
            if len(sel) < 2 or not exists(clip):
                continue
            rate = (len(sel) - 1) / max(t[sel[-1]] - t[sel[0]], 1e-6)
            step = max(1, round(window_s * rate))
            for s in range(0, len(sel) - step + 1, step):
                idx = sel[s : s + step]
                cands.append((float(count[idx].mean()), clip, float(t[idx[0]] - t[sel[0]])))
        if not cands:
            continue
        dense = [c for c in cands if c[0] >= dense_ratio * thresholds[mac]]
        rng.shuffle(dense)
        some = [c for c in cands if c[0] >= 5] or cands
        chosen = dense[:dense_per_day] + [rng.choice(some) for _ in range(random_per_day)]
        out += [Window(mac, clip, offset, day) for _, clip, offset in chosen]
    return out


# --------------------------------------------------------------------------------------
# Rendering a window as a stream
# --------------------------------------------------------------------------------------


def black_outside(frames: Sequence[np.ndarray], keep: np.ndarray) -> list[np.ndarray]:
    """yuv420p frames (flat) with everything outside `keep` black: a camera privacy mask."""
    h, w = keep.shape
    off_y = keep == 0
    off_c = (keep.reshape(h // 2, 2, w // 2, 2).max(axis=(1, 3)) == 0).ravel()
    out = []
    for f in frames:
        f = f.copy()
        f[: w * h].reshape(h, w)[off_y] = 16
        f[w * h : w * h * 5 // 4][off_c] = 128
        f[w * h * 5 // 4 :][off_c] = 128
        out.append(f)
    return out


def render_window(
    window: Window,
    setting: StreamSetting,
    frame_size: tuple[int, int],
    keep: np.ndarray,
    *,
    per_window: int,
    window_s: float,
    tmp: Path,
) -> tuple[list[np.ndarray], list[np.ndarray], int] | None:
    """(clean BGR frames, delivered BGR frames, measured kbps) for `per_window` frames of the window."""
    w, h = frame_size
    yuv = w * h * 3 // 2
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{window.offset:.2f}", "-t", str(window_s), "-i", str(window.clip),
         "-vf", f"fps={setting.fps},scale={w}:{h}", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
        capture_output=True, timeout=600,
    )  # fmt: skip
    n = len(r.stdout) // yuv
    if n < per_window + 2:
        return None
    clean = [np.frombuffer(r.stdout[i * yuv : (i + 1) * yuv], np.uint8) for i in range(n)]
    pick = np.linspace(1, n - 1, per_window).round().astype(int)
    clean_bgr = [cv2.cvtColor(clean[i].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420) for i in pick]
    if setting.enc == "clean":
        return clean_bgr, clean_bgr, 0
    src = black_outside(clean, keep) if setting.masked else clean
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        e = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
             "-r", str(setting.fps), "-i", "-", *encoder_args(setting), str(tmp)],
            input=b"".join(s.tobytes() for s in src), capture_output=True, timeout=600,
        )  # fmt: skip
        if e.returncode:
            logger.warning("Encoding %s failed: %s", setting.key, e.stderr.decode(errors="replace")[-300:])
            return None
        kbps = round(tmp.stat().st_size * 8 / (n / setting.fps) / 1e3)
        d = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(tmp), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=600,
        ).stdout  # fmt: skip
    finally:
        tmp.unlink(missing_ok=True)
    fs = w * h * 3
    if len(d) // fs < n:
        return None
    delivered = [np.frombuffer(d[i * fs : (i + 1) * fs], np.uint8).reshape(h, w, 3) for i in pick]
    return clean_bgr, delivered, kbps


# --------------------------------------------------------------------------------------
# Writing pairs
# --------------------------------------------------------------------------------------


class Teacher:
    """The model that sets the targets, with each camera's region mask; thread-safe."""

    def __init__(self, model, frame_size: tuple[int, int], mask_dir: Path | None):
        self.model, self.frame_size, self.mask_dir = model, frame_size, mask_dir
        self._predictors: dict[str, tuple[object, np.ndarray]] = {}
        self._lock = threading.Lock()

    def for_camera(self, mac: str):
        from birdcount.core.inference import DensityPredictor
        from birdcount.core.masks import RegionMasks

        with self._lock:
            if mac not in self._predictors:
                masks = RegionMasks.from_dir(self.mask_dir, [mac], self.frame_size, name=mac)
                pred = DensityPredictor(self.model, frame_size=self.frame_size, masks=masks, max_batch=16)
                keep = masks.keep[0] if masks is not None else np.ones(self.frame_size[::-1], np.uint8)
                self._predictors[mac] = (pred, keep)
            return self._predictors[mac]

    def count(self, mac: str, frames: list[np.ndarray], *, want_density: bool):
        pred, _ = self.for_camera(mac)
        with self._lock:
            return pred.predict(
                frames, [0] * len(frames) if pred.masks is not None else None, want_density=want_density
            )


def write_pairs(
    out: Path,
    jobs: Sequence[tuple[Window, StreamSetting]],
    teacher: Teacher,
    *,
    thresholds: dict[str, float],
    per_window: int = 10,
    window_s: float = 60.0,
    dense_ratio: float = 0.6,
    workers: int = 8,
    prefix: str = "w",
) -> int:
    """Render every job and write its pairs; appends to an existing index. Returns the pairs written."""
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    new = not index_path.exists()
    written = 0

    def run(item):
        k, (window, setting) = item
        _, keep = teacher.for_camera(window.mac)
        res = render_window(
            window, setting, teacher.frame_size, keep, per_window=per_window, window_s=window_s,
            tmp=out / "tmp" / f"{prefix}{k:05d}.mkv",
        )  # fmt: skip
        return k, window, setting, res

    with open(index_path, "a", newline="", encoding="utf-8") as fh, ThreadPoolExecutor(workers) as ex:
        wr = csv.writer(fh)
        if new:
            wr.writerow(INDEX_FIELDS)
        for done, (k, window, setting, res) in enumerate(ex.map(run, enumerate(jobs)), 1):
            if res is None:
                logger.warning(
                    "Skipped %s at %.0f s (%s): too short or not decodable.",
                    window.clip.name,
                    window.offset,
                    setting.key,
                )
                continue
            clean, delivered, kbps = res
            _, keep = teacher.for_camera(window.mac)
            t = teacher.count(window.mac, clean, want_density=True)
            s = teacher.count(window.mac, delivered, want_density=False)
            threshold = thresholds.get(window.mac)
            for i, frame in enumerate(delivered):
                name = f"{prefix}{k:05d}_{i}"
                cv2.imwrite(
                    str(out / "pairs" / f"{name}.png"), frame * keep[..., None], [cv2.IMWRITE_PNG_COMPRESSION, 1]
                )
                np.save(out / "pairs" / f"{name}.npy", t.density[i].astype(np.float16))
                dense = "" if threshold is None else int(t.counts[i] >= dense_ratio * threshold)
                wr.writerow([name, window.mac, window.clip.name, round(window.offset, 2), setting.key, setting.enc,
                             setting.fps, setting.kbps, kbps, setting.gop, int(setting.masked),
                             round(float(t.counts[i]), 2), round(float(s.counts[i]), 2), dense])  # fmt: skip
                written += 1
            fh.flush()
            if done % 25 == 0:
                logger.info("%d/%d windows, %d pairs.", done, len(jobs), written)
    return written
