r"""Density over time for one video: how crowded the pen gets, minute by minute.

The evaluation ops in this folder score still images against human points. This
one answers a different question — *when* does the flock crowd up — by running
the density model over a video and reporting two numbers per sampled frame:

  count  the whole map integrated: how many birds are in the frame
  peak   the busiest `--window` x `--window` patch of the map, integrated: how
         many birds are standing inside one small area. This is the pile-up
         signal; the frame count can stay flat while birds bunch into a corner.

Both come straight from the density map, so they are absolute (`birds`), not a
per-frame normalization — two clips, or two hours of one clip, are comparable.
`--report` chooses which of the two the figure and the summary present — the
whole-frame density, the local one, or both (the default); the CSV and the JSON
always carry both, so changing your mind is a re-plot, not a re-run.

Artifacts land in `--output-dir`: `timeline.csv` and `timeline.json` for further
analysis, `timeline.png` as a publication-ready figure, and a few overlay JPGs
of the busiest moments. `--save-frames` keeps every sampled frame as well, in
`frames/`, which is how you get one picture per second of recording. Overlays
carry a caption sized for the saved image and, with `--mask-image`, a red frame
around the region the model was shown. The web UI
parses the SAMPLE lines below and draws its own interactive chart while the run
is still going.

Usage:
    # one sample per second of a clip, clock axis starting at 04:00
    python webui/ops/video_density_timeline.py --video ../data/demo.mkv --start-time 04:00

    # keep one overlay picture per second, at half size
    python webui/ops/video_density_timeline.py --video ../data/demo.mkv \
        --sample-seconds 1 --save-frames overlay --frame-width 960

    # a long recording: sample every 10s, ignore the corridor, and mark the
    # alarm's own 150-birds-in-frame level on the count series
    python webui/ops/video_density_timeline.py --video ../data/day.mkv \
        --sample-seconds 10 --mask-image ../data/image_mask.png \
        --threshold 150 --threshold-metric count

The checkpoint comes from `--ckpt`, else MODEL_PATH in .env (same as test.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import dotenv
import numpy as np
import torch
import torch.nn.functional as F


# Run as a plain script and sys.path[0] is this folder, so the project's own
# packages (models, utils) would not be importable. Put the root first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from alarm.camera_ids import resolve_camera_id  # noqa: E402
from models.shufflenet import get_shufflenet_density_model  # noqa: E402
from runtime.camera_identity import extract_mac  # noqa: E402
from utils import HEATMAP_VMAX, density_to_heatmap  # noqa: E402


dotenv.load_dotenv()
warnings.simplefilter("ignore", UserWarning)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Accepted spellings for --start-time, most specific first. A bare clock time is
# the common case (the recording's wall-clock start); a full date is there for
# clips that span midnight.
_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%H:%M:%S",
    "%H:%M",
)

# Overlay blend weights — same as test.py / inference_video.py, so the peak
# frames read like every other density picture in this project.
HEATMAP_ALPHA_BG = 0.5
HEATMAP_ALPHA_FG = 0.5
JPEG_QUALITY = 92

# BGR. Red for the mask outline, matching tools/inference_video.py's alert
# contour; white text with a black outline reads on both the pen floor and a
# red-hot patch of heatmap.
MASK_REGION_COLOR = (0, 0, 255)
CAPTION_COLOR = (255, 255, 255)
PEAK_BOX_COLOR = (255, 255, 255)


def frame_filename(index: int, seconds: float) -> str:
    """Name of one sampled frame inside `frames/`.

    Index first so the folder sorts in recording order, then the whole second it
    was taken at, so a moment can be found without opening the CSV. The web UI
    rebuilds this name to open the frame behind a point on the chart, so the two
    spellings have to stay in step (see `framePath` in static/app.js).
    """
    return f"frame_{index:05d}_{int(seconds):06d}s.jpg"


def blend_density(original: np.ndarray, density: np.ndarray, vmax: float, black_mask=None) -> np.ndarray:
    """The density map painted over a copy of the frame, at the frame's size."""
    height, width = original.shape[:2]
    upsampled = cv2.resize(np.asarray(density, dtype=np.float32), (width, height))
    heatmap, heat_mask = density_to_heatmap(upsampled, vmax=vmax)
    overlay = original.copy()
    heat = heat_mask.astype(bool)
    if black_mask is not None:  # a masked-out region carries no density worth showing
        heat &= ~black_mask
    if heat.any():
        overlay[heat] = cv2.addWeighted(original[heat], HEATMAP_ALPHA_BG, heatmap[heat], HEATMAP_ALPHA_FG, 0)
    return overlay


def draw_mask_region(image: np.ndarray, contours, color=MASK_REGION_COLOR) -> None:
    """Outline the area the model actually looked at.

    Same red frame `tools/inference_video.py` draws: with a mask in play, a
    picture that does not show where the mask ends invites the reader to count
    birds the model was never shown.
    """
    if contours is None or len(contours) == 0:
        return
    cv2.drawContours(image, contours, -1, color, thickness=max(3, round(image.shape[1] / 500)))


def draw_caption(image: np.ndarray, text: str, corner: str = "top-left") -> None:
    """Caption at the top of the image, outlined so it survives any background.

    Sized against the image it is drawn on (~1/45th of the width per character
    row), so it stays readable in a thumbnail — which is why the caption goes on
    *after* `save_jpeg` has downscaled, not before.
    """
    scale = max(0.8, image.shape[1] / 960)
    thickness = max(2, int(2 * scale))
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    margin = max(10, int(9 * scale))
    x = image.shape[1] - text_w - margin if corner == "top-right" else margin
    origin = (max(margin, x), text_h + margin)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(5 * scale), cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, CAPTION_COLOR, thickness, cv2.LINE_AA)


# Decimals per series in a caption: a bird count is a whole number, while the
# peak is a density integral where the first decimal carries real information.
_METRIC_DECIMALS = {"count": 0, "peak": 1}


def metric_text(sample: dict, key: str, threshold: float = 0.0, threshold_metric: str = "") -> str:
    """One series for a caption, carrying its threshold when it owns one.

    The threshold rides next to the number it judges rather than sitting alone
    in a corner: `count 173 / thr 150 OVER` answers "is this frame an alarm?" at
    a glance, where a bare `thr 150` would leave the reader comparing by eye.
    """
    decimals = _METRIC_DECIMALS[key]
    text = f"{key} {sample[key]:.{decimals}f}"
    if threshold > 0 and threshold_metric == key:
        text += f" / thr {threshold:.{decimals}f}"
        if sample[key] >= threshold:
            text += " OVER"
    return text


def save_jpeg(path: Path, image: np.ndarray, width: int = 0, caption: str = "", corner: str = "top-left") -> int:
    """Write `image`, optionally downscaled to `width`. Returns the bytes written.

    The caption is drawn here, after the resize, so its size is set by the image
    that is actually saved rather than by the frame it came from.
    """
    if width and image.shape[1] != width:
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    elif caption:
        image = image.copy()  # never caption the caller's frame in place
    if caption:
        draw_caption(image, caption, corner)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    try:
        return path.stat().st_size
    except OSError:
        return 0


def series_panels(meta: dict) -> list[tuple[str, str, str, str]]:
    """(title, sample key, unit, colour) for the series this run reports.

    `meta["report"]` selects them: the whole-frame density, the local density in
    the busiest window, or both. The colours are the web UI's, so the exported
    figure and the page on screen read as the same chart.
    """
    panels = [
        ("Max local density", "peak", f"birds per {meta['window_px']}×{meta['window_px']} px patch", "#c2410c"),
        ("Flock count", "count", "birds in frame", "#2563eb"),
    ]
    report = meta.get("report", "both")
    return [panel for panel in panels if report in ("both", panel[1])]


def threshold_from_alarm_config(video: str, config_path: Path) -> tuple[float, str]:
    """The deployed pile-up threshold for the camera this clip came from.

    Matches on the bare MAC in the filename (`B8A44FD51C3C.mkv`), the same
    identity the live runtime uses — see `runtime.camera_identity`. Returns
    `(0.0, reason)` when there is nothing to match, so the caller can say why
    rather than silently charting no threshold.

    These are whole-frame counts, so they belong to the `count` series.
    """
    mac = extract_mac(Path(video).stem) or extract_mac(video)
    if not mac:
        return 0.0, f"no MAC in the path {Path(video).name!r}"
    if not config_path.is_file():
        return 0.0, f"{config_path} not found"
    try:
        cameras = json.loads(config_path.read_text(encoding="utf-8")).get("cameras", [])
    except Exception as e:
        return 0.0, f"could not read {config_path}: {e}"

    by_id = {str(c["camera_id"]): c for c in cameras}
    camera_id = resolve_camera_id(mac=mac, known=by_id)
    if camera_id is None:
        return 0.0, f"{mac} is not in {config_path.name}"
    threshold = float(by_id[camera_id].get("threshold", 0.0))
    return threshold, camera_id


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chart predicted flock count and peak local density over the length of a video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("input")
    g.add_argument("--video", required=True, help="video file to analyse (mp4/mkv/avi/mov/...)")
    g.add_argument(
        "--mask-image",
        default=None,
        help="mask image the size of the frame; black pixels are blanked before inference so fixed "
        "structures outside the pen cannot contribute density",
    )
    g.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="seconds of video between two sampled frames; raise it for long recordings",
    )
    g.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="only analyse the first N seconds of the video (0 = the whole file)",
    )
    g.add_argument(
        "--start-time",
        default="",
        help="wall-clock time of the first frame ('16:20', '2026-08-28 16:20:00'). Given one, the chart's "
        "x axis is clock time like the recording; left empty it is elapsed time",
    )
    g.add_argument(
        "--frame-size",
        default="",
        help="resize every frame to WIDTHxHEIGHT before inference (e.g. 1080x720, what the live runtime "
        "feeds the model). Empty keeps the video's own resolution",
    )
    g.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="upper bound on sampled frames; --sample-seconds is stretched to respect it, so pointing this "
        "at an all-day recording cannot produce a million-point chart",
    )

    g = p.add_argument_group("model")
    g.add_argument("--ckpt", default=None, help="checkpoint path; defaults to MODEL_PATH from .env")
    g.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    g.add_argument("--batch-size", type=int, default=16, help="frames per forward pass")
    g.add_argument(
        "--no-fuse",
        action="store_true",
        help="skip Conv+BN fusion (debug only; fusion is mathematically equivalent)",
    )
    g.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")

    g = p.add_argument_group("density")
    g.add_argument(
        "--window",
        type=int,
        default=16,
        help="side of the sliding window, in density cells (one cell = 8 source pixels), whose integral is "
        "reported as the peak local density. The default 16 cells is a 128x128 px patch — about one bird "
        "across on a pen camera, so the number reads as 'birds standing on top of each other'",
    )
    g.add_argument(
        "--report",
        default="both",
        choices=["both", "count", "peak"],
        help="which series the figure and the summary report: 'count' is the whole-frame density (every bird "
        "in view), 'peak' is the local density inside the busiest --window patch, 'both' gives one panel each. "
        "Either way every sample of both series is written to the CSV and the JSON, so this is a presentation "
        "choice and not a re-run",
    )
    g.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="alert level drawn across the chart, in birds (0 = no line)",
    )
    g.add_argument(
        "--alarm-config",
        default="configs/alarm.json",
        help="pile-up alarm config to take the threshold from when --threshold is not given. The camera is "
        "matched on the bare MAC in the video filename; its threshold is a whole-frame count, so it lands on "
        "the 'count' series. Pass --threshold to override, or point this at a missing file to disable",
    )
    g.add_argument(
        "--threshold-metric",
        default="peak",
        choices=["peak", "count"],
        help="which series --threshold belongs to: the peak local density, or the whole-frame count the "
        "deployed alarm uses. It is only drawn on that one, since the two have different scales",
    )
    g.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="samples in the centered rolling mean drawn over the raw series (0 or 1 = no smoothing)",
    )
    g.add_argument("--vmax", type=float, default=HEATMAP_VMAX, help="density painted as full-scale red")

    g = p.add_argument_group("output")
    g.add_argument(
        "--output-dir",
        default=None,
        help="artifact directory; defaults to outputs/video_density/<video name>",
    )
    g.add_argument(
        "--peak-frames",
        type=int,
        default=4,
        help="busiest moments saved as density overlay JPGs (0 = none)",
    )
    g.add_argument(
        "--peak-gap-seconds",
        type=float,
        default=15.0,
        help="minimum spacing between two saved peak frames, so they are not four shots of one event",
    )
    g.add_argument(
        "--save-frames",
        default="none",
        choices=["none", "overlay", "plain"],
        help="also save every sampled frame, not only the busiest ones: 'overlay' blends the density map over "
        "it and captions it, like the peak images; 'plain' saves the frame untouched, which is what you would "
        "hand to an annotator. They go to <output-dir>/frames/, one per sample — at --sample-seconds 1 that is "
        "3600 images per hour of video, so watch the disk (and see --frame-width)",
    )
    g.add_argument(
        "--caption-corner",
        default="top-left",
        choices=["top-left", "top-right"],
        help="where the time/count caption sits on a saved overlay",
    )
    g.add_argument(
        "--frame-width",
        type=int,
        default=0,
        help="downscale saved frames to this width, keeping the aspect ratio (0 = the size they were analysed "
        "at). Affects the saved images only, never the inference",
    )
    g.add_argument("--no-chart", action="store_true", help="skip the rendered timeline.png figure")
    return p


def resolve_checkpoint(args: argparse.Namespace) -> str:
    path = args.ckpt or os.getenv("MODEL_PATH")
    if not path:
        raise SystemExit("No checkpoint specified. Pass --ckpt or set MODEL_PATH in .env.")
    if not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    return path


def parse_start_time(text: str) -> datetime | None:
    """'16:20' / '2026-08-28 16:20:00' -> datetime, or None when not given."""
    text = (text or "").strip()
    if not text:
        return None
    for fmt in _TIME_FORMATS:
        try:
            stamp = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt:  # a bare clock time: date is irrelevant, only the time of day shows
            stamp = stamp.replace(year=2000, month=1, day=1)
        return stamp
    raise SystemExit(f"--start-time expects HH:MM, HH:MM:SS or 'YYYY-MM-DD HH:MM:SS' (got {text!r})")


def parse_frame_size(spec: str) -> tuple[int, int] | None:
    """'1080x720' -> (1080, 720). None when no resize was requested."""
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        width, height = (int(v) for v in spec.lower().split("x", 1))
    except ValueError:
        raise SystemExit(f"--frame-size expects WIDTHxHEIGHT, e.g. '1080x720' (got {spec!r})") from None
    if width < 32 or height < 32:
        raise SystemExit(f"--frame-size is too small to run the model on (got {spec!r})")
    return width, height


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------


def preprocess_batch(frames_bgr: list[np.ndarray], device: torch.device, mean, std) -> torch.Tensor:
    """List of HxWx3 uint8 BGR frames -> normalized (B, 3, H, W) tensor on device."""
    arr = np.stack(frames_bgr, axis=0)
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    t = t[..., [2, 1, 0]]  # BGR -> RGB
    t = t.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    t.sub_(mean).div_(std)
    return t


def window_peaks(maps: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Busiest window per map: (B,) integral and (B,) flat index of its top-left cell.

    `avg_pool2d * window^2` is the sum over every window position, which is the
    number of birds standing inside that patch — the same quantity as the frame
    count, just restricted to a small area, so the two series share a unit.
    """
    height, width = maps.shape[-2:]
    size = max(1, min(window, height, width))
    sums = F.avg_pool2d(maps, kernel_size=size, stride=1) * (size * size)
    flat = sums.flatten(1)
    values, indices = flat.max(dim=1)
    return values, indices


def peak_box(index: int, map_shape: tuple[int, int], window: int, frame_w: int, frame_h: int) -> tuple[int, ...]:
    """Flat window index on the density grid -> (x0, y0, x1, y1) in frame pixels."""
    map_h, map_w = map_shape
    size = max(1, min(window, map_h, map_w))
    positions_w = map_w - size + 1
    row, col = divmod(int(index), max(positions_w, 1))
    scale_x, scale_y = frame_w / map_w, frame_h / map_h
    x0, y0 = int(col * scale_x), int(row * scale_y)
    x1, y1 = int((col + size) * scale_x), int((row + size) * scale_y)
    return x0, y0, x1, y1


def load_static_mask(mask_path: str, target_w: int, target_h: int) -> tuple[np.ndarray, tuple]:
    """Read the mask image.

    Returns (region, contours): `region` is True where the frame must be blanked,
    `contours` outline the area that survives — the red frame drawn on every
    saved overlay, so a reader can see what the model was shown.
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_img is None:
        raise SystemExit(f"Cannot open mask image: {mask_path}")
    mask_h, mask_w = mask_img.shape[:2]
    if (mask_w, mask_h) != (target_w, target_h):
        # Nearest keeps the mask binary; a mask drawn at the video's native size
        # is still usable after --frame-size shrinks the frames.
        mask_img = cv2.resize(mask_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    region = np.all(mask_img < 10, axis=2)
    contours, _ = cv2.findContours((~region).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return region, tuple(contours)


def sample_video(
    args, model, device, black_mask, mask_contours, target_size, output_dir, start
) -> tuple[list[dict], dict]:
    """Run the model over the sampled frames and return (samples, video metadata).

    With --save-frames the images are written here, inside the pass that
    already holds the frame and its density map: a second pass would decode
    the whole video again to produce pictures we have in hand.
    """
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.seconds > 0:
        wanted = max(1, int(args.seconds * fps))
        total_frames = min(total_frames, wanted) if total_frames > 0 else wanted
    duration = total_frames / fps if total_frames > 0 else 0.0

    interval = max(args.sample_seconds, 1.0 / fps)
    # A day-long recording at one sample per second would be 86k points: chart
    # noise, a slow page, and nothing the eye can read. Stretch the interval
    # instead of silently truncating the video.
    if args.max_samples > 0 and duration > 0 and duration / interval > args.max_samples:
        interval = duration / args.max_samples
        print(f"[note] --max-samples {args.max_samples} raised the sampling interval to {interval:.2f}s")
    step = max(1, int(round(fps * interval)))
    interval = step / fps

    infer_w, infer_h = target_size if target_size else (width, height)
    print(f"Video: {args.video}")
    print(f"  {width}x{height} @ {fps:.2f} fps · {total_frames or '?'} frames · {duration:.1f}s")
    print(f"  inference at {infer_w}x{infer_h}, one sample every {interval:.2f}s (every {step} frames)")
    print(f"  peak window: {args.window}x{args.window} density cells")

    frames_dir = None
    if args.save_frames != "none":
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        expected = max(1, total_frames // step) if total_frames > 0 else 0
        size_note = f"{expected} images" if expected else "one image per sample"
        width_note = f"{args.frame_width}px wide" if args.frame_width else "full size"
        print(f"  saving {args.save_frames} frames to {frames_dir} ({size_note}, {width_note})")
    print("-" * 78)

    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    use_amp = (not args.no_amp) and device.type == "cuda"

    samples: list[dict] = []
    batch_frames: list[np.ndarray] = []
    batch_meta: list[tuple[int, float]] = []
    # The frame as decoded, kept only while frames are being saved: the model
    # is fed the masked copy, but a saved picture should show the real scene.
    batch_original: list[np.ndarray] = []
    written = {"count": 0, "bytes": 0}
    # The model's stride is whatever the network emits, so read it off the first
    # output rather than assuming the training-time downsample ratio.
    grid: dict[str, float] = {}

    def flush() -> None:
        if not batch_frames:
            return
        inputs = preprocess_batch(batch_frames, device, mean, std)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
        maps = outputs.float()
        counts = maps.sum(dim=(1, 2, 3)).cpu().numpy()
        peaks, indices = window_peaks(maps, args.window)
        peaks = peaks.cpu().numpy()
        indices = indices.cpu().numpy()
        map_shape = (maps.shape[-2], maps.shape[-1])
        grid.setdefault("cell_px", infer_w / map_shape[1])

        for i, (frame_index, seconds) in enumerate(batch_meta):
            sample = {
                "i": len(samples),
                "frame": frame_index,
                "t": round(float(seconds), 3),
                "count": round(float(counts[i]), 3),
                "peak": round(float(peaks[i]), 3),
                "box": peak_box(indices[i], map_shape, args.window, infer_w, infer_h),
            }
            samples.append(sample)

            if frames_dir is not None:
                original = batch_original[i]
                caption = ""
                if args.save_frames == "plain":
                    # Nothing drawn on it at all: a plain frame is meant to be
                    # annotated, and the name already carries index and second.
                    image = original
                else:
                    image = blend_density(original, maps[i, 0].cpu().numpy(), args.vmax, black_mask)
                    draw_mask_region(image, mask_contours)
                    caption = "  ".join(
                        (
                            clock_of(start, sample["t"]),
                            metric_text(sample, "count", args.threshold, args.threshold_metric),
                            metric_text(sample, "peak", args.threshold, args.threshold_metric),
                        )
                    )
                written["bytes"] += save_jpeg(
                    frames_dir / frame_filename(sample["i"], sample["t"]),
                    image,
                    args.frame_width,
                    caption,
                    args.caption_corner,
                )
                written["count"] += 1
                # One early estimate beats discovering the disk is full an hour in.
                if written["count"] == 1 and total_frames > 0:
                    each = written["bytes"] / 1e6
                    print(f"  [note] ~{each:.1f} MB per saved frame, ~{each * (total_frames // step):.0f} MB in total")
            # One line per sample, flushed as it goes: this is what the web UI
            # parses to grow its chart while the run is still in flight.
            print(
                f"  SAMPLE {sample['i']} | t {sample['t']:.3f} | count {sample['count']:.2f} | peak {sample['peak']:.3f}"
            )
        batch_frames.clear()
        batch_meta.clear()
        batch_original.clear()

    frame_index = 0
    try:
        while True:
            if total_frames > 0 and frame_index >= total_frames:
                break
            if frame_index % step:
                # Decoding a frame nobody looks at is the bulk of the cost on a
                # long recording; grab() skips it.
                if not capture.grab():
                    break
                frame_index += 1
                continue

            ok, frame = capture.read()
            if not ok:
                break
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            if frames_dir is not None:
                batch_original.append(frame)
            if black_mask is not None:
                frame = frame.copy()
                frame[black_mask] = 0

            batch_frames.append(frame)
            batch_meta.append((frame_index, frame_index / fps))
            if len(batch_frames) >= args.batch_size:
                flush()
            frame_index += 1
        flush()
    finally:
        capture.release()

    meta = {
        "video": os.path.abspath(args.video),
        "name": Path(args.video).name,
        "width": width,
        "height": height,
        "infer_width": infer_w,
        "infer_height": infer_h,
        "fps": round(float(fps), 3),
        "frames": frame_index,
        "duration": round(frame_index / fps, 3),
        "interval": round(interval, 4),
        "window": args.window,
        "cell_px": round(grid.get("cell_px", 8.0), 2),
        "window_px": int(round(args.window * grid.get("cell_px", 8.0))),
        "report": args.report,
        "threshold": args.threshold,
        "threshold_metric": args.threshold_metric,
        "smooth": args.smooth,
        "mask": os.path.abspath(args.mask_image) if args.mask_image else None,
        # The web UI rebuilds a frame's path from this directory plus
        # frame_filename(), so a point on the chart can open its own picture.
        "frames_dir": str(frames_dir) if frames_dir is not None else None,
        "frames_style": args.save_frames,
        "frames_saved": written["count"],
    }
    if frames_dir is not None:
        print(f"Saved {written['count']} frames ({written['bytes'] / 1e6:.0f} MB) to {frames_dir}")
    return samples, meta


# ----------------------------------------------------------------------------
# Artifacts
# ----------------------------------------------------------------------------


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Centered rolling mean, shrinking at the edges so the line spans the axis."""
    if window <= 1 or not values:
        return list(values)
    arr = np.asarray(values, dtype=np.float64)
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")[: len(arr)].tolist()


def clock_of(start: datetime | None, seconds: float) -> str:
    """Wall-clock stamp for a sample, or elapsed h:mm:ss when no start is known."""
    if start is not None:
        return (start + timedelta(seconds=seconds)).strftime("%H:%M:%S")
    total = int(round(seconds))
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def summarize(samples: list[dict], meta: dict, start: datetime | None) -> dict:
    counts = np.asarray([s["count"] for s in samples], dtype=np.float64)
    peaks = np.asarray([s["peak"] for s in samples], dtype=np.float64)
    peak_at = samples[int(peaks.argmax())]
    busiest = samples[int(counts.argmax())]
    return {
        "samples": len(samples),
        "duration": meta["duration"],
        "count_mean": round(float(counts.mean()), 2),
        "count_max": round(float(counts.max()), 2),
        "count_min": round(float(counts.min()), 2),
        "count_p95": round(float(np.percentile(counts, 95)), 2),
        "count_max_t": busiest["t"],
        "count_max_clock": clock_of(start, busiest["t"]),
        "peak_mean": round(float(peaks.mean()), 2),
        "peak_max": round(float(peaks.max()), 2),
        "peak_p95": round(float(np.percentile(peaks, 95)), 2),
        "peak_max_t": peak_at["t"],
        "peak_max_clock": clock_of(start, peak_at["t"]),
    }


def write_csv(path: Path, samples: list[dict], start: datetime | None, smooth: int) -> None:
    smoothed_count = rolling_mean([s["count"] for s in samples], smooth)
    smoothed_peak = rolling_mean([s["peak"] for s in samples], smooth)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample", "frame", "seconds", "clock", "count", "peak_density", "count_smooth", "peak_smooth"]
        )
        for sample, count_s, peak_s in zip(samples, smoothed_count, smoothed_peak):
            writer.writerow(
                [
                    sample["i"],
                    sample["frame"],
                    f"{sample['t']:.3f}",
                    clock_of(start, sample["t"]),
                    f"{sample['count']:.3f}",
                    f"{sample['peak']:.4f}",
                    f"{count_s:.3f}",
                    f"{peak_s:.4f}",
                ]
            )


def render_chart(path: Path, samples: list[dict], meta: dict, summary: dict, start: datetime | None) -> bool:
    """Publication-ready timeline figure. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib is not installed — skipping timeline.png (`pip install matplotlib`)")
        return False

    seconds = [s["t"] for s in samples]
    if start is not None:
        x = [start + timedelta(seconds=value) for value in seconds]
    else:  # elapsed time, still plotted on a date axis so the formatter is shared
        base = datetime(2000, 1, 1)
        x = [base + timedelta(seconds=value) for value in seconds]

    panels = series_panels(meta)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "figure.dpi": 200})
    # One panel gets its own height rather than half a two-panel figure: a chart
    # for a paper should not come out letterboxed because the other series was
    # not asked for.
    height_in = 2.9 * len(panels) + 0.4
    figure, axes = plt.subplots(len(panels), 1, figsize=(9.2, height_in), sharex=True)

    for axis, (title, key, unit, color) in zip(np.atleast_1d(axes), panels):
        values = [s[key] for s in samples]
        smoothed = rolling_mean(values, meta["smooth"])
        axis.fill_between(x, values, color=color, alpha=0.13, linewidth=0)
        axis.plot(x, values, color=color, alpha=0.45, linewidth=0.9)
        if meta["smooth"] > 1:
            axis.plot(x, smoothed, color=color, linewidth=1.9, solid_capstyle="round")
        # The threshold belongs to one series: 150 birds in the frame and 150
        # inside one patch are different events, and drawing it on both panels
        # washes the other one in red for no reason.
        threshold = meta["threshold"] if meta["threshold_metric"] == key else 0.0
        if threshold > 0:
            axis.axhline(threshold, color="#c93632", linestyle=(0, (5, 4)), linewidth=1.0, alpha=0.8)
            axis.axhspan(threshold, max(max(values), threshold) * 1.12, color="#c93632", alpha=0.05, linewidth=0)

        top = int(np.argmax(values))
        axis.plot(
            x[top],
            values[top],
            "o",
            color=color,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=5,
        )
        # A peak at either end of the recording would hang off the axis if the
        # label stayed centered on it.
        position = (seconds[top] - seconds[0]) / max(seconds[-1] - seconds[0], 1e-6)
        axis.annotate(
            f"{values[top]:.1f} @ {clock_of(start, seconds[top])}",
            xy=(x[top], values[top]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="left" if position < 0.15 else "right" if position > 0.85 else "center",
            fontsize=8,
            color=color,
            fontweight="semibold",
        )

        axis.set_title(title, loc="left", fontweight="semibold", color="#172033")
        axis.set_ylabel(unit, fontsize=8.5, color="#526176")
        axis.set_ylim(0, max(max(values), threshold) * 1.18 or 1.0)
        axis.margins(x=0.01)
        axis.grid(axis="y", color="#e5eaf1", linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#d7dee8")
        axis.tick_params(colors="#64748b", labelsize=8)

    span = max(seconds[-1] - seconds[0], 1.0)
    axis = np.atleast_1d(axes)[-1]
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if span > 900 else "%H:%M:%S"))
    axis.set_xlabel("clock time" if start is not None else "elapsed time", fontsize=8.5, color="#526176")
    figure.autofmt_xdate(rotation=0, ha="center")

    figure.suptitle(
        f"Predicted density over time — {meta['name']}",
        x=0.012,
        y=0.985,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#172033",
    )
    peaks = {
        "peak": f"peak {summary['peak_max']:.1f} birds in one patch",
        "count": f"peak {summary['count_max']:.0f} birds in frame",
    }
    headline = (
        f"{peaks['peak']}, {summary['count_max']:.0f} in frame" if meta["report"] == "both" else peaks[meta["report"]]
    )
    # Title and subtitle are text of a fixed point size, so their room has to be
    # reserved in inches: as a fraction they would swallow a one-panel figure.
    figure.text(
        0.012,
        0.985 - 0.32 / height_in,
        f"{summary['samples']} samples · every {meta['interval']:.2f}s · "
        f"{clock_of(start, seconds[0])}–{clock_of(start, seconds[-1])} · {headline}",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#748298",
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.62 / height_in))
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return True


def pick_peak_samples(samples: list[dict], count: int, gap_seconds: float, metric: str = "peak") -> list[dict]:
    """Busiest moments, kept `gap_seconds` apart so they are not one event.

    "Busiest" follows the series being reported: with --report count the frames
    saved are the fullest ones, otherwise the most locally crowded ones.
    """
    chosen: list[dict] = []
    for sample in sorted(samples, key=lambda s: s[metric], reverse=True):
        if len(chosen) >= count:
            break
        if all(abs(sample["t"] - other["t"]) >= gap_seconds for other in chosen):
            chosen.append(sample)
    return sorted(chosen, key=lambda s: s["t"])


def write_peak_frames(
    args,
    model,
    device,
    samples: list[dict],
    meta: dict,
    start: datetime | None,
    target_size,
    black_mask,
    mask_contours,
    out_dir: Path,
) -> list[dict]:
    """Re-read the busiest frames and save density overlays of them."""
    if args.peak_frames <= 0 or not samples:
        return []

    metric = "count" if args.report == "count" else "peak"
    chosen = pick_peak_samples(samples, args.peak_frames, args.peak_gap_seconds, metric)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"[warn] cannot reopen {args.video} for peak frames")
        return []

    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    written: list[dict] = []
    try:
        for rank, sample in enumerate(chosen, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame"])
            ok, frame = capture.read()
            if not ok:
                continue
            if target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            original = frame.copy()
            if black_mask is not None:
                frame = frame.copy()
                frame[black_mask] = 0

            inputs = preprocess_batch([frame], device, mean, std)
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=use_amp):
                density = model(inputs).float()[0, 0].cpu().numpy()

            overlay = blend_density(original, density, args.vmax, black_mask)
            draw_mask_region(overlay, mask_contours)
            # White, so the busiest window is never confused with the red mask
            # outline around it.
            x0, y0, x1, y1 = sample["box"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), PEAK_BOX_COLOR, max(2, round(overlay.shape[1] / 800)))

            # Peak frames stay full size whatever --frame-width says: there are a
            # handful of them and they are the pictures that end up in a report.
            path = out_dir / f"peak_{rank:02d}_t{int(sample['t']):06d}.jpg"
            caption = "  ".join(
                (
                    clock_of(start, sample["t"]),
                    metric_text(sample, "peak", args.threshold, args.threshold_metric),
                    metric_text(sample, "count", args.threshold, args.threshold_metric),
                )
            )
            save_jpeg(path, overlay, 0, caption, args.caption_corner)
            written.append({**sample, "path": str(path), "clock": clock_of(start, sample["t"])})
    finally:
        capture.release()
    return written


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds must be > 0")
    if args.window < 1 or args.batch_size < 1:
        raise SystemExit("--window and --batch-size must be >= 1")
    if not os.path.isfile(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    start = parse_start_time(args.start_time)
    target_size = parse_frame_size(args.frame_size)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.strip())

    checkpoint = Path(resolve_checkpoint(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_shufflenet_density_model(model_path=checkpoint, device=device, fuse=not args.no_fuse)
    model.eval()
    print(f"Model: {checkpoint}  (device={device})")

    black_mask, mask_contours = None, ()
    if args.mask_image:
        probe = cv2.VideoCapture(args.video)
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()
        size = target_size or (width, height)
        black_mask, mask_contours = load_static_mask(args.mask_image, *size)

    if args.threshold <= 0:
        alarm_config = Path(args.alarm_config)
        if not alarm_config.is_absolute():
            alarm_config = PROJECT_ROOT / alarm_config
        threshold, note = threshold_from_alarm_config(args.video, alarm_config)
        if threshold > 0:
            # An alarm-config threshold is a whole-frame count, whatever
            # --threshold-metric happens to default to.
            args.threshold, args.threshold_metric = threshold, "count"
            print(f"Threshold: {threshold:.0f} birds, from {alarm_config.name} ({note})")
            if black_mask is None:
                # These thresholds were calibrated on hard-black masked input,
                # and the masks drop a third to two thirds of the frame. Charted
                # against an unmasked count the line sits far too low.
                print(
                    "  [warn] no --mask-image: this threshold was calibrated on masked input, so an "
                    "unmasked count will cross it early"
                )
        else:
            print(f"Threshold: none ({note})")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "outputs" / "video_density" / Path(args.video).stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, meta = sample_video(args, model, device, black_mask, mask_contours, target_size, output_dir, start)
    if not samples:
        raise SystemExit("No frames were sampled — is the video readable?")

    meta.update({"start_time": start.strftime("%Y-%m-%d %H:%M:%S") if start else "", "checkpoint": str(checkpoint)})
    summary = summarize(samples, meta, start)

    csv_path = output_dir / "timeline.csv"
    json_path = output_dir / "timeline.json"
    write_csv(csv_path, samples, start, args.smooth)
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "samples": samples}, indent=2),
        encoding="utf-8",
    )

    chart_path = output_dir / "timeline.png"
    has_chart = False if args.no_chart else render_chart(chart_path, samples, meta, summary, start)
    frames = write_peak_frames(
        args, model, device, samples, meta, start, target_size, black_mask, mask_contours, output_dir
    )

    print("-" * 78)
    print("DENSITY TIMELINE")
    print(f"  Samples          : {summary['samples']} over {summary['duration']:.1f}s")
    # --report picks which series is the answer here; the other one is still in
    # the CSV and the JSON for anyone who wants it.
    if args.report in ("both", "peak"):
        print(
            f"  Peak density     : {summary['peak_max']:.2f} birds in one {meta['window_px']}px patch "
            f"at {summary['peak_max_clock']}"
        )
        print(f"  Mean density     : {summary['peak_mean']:.2f}   (p95 {summary['peak_p95']:.2f})")
    if args.report in ("both", "count"):
        print(f"  Peak count       : {summary['count_max']:.1f} birds at {summary['count_max_clock']}")
        print(f"  Mean count       : {summary['count_mean']:.1f}   (p95 {summary['count_p95']:.1f})")
    if args.threshold > 0 and args.report not in ("both", args.threshold_metric):
        print(
            f"  [note] --threshold is on the '{args.threshold_metric}' series, which --report "
            f"{args.report} leaves out, so no threshold line is drawn"
        )
    print(f"  Artifacts        : {output_dir}")

    # Machine-readable trailer: the web UI reads these to render its own chart,
    # summary cards and peak-frame strip. One JSON object per line.
    print("TIMELINE META " + json.dumps(meta))
    print("TIMELINE SUMMARY " + json.dumps(summary))
    for kind, path, ok in (("chart", chart_path, has_chart), ("csv", csv_path, True), ("json", json_path, True)):
        if ok:
            print("TIMELINE ARTIFACT " + json.dumps({"kind": kind, "path": str(path)}))
    for frame in frames:
        print(
            "TIMELINE ARTIFACT "
            + json.dumps(
                {
                    "kind": "frame",
                    "path": frame["path"],
                    "t": frame["t"],
                    "clock": frame["clock"],
                    "peak": frame["peak"],
                    "count": frame["count"],
                }
            )
        )


if __name__ == "__main__":
    main()
