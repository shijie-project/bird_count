# Live deployment

The live runtime counts every camera continuously and raises pile-up alarms.

```bash
python -m birdcount.deploy --check     # validate everything, list what each camera gets, exit
python -m birdcount.deploy             # run (also: birdcount-deploy)
```

`--check` loads the settings, topology, alarm config, checkpoint and masks and
prints, per camera, its zone, devices, alarm threshold and mask, flagging any
camera without both, and how long the alarm evidence paths will get. Run it
after every configuration change.

## Processes

The **grabber** decodes every camera (on the GPU video engine when
`HW_DECODE=1`) into a shared-memory frame ring at `FPS`; the **inferencer**
batches the ring through the model and returns counts and density maps; the
**main process** runs the alarm, the handlers and the panels and never loads
torch; the **monitor** (a camera grid window) runs only while it is open.

Stopping: the operator panel's **Terminate**, Ctrl+C or closing the console
all run the same shutdown — devices off, the current recording segment
finished, evidence and logs flushed — before the processes exit. Only one
runtime can run on a machine at a time.

## Cameras, zones and identity

`configs/topology.yaml` lists the zones; each zone has cameras, speakers and
smart plugs. A camera is `IP` or `{ip: IP, mac: MAC}`. The **bare MAC** (12
hex digits, no separators, e.g. `B8A44FD51C3C`) is the camera's identity: it
picks its region mask and its alarm threshold, and unlike an IP it survives
re-addressing. A camera without a MAC counts unmasked and never alarms. Fill
the MACs in on the farm network:

```bash
python -m birdcount.deploy.tools.discover_ids            # paste-ready MACs
python -m birdcount.deploy.tools.discover_ids --write    # write them into topology.yaml
python -m birdcount.deploy.tools.discover_ids --verify   # compare topology.yaml with the cameras
```

The runtime never asks a camera for its MAC: a camera rebooting at startup
would lose its mask for the whole run.

`SOURCE_TYPE=video` replays the clips in `VIDEO_PATHS` instead of the cameras;
a clip's MAC comes from its filename, and the zone of the camera with that MAC
decides which devices its alarm drives.

## Region masks

Each camera counts only its own part of the shed. `MASK_DIR` holds one PNG per
camera named by MAC, white = count, black = ignore. Draw or edit them on a
frame of each camera with

```bash
python -m birdcount.deploy.tools.mask_editor
```

**Masks are not cosmetic.** The thresholds were calibrated on masked counts,
and the masks remove 29–67 % of the frame: an unmasked camera over-counts
(about 36 % on average) and raises false alarms. The mask is applied twice —
to the frame before the model (it was trained on frames with that area
literally black) and to the density map before summing (to clamp what bleeds
across the edge). A camera without a mask counts the whole frame and is named
in a startup warning.

## Pile-up alarm

Thresholds, rules and which handler acts at which level live in one place,
`configs/alarm.json`, keyed by MAC (in the vendor's `axisN/MAC` spelling,
which the SMS text quotes). Per camera:

```text
count >= threshold for 10 s          -> Level 1
event still open after 3 more min    -> Level 2 (count at the threshold again)
event still open after 3 more min    -> Level 3, then no further escalation
count < 0.9 x threshold for 30 s     -> recovery: the event closes
```

The count compared is the median of the last 2 s (`smoothing_seconds`), so
one noisy frame neither restarts the 10 s breach nor the 30 s recovery, and
an event clears only below `recovery_ratio` x threshold, so a count hovering
at the threshold keeps one event open instead of opening and closing one
(two SMS) every minute. Both are under `rules`; 0 and 1.0 turn them off.

A directional-motion filter skips the common false positive of a dense flock
walking across the frame: it tracks the density centroid and ignores samples
whose drift is fast, far and consistent. Handlers subscribe to levels under
`actions` (SMS 1–3, speakers and smart plugs 1 by default); recovery reaches
all of them, so a device switched on is switched off when the event closes,
and stays on while any other camera in its zone is still alarming. The
operator panel's **Cancel All** drops every open event and stops every device.

```bash
ENABLE_SMS_ALARM=1 python -m birdcount.deploy        # dry run: payloads and evidence on disk, nothing sent
# real SMS: also SMS_ALARM_REAL_WORKER=1 and FARM_SMS_API_KEY in .env
```

Evidence goes to `ALARM_OUTPUT_DIR` (default the alarm config's
`outputs/alarm`): `events.jsonl` and `notifications.jsonl`, and per event
`events/<camera>/<time>_<id>/` with `counts.csv`, `pre_window_counts.csv`,
`summary.json` and the alarm snapshots (`level1_<unix time>.jpg`, ...). The
files and CSV columns are the delivery package's; every payload carries the
full event id and snapshot path.

Windows limits a path to 259 characters unless long paths are enabled, and
that setting cannot be assumed on a new machine. Evidence names are kept short
(about 90 characters below the output folder), so any output folder up to ~170
characters works without it; `--check` prints the longest path, and the
runtime warns at startup if one would not fit.

**Rehearsing an event.** `configs/alarm_debug.json` keeps the real thresholds
with short timings (Level 1 after 5 s, +20 s per level, recovery after 10 s)
and writes to `outputs/alarm_debug/`. With `DEBUG=1` and SMS in dry run, the
debug panel's per-camera trigger makes that camera look piled up (1.2 × its
threshold), so one click plays Level 1 → 2 → 3 and releasing it the recovery.
With real SMS on, the trigger only switches the zone's devices.

```bash
DEBUG=1 ENABLE_SMS_ALARM=1 ALARM_CONFIG_PATH=configs/alarm_debug.json python -m birdcount.deploy
```

## Records, monitor and devices

- **Count log** (`ENABLE_COUNT_LOGGER=1`): inside `COUNT_LOG_HOURS`, one CSV
  row per camera per second (count mean/max/min, centroid, alarm level, motion
  exclusions) in `COUNT_LOG_DIR/<date>/<MAC>.csv`, plus snapshots from each
  camera's floor up (60 % of its threshold, rounded up to 10; below it the
  shed is at ease) — per 10-bird bin up to 3 a day, 20 min apart, so a week
  holds ~20 per bin for the calibration (`high` from 80 % of the threshold,
  `bin` below) — and a `meta.json` naming the model, thresholds and floors.
  About 40 MB of rows and ~600 snapshots (some tens of MB) a day for 20 cameras;
  replaying them through the alarm engine is how thresholds are re-tuned.
  Counts are only comparable under the model in `meta.json`.
- **Live status** (`STATUS_PATH`): every camera's count, alarm level and
  picture state, rewritten every second for the farm backend.
- **Recording** (`ENABLE_VIDEO_RECORDER=1`): every camera into
  `VIDEO_SEGMENT_SECONDS` segments at the inference rate, under
  `VIDEO_RECORD_DIR/stream_<n>/<date>/`; switchable per camera in the debug panel.
- **Monitor** (`ENABLE_MONITOR=1`, or the operator panel): a grid of every
  camera with its count, optionally the density heatmap and the mask.
- **Audit log** (`AUDIT_LOG_PATH`): every operator action and alarm as JSONL.
- **Devices by hand**: `python -m birdcount.deploy.tools.devices plug status`
  (`speaker` / `plug`, `on` / `off` / `status` / `toggle`, one zone with `--zone`).

## Farm backend

```bash
python -m birdcount.deploy.webui            # http://127.0.0.1:8430
```

The page to keep open on the farm computer, a separate process from the
runtime that reads only the files the runtime writes, so it shows the history
while the runtime is down and the live state within a second or two while it
runs. **Farm**: every camera on the shed's floor plan (zones from
`topology.yaml`; cameras whose MAC is not there yet are grouped by recorder),
with its count against its threshold, and the day's alarm periods of every
camera on one time line. **Cameras**: each camera's day in a sortable table.
A camera's page: its count over the day against the threshold and the
recovery line (drag across it to zoom in), its alarm events with the pictures
sent, and its snapshots with the model's level on each (its reason on hover), by level,
by count or in time order; zoomed in, the snapshots are those in the stretch shown.
**Check alarms**: one alarm event at a time, with the pictures it sent (Level 1,
2, 3, recovery), how long it lasted and the model's level: `1` real pile-up /
`2` false alarm / `3` can't tell, into `COUNT_LOG_DIR/<date>/labels.csv`; the
area outside the camera's mask is darkened as the model sees it (`M`
toggles). Per camera the verdicts say whether its threshold alarms too often
(the Thresholds page counts them); the header and the farm lanes mark the
alarms still to check. **Model ratings**:
the day's `ratings.csv` at a glance, one lane per camera (each rated picture by count and
crowding level, with the threshold); a lane opens that camera's page. **Thresholds**: the
calibration's suggestion per camera (see [Threshold calibration](calibration.md)).
**Masks**: a camera's counted area
as polygons over its latest snapshot (drag a point, click an edge to add one,
right-click to drop one or click it and press Delete, drag inside to move it all,
Ctrl+Z; Backspace while drawing takes the last point back); Apply writes the
mask file in `MASK_DIR` that the runtime, the evaluation and the training read,
after copying the old one to `MASK_DIR/_history/`, where Restore takes it back.
The runtime reads masks at start, and each count-log day records the masks it
was counted with (`meta.json` "masks").

The header's **Start / Stop / Restart** (each asks once more) run the runtime
from the page, whether it was started there or by hand: Stop asks it to shut
down the way Terminate does (devices off, recording finished; a request file
next to `STATUS_PATH`), and ends its processes only when it has not stopped
within a minute, as a hung runtime would not. Start runs
`python -m birdcount.deploy` in a console window of its own, so it keeps
running when the page's process ends; the header shows "Starting…" until its
first heartbeat, or why it did not come up. Restart after Apply on the Masks
page, or when the runtime stops answering. It listens on this machine only.
