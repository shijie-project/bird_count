# birdcount

Density-map bird counting for poultry pile-up detection: a live multi-camera
deployment that counts every camera several times a second and raises
pile-up alarms (`birdcount.deploy`), and the training, evaluation, video
and labelling tooling that produces its model (`birdcount.training`), on a
shared core (`birdcount.core`).

```
birdcount/
├── core/        settings, paths, the model, the inference pipeline, masks, camera identity, drawing
├── deploy/      the live runtime: frame ring, grabber, inferencer, alarm, handlers, monitor, panels, tools
└── training/    data, train (+ sweeps), eval, video, labeling, webui (the control panel)
configs/         topology.yaml (the farm), alarm.json (thresholds and rules)
tests/           mirrors the package; pytest
```

`deploy` and `training` both build on `core` and never import each other;
`tests/test_layering.py` enforces it.

## Setup

```bash
# 1. PyTorch built for your CUDA driver (pip would otherwise pick the CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. This package, editable, with what you need
pip install -e ".[deploy]"              # the live runtime
pip install -e ".[training]"            # video timelines, Label Studio
pip install -e ".[dev]"                 # tests, ruff, pre-commit
pre-commit install

# 3. Configuration
cp .env.example .env                    # then edit; .env is git-ignored
```

`.env.example` lists every key with what it does; it is the catalogue, and a
key in `.env` that no setting reads is reported at startup (a typo would
otherwise silently do nothing). Relative paths in `.env` are resolved against
the project root, so commands behave the same from any working directory.
Process environment variables override `.env`.

### Data layout

```
DATA_ROOT (default ../data)
├── images/
│   ├── all/<frame>.jpg          every labelled frame
│   ├── train/, val/             the split (written by `labeling split`)
│   └── masks/<MAC>.png          region masks, if MASK_DIR points here
├── annotations/
│   └── all.json, train.json, val.json
└── videos/                      recordings for the video tools and video mode
```

A split's json is `{"images": [...], "annotations": [{"image_id": "<frame stem>",
"points": [[x, y], ...]}]}`, points in the image's own pixels. `MODEL_PATH` is
the checkpoint every app counts with; `OUTPUT_ROOT` (default `outputs/`)
receives every run.

## Live deployment

```bash
python -m birdcount.deploy --check     # validate everything, list what each camera gets, exit
python -m birdcount.deploy             # run (also: birdcount-deploy)
```

`--check` loads the settings, topology, alarm config, checkpoint and masks and
prints, per camera, its zone, devices, alarm threshold and mask, flagging any
camera without both, and how long the alarm evidence paths will get. Run it
after every configuration change.

Processes: the **grabber** decodes every camera (on the GPU video engine when
`HW_DECODE=1`) into a shared-memory frame ring at `FPS`; the **inferencer**
batches the ring through the model and returns counts and density maps; the
**main process** runs the alarm, the handlers and the panels and never loads
torch; the **monitor** (a camera grid window) runs only while it is open.

Stopping: the operator panel's **Terminate**, Ctrl+C or closing the console
all run the same shutdown — devices off, the current recording segment
finished, evidence and logs flushed — before the processes exit. Only one
runtime can run on a machine at a time.

### Cameras, zones and identity

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

### Region masks

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

### Pile-up alarm

Thresholds, rules and which handler acts at which level live in one place,
`configs/alarm.json`, keyed by MAC (in the vendor's `axisN/MAC` spelling,
which the SMS text quotes). Per camera:

```text
count >= threshold for 10 s          -> Level 1
event still open after 3 more min    -> Level 2
event still open after 3 more min    -> Level 3, then no further escalation
count < threshold for 30 s           -> recovery: the event closes
```

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

### Records, monitor and devices

- **Count log** (`ENABLE_COUNT_LOGGER=1`): inside `COUNT_LOG_HOURS`, one CSV
  row per camera per second (count mean/max/min, centroid, alarm level, motion
  exclusions) in `COUNT_LOG_DIR/<date>/<MAC>.csv`, plus review snapshots —
  `high` at 80 % of the threshold (at most one a minute) and `routine` every
  15 minutes — and a `meta.json` naming the model. About 40 MB of rows a day
  for 20 cameras; replaying them through the alarm engine is how thresholds are
  re-tuned. Counts are only comparable under the model in `meta.json`.
- **Recording** (`ENABLE_VIDEO_RECORDER=1`): every camera into
  `VIDEO_SEGMENT_SECONDS` segments at the inference rate, under
  `VIDEO_RECORD_DIR/stream_<n>/<date>/`; switchable per camera in the debug panel.
- **Monitor** (`ENABLE_MONITOR=1`, or the operator panel): a grid of every
  camera with its count, optionally the density heatmap and the mask.
- **Audit log** (`AUDIT_LOG_PATH`): every operator action and alarm as JSONL.
- **Devices by hand**: `python -m birdcount.deploy.tools.devices plug status`
  (`speaker` / `plug`, `on` / `off` / `status` / `toggle`, one zone with `--zone`).

## Training

```bash
python -m birdcount.training.train                       # the production recipe, ~12 min on an RTX 5090
python -m birdcount.training.train --lr 5e-5 --name lr5e-5
python -m birdcount.training.train --resume outputs/train/<run>
python -m birdcount.training.train --help                # every option, with its default
```

The defaults are the recipe of the production model: DM-Count loss (OT +
count + TV) with a Gaussian-density auxiliary loss, AdamW at 1e-5 with 5
warm-up epochs then cosine decay, EMA 0.996, 1000 epochs of 512 px crops.
Each run writes `OUTPUT_ROOT/train/<time>[-name]/`: `config.json`, `train.log`,
`metrics.jsonl` (one line per epoch), `summary.json`, `last.tar` (resume
state) and the five best `best_ep*_mae*_rmse*.pth`. Validation scores the EMA
weights exactly as deployed (720p, region masks, fused fp16), and the best
checkpoint minimises 2·RMSE + MAE. Ctrl+C saves the state of the last
finished epoch; `--resume` continues with the run's own config.

`--hard-mining-gamma` (off by default) oversamples the images the model keeps
getting wrong, by their smoothed *relative* error, with the weights capped so
mislabelled images cannot take over an epoch.

```bash
python -m birdcount.training.train.sweep --grid lr=1e-5,5e-5 --grid waux=0.5,1 -- --epochs 300
```

A sweep runs one training per combination in `OUTPUT_ROOT/train/sweep-<time>/`
and prints a leaderboard; re-running it with `--sweep-dir` resumes it.

## Evaluation

```bash
python -m birdcount.training.eval --split val                    # MODEL_PATH, or --ckpt
python -m birdcount.training.eval --split val --blobs --grid 4x6 # + errors per density blob / grid tile
```

Frames are evaluated as the runtime sees them: fitted to 720p, region mask by
MAC, fused fp16. The headline numbers are **counting accuracy** (1 −
Σ|pred − GT| / ΣGT, the share of all birds counted right) and
**Acc@(5,10%)** (images within 5 birds or 10 %, whichever is looser); MAE,
RMSE, bias, per-band errors and pile-up detection follow. Results go to
`OUTPUT_ROOT/eval/<checkpoint>/<split>-<time>/`: `metrics.json`,
`per_image.csv`, `blobs.json` / `grid.json`, and overlays.

## Video analysis

```bash
python -m birdcount.training.video ../data/videos/<clip>.mkv               # count + busiest patch, one sample a second
python -m birdcount.training.video <clip>.mkv --every 0 --render overlay   # full-rate overlay video
python -m birdcount.training.video ../data/videos --frames boxes --count-threshold 140
```

Counted exactly as deployed. Each video gets `OUTPUT_ROOT/video/<clip>/<time>/`
with `samples.jsonl` (written as the run goes), `timeline.csv`,
`summary.json`, `timeline.png` and overlays of the busiest moments in
`peaks/`. A camera's alarm threshold, for `--count-threshold`, is listed by
`python -m birdcount.deploy --check`.

## Labelling

```bash
python -m birdcount.training.labeling pull -o export.json             # the Label Studio project, via the API
python -m birdcount.training.labeling convert export.json -o ../data/annotations/all.json
python -m birdcount.training.labeling split                           # train / val by camera and recording session
python -m birdcount.training.labeling prelabel <eval run>/blobs.json  # density blobs as predictions to correct
python -m birdcount.training.labeling --help                          # merge, to-ls, import, dedupe, apply-mask, ...
```

The server, token and project come from `LABEL_STUDIO_*` in `.env`; image URLs
are paths under `LABEL_STUDIO_FILES_ROOT` (Label Studio's local-files root,
default DATA_ROOT). Commands that delete or overwrite anything in Label Studio
only report unless given `--apply`; `dedupe-images --apply` moves duplicates
to a quarantine folder with a manifest, never deletes.

## Control panel

```bash
python -m birdcount.training.webui          # http://127.0.0.1:8420
```

A local web page for all of the training side: forms generated from each
tool's own options, live training curves, evaluation tables and overlays,
video timelines, a dataset browser (annotated points, and an evaluation's
prediction per image), and Label Studio with its ngrok tunnel
(`LABEL_STUDIO_DATA_DIR`, `LABEL_STUDIO_PUBLIC_URL`). Runs land in OUTPUT_ROOT
exactly as from a shell, and runs started from a shell are listed too. Stop
asks a run to stop as Ctrl+C would, so training keeps its state for
`--resume`. It listens on this machine only unless given `--host 0.0.0.0`.

## Design notes

Things that are easy to break without noticing:

- **The calibration depends on the preprocessing.** Thresholds were fitted to
  counts at 1280×720 with hard-black masks and the density masked before the
  sum. Change `IMAGE_WIDTH`/`IMAGE_HEIGHT`, the masks or the mask handling and
  every threshold needs re-fitting (from the count log).
- **One inference path.** Deployment, evaluation, validation during training
  and the video tools all count through `core.inference.DensityPredictor` with
  a model from `core.model.load_inference_model` (eval, Conv+BN fused, fp16),
  so a number from any of them is the number the alarm would have seen.
- **Output stride 8.** The density map is 1/8 of the input in each direction;
  inputs must be multiples of 8 (`core.model.OUTPUT_STRIDE`, never a literal 8).
- **EMA weights are the model.** Validation and every saved `best_*.pth` use
  the EMA copy.
- **The OT loss is averaged over the batch**, so its weight `--wot` does not
  depend on the batch size.
- **Heatmaps use an absolute scale** (`core.visualize.HEATMAP_VMAX`): never
  normalise a density map per frame, or a sparse frame looks as hot as a
  pile-up.
- **Heavy imports stay where they are used.** The grabber, monitor, main
  process and control panel never load torch; `tests/test_import_cost.py`
  holds every entry module to an import budget.

## Coming from v2

| v2 | v3 |
|---|---|
| `python run.py` | `python -m birdcount.deploy` (`--check` first) |
| `python train.py ...` | `python -m birdcount.training.train` (see `--help`; the defaults are the production recipe) |
| `python test.py ...` | `python -m birdcount.training.eval` |
| `tools/sweep.py`, `tools/summarize_sweep.py` | `python -m birdcount.training.train.sweep` |
| `tools/inference_video.py`, the web UI's video timeline | `python -m birdcount.training.video` |
| `tools/annotations/*.py`, the web UI's Label Studio ops | `python -m birdcount.training.labeling <command>` |
| `tools/discover_camera_ids.py` | `python -m birdcount.deploy.tools.discover_ids` |
| `tools/handlers/*_switcher.py` | `python -m birdcount.deploy.tools.devices` |
| the web UI's region mask GUI | `python -m birdcount.deploy.tools.mask_editor` (`labeling apply-mask` for image folders) |
| `python -m webui`, `tools/starter.sh label-studio` | `python -m birdcount.training.webui` |
| `tools/visualize_data.py` (streamlit) | the control panel's Data page |
| `tools/exports/`, `tools/calibrate_trt_int8.py` | removed: the runtime is PyTorch, which is not the bottleneck |

- **`.env`:** `DEMO_VIDEO_PATH` / `DEMO_VIDEO_PATHS` are now `VIDEO_PATHS`;
  `VERBOSE_DEBUG` is gone. New keys (all in `.env.example`): `DATA_ROOT`,
  `OUTPUT_ROOT`, `LOG_DIR`, `LOG_LEVEL`, `TOPOLOGY_PATH`, `CAMERA_URL` /
  `CAMERA_USERNAME` / `CAMERA_PASSWORD`, `HW_DECODE`, `MONITOR_FPS`,
  `FARM_SMS_API_KEY` (v2 read it from the shell only) and the
  `LABEL_STUDIO_*` keys.
- **Topology:** a camera's MAC sits next to its IP (`{ip, mac}`) instead of in
  a separate `camera_ids:` list aligned by position.
- **Outputs:** every run is under `OUTPUT_ROOT/{train,eval,video}/`
  (checkpoints included) instead of `../ckpts/<hyper-parameters>/` and
  `logs/webui/`. v2 checkpoints load as they are.
- **Evaluation numbers changed meaning:** v3 evaluates at the deployment
  geometry with masks, so its MAE is not comparable with v2 `test.py`'s.
- **Unchanged on purpose:** `configs/alarm.json` (format and thresholds), the
  SMS text, the evidence layout and the count log format.

## Development

```bash
pytest                                   # everything
pytest -m "not gpu and not slow"         # what the pre-commit hook runs
```

`tests/test_layering.py` enforces the package structure (dependency direction,
private modules, the control panel as a leaf) and `tests/test_import_cost.py`
keeps light modules from loading torch / cv2 / numpy. Ruff formats and lints
(`pre-commit run -a`).
