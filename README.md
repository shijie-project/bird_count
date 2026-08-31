# Bird Count

Crowd-counting model for chicken pile-up detection. A ShuffleNet-V2 U-Net
density estimator is trained on point annotations, then deployed at the edge
to monitor up to ~22 simultaneous RTSP camera streams. A pile-up event is
declared when the estimated density in a frame exceeds a configurable
threshold.

- **Task**: density estimation → integrate to per-frame chicken count → threshold
- **Model**: ShuffleNet-V2 1.0× (truncated at stage3) + U-Net lateral skip; ~1.4M params; output stride 8
- **Loss**: DM-Count (OT + count L1 + distribution-match) plus a Gaussian-density auxiliary term that fixes DM-Count's "silent on empty regions" failure mode
- **Eval target**: 1080Ti (FP16/INT8), batch 22 at 720×1080 or 1080×1920

______________________________________________________________________

## Quick start

```bash
# 1. Install (pin torch to your CUDA build first, then:)
pip install -r requirements.txt

# 2. Point the model loader at a checkpoint via .env
echo MODEL_PATH=../ckpts/your_best.pth > .env

# 3. Train, eval, deploy
python train.py                                  # train
python test.py                                   # evaluate (writes density overlays + metrics)
python run.py                                    # live-stream deployment

# ...or drive train/test from the browser
python -m webui                                  # http://127.0.0.1:8420
```

______________________________________________________________________

## Project layout

```
bird_count/
├── train.py              training entrypoint
├── trainer.py            training orchestration (Trainer class)
├── test.py               offline evaluation on a dataset split
├── run.py                live-stream inference launcher
├── metrics.py            counting metrics: MAE/RMSE/R²/F1/etc.
├── utils.py              Logger, AverageMeter, SaveHandle, set_seed
├── requirements.txt
├── .env                  MODEL_PATH and other env vars (not committed)
│
├── webui/                browser control panel (train / test / annotations)
│   ├── server.py         FastAPI routes (schema, runs, log tail, gallery)
│   ├── runs.py           child-process supervision + log parsing
│   ├── services.py       long-running servers (Label Studio), outside the run slot
│   ├── schema.py         ENTRYPOINTS registry + argparse introspection
│   ├── ops/              the scripts only this UI drives (one file per op)
│   └── static/           single-page front end (no build step, no CDN)
│
├── datasets/             training-data pipeline
│   ├── bird.py           Bird Dataset class + transform factories + DOWNSAMPLE_RATIO
│   ├── transforms.py     keypoint-aware Compose + 11 augmentation classes
│   ├── targets.py        density-target generation + sum-pool helper
│   └── __init__.py       collate, seed_worker
│
├── models/               density-estimator architecture
│   ├── shufflenet.py     ShuffleNetDensityNet + U-Net skip + factory
│   ├── ema.py            ModelEMA (eval-time exponential moving average)
│   └── utils.py          generic helpers: Conv+BN fusion, state_dict extractor
│
├── losses/               training loss
│   ├── dm_count.py       DMCountLoss orchestrator (returns total + parts dict)
│   ├── ot.py             OTLoss (Sinkhorn-based)
│   ├── density.py        DensityAuxLoss (Gaussian-smoothed L1)
│   └── sinkhorn.py       Sinkhorn-Knopp solver
│
├── alarm/                pile-up alarm core (pure stdlib, vendored)
│   └── camera_ids.py     bare MAC -> the config's axisN/MAC ids
│   ├── state_machine.py  Level 1/2/3 + recovery per camera
│   ├── motion_filter.py  drops "whole flock walking past" false positives
│   ├── evidence.py       per-event counts.csv / snapshots / actions.jsonl
│   ├── sms_worker.py     POSTs snapshot + message to the SMS service
│   └── camera_ids.py     stream -> `axisN/MAC` resolution
│
├── configs/
│   └── alarm.json        alarm thresholds, rules, SMS worker settings
│
├── runtime/              live deployment subsystem
│   ├── stream_capture.py  RTSP capture per camera into shared memory
│   ├── memory_manager.py  shared-memory state machine for cross-process frames
│   ├── inference_process.py  GPU batch inference loop
│   ├── result_process.py     dispatches results to handlers
│   ├── task_dispatcher.py    orchestrates capture/inference/result lifecycles
│   ├── handlers/             smart-plug, speaker, monitor, sms_alarm handlers
│   ├── audit.py
│   ├── config.py             pydantic config schema
│   └── gui.py
│
└── tools/                stand-alone utilities
    ├── annotations/         export → training-JSON pipeline (files, not the API)
    ├── inference_video.py   density overlay over a single video file
    ├── export_onnx.py       export trained .pth -> .onnx with dynamic axes
    ├── calibrate_trt_int8.py  build a TensorRT INT8 calibration cache
    ├── test_inference_time.py  latency benchmark across runtimes
    ├── sweep.py             cross-platform hyperparameter sweep driver
    ├── sweep.sh             same as sweep.py for Linux/macOS users
    ├── summarize_sweep.py   leaderboard of best_*.pth files in a sweep dir
    ├── visualize_data.py    Streamlit data inspector
    └── plug_switcher.py     manual TP-Link Tapo plug control
```

`../data/` (sibling of this repo) holds the dataset; `../ckpts/` holds
checkpoints. Both paths are CLI-overridable via `--data-dir` / `--checkpoint-dir`.

______________________________________________________________________

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
.venv\Scripts\activate              # Windows PowerShell
```

### 2. PyTorch with the right CUDA

`requirements.txt` lists `torch>=2.0` / `torchvision>=0.15` so your CUDA
matches your driver. For a 1080Ti on CUDA 12.1:

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 3. Everything else

```bash
pip install -r requirements.txt
```

### 4. Data layout

The `Bird` dataset expects:

```
../data/
├── images/
│   ├── train/<frame>.jpg
│   └── val/<frame>.jpg
└── annotations/
    ├── train/<frame>.txt    # one keypoint per line
    └── val/<frame>.txt
```

Annotation format per line: either `x_pixel y_pixel` (2 cols) or
`class_id x_norm y_norm` (3 cols, YOLO-style — auto-rescaled to pixels).

### 5. .env

```
MODEL_PATH=../ckpts/<your_best>.pth
```

Used by `test.py`, `tools/inference_video.py`, and the `runtime/` deployment
to find the model checkpoint without passing `--ckpt` every time.

______________________________________________________________________

## Workflows

### Training

```bash
python train.py \
    --data-dir ../data --checkpoint-dir ../ckpts \
    --crop-size 512 --batch-size 64 \
    --max-epoch 1000 --warmup-epochs 5 \
    --val-start 30 --val-epoch 5
```

What happens:

- Training crops are augmented with multi-scale (0.8–1.25×), random
  square crop, h/v flip, 90° rotation, color jitter, gamma, and Gaussian noise.
- Loss is DM-Count (OT + count + TV) + Gaussian-density auxiliary term
  (`--aux-sigma 2.0` by default); see `losses/dm_count.py`.
- Optimizer is AdamW with linear warmup → cosine annealing.
- An EMA copy of the weights is maintained for evaluation
  (`--ema-decay 0.999`); validation always runs on the EMA copy.
- Best model selected by `2 * MSE + MAE`; `--max-best-ckpts 5` keeps a rolling
  window of the best checkpoints (older ones are deleted).
- Per-epoch full state (`<epoch>_ckpt.tar`) lets you resume; only the
  most-recent one is kept.

Resume from an interrupted run:

```bash
python train.py --resume ../ckpts/<run-dir>/<N>_ckpt.tar
```

Common knobs (run `python train.py --help` for the full list):

| Flag                      | Default | Note                                                     |
| ------------------------- | ------- | -------------------------------------------------------- |
| `--lr`                    | 1e-5    | peak LR; cosine annealed to lr×0.01                      |
| `--batch-size`            | 64      | also see `--accum-steps`                                 |
| `--accum-steps`           | 1       | gradient accumulation for larger effective batch         |
| `--ema-decay`             | 0.999   | drop to ~0.996 if you raise `--accum-steps` to 4+        |
| `--waux`                  | 1.0     | Gaussian-density aux loss weight (set 0 to disable)      |
| `--aux-sigma`             | 2.0     | sigma in density-map (H/8) pixels; ≈ chicken radius / 8  |
| `--no-freeze-backbone-bn` | off     | finetune backbone BN stats (default: frozen to ImageNet) |
| `--deterministic`         | off     | enable cuDNN determinism for reproducible runs           |
| `--hard-mining-gamma`     | 0.0     | oversample high-error images; 0 disables (see below)     |

#### Hard-example mining

Off by default. `--hard-mining-gamma > 0` swaps the train loader's plain shuffle
for a `WeightedRandomSampler` whose weights track how wrong the model is on each
image, so images it keeps missing come up more often:

```bash
python train.py --hard-mining-gamma 1.0 --hard-mining-start-epoch 50
```

Three details are what make it work rather than backfire (`hard_mining.py`):

- **Relative error, not absolute.** Train counts span ~9–218 birds, so
  `|pred - gt|` is dominated by the crowded images regardless of how well the
  model does on them. The error is divided by the crop's GT count, floored by
  `--hard-mining-min-count` so near-empty crops don't explode the ratio.
- **Smoothed across visits** (`--hard-mining-ema 0.9`). Each visit sees a
  different random crop, so one observation mixes "this image is hard" with
  "this crop was hard".
- **Clipped** to `--hard-mining-clip 3.0`× either side of uniform. The
  highest-error images are also where mislabeled data lives; without the cap,
  mining will happily spend the epoch fitting bad annotations.

`--hard-mining-start-epoch` keeps sampling uniform early on (the error EMA still
warms up meanwhile) — before the model has learned anything, "hardest" just
means "most birds". Every `--val-epoch` epochs the log prints the current weight
range and the five hardest images; that list doubles as an annotation-quality
check, since a genuinely mislabeled image parks itself at the top and stays
there. Tracker state rides along in `<epoch>_ckpt.tar`, so `--resume` picks up
the accumulated errors instead of restarting cold.

Watch validation MAE after enabling it: if it degrades, the run is overfitting
to hard samples or to label noise — lower `gamma` or tighten `clip`.

### Evaluation

```bash
python test.py \
    --data-path ../data/annotated --split val \
    --ckpt ../ckpts/<run-dir>/best_ep0042_mae5.32_mse8.71.pth \
    --pileup-threshold 100 \
    --metrics-out eval.json
```

Outputs **two report sections**:

1. **EXHIBITION SUMMARY** — audience-friendly: average miscount in chickens,
   "% of images within ±N chickens", pile-up detection rate, false alarms.
1. **TECHNICAL METRICS** — MAE, RMSE, NAE, MAPE, Bias, R², Pearson, plus
   stratified MAE by count band (Empty / Low / Medium / High / Pile-up) and
   pile-up confusion matrix at the chosen threshold.

All artifacts go under `--output-dir`, which defaults to the checkpoint's own
directory (`../ckpts/<run-dir>/`). `--metrics-out` writes the structured report
as JSON — a relative path lands inside the output dir, an absolute path is used
as-is. Density-overlay PNGs are written to `<output-dir>/density_maps/` unless
you pass `--no-density-map`.

### Web UI

```bash
python -m webui                       # serves http://127.0.0.1:8420 and opens a tab
python -m webui --port 9000 --host 0.0.0.0   # expose on the LAN
python webui/__main__.py --no-browser        # same thing, for an IDE run config
```

To debug it in PyCharm, use the bundled **webui** run configuration (module
`webui`, working directory = project root) and hit Debug. Do not add `--reload`:
uvicorn's reloader runs the app in a child process, where your breakpoints will
not be hit. Breakpoints in `train.py`, `label-studio` etc. will not be hit
either — those are separate child processes; debug them with their own run
configuration.

A local control panel for training, evaluation and the annotation pipeline. The
configuration form is generated from each script's `argparse` spec at startup —
add a flag to a script and it shows up in the UI with its help text and default,
no UI change needed. Fields that differ from the default are marked with a dot,
required fields are badged (Start stays disabled until they are filled), and the
exact command is shown above the log so a run is always reproducible from a
terminal.

- **Train** — live loss / MAE / MSE curves for train and val, parsed from the
  trainer's own log lines; `saved best ->` events are highlighted.
- **Test** — metric cards (MAE, RMSE, MAPE, bias, R²), a per-image table sorted
  worst-first, a GT-vs-Pred scatter against the identity line, and the density
  overlays `test.py` writes, biggest error first.
- **Video** — point it at a recording (browse the files on this machine, or
  upload one) and it charts how crowded the pen gets over its length: the
  **flock count** per frame and the **max local density**, i.e. the birds inside
  the busiest `--window` patch of the density map, which is the pile-up signal a
  flat total count hides. `--report` picks which of the two the figure and the
  summary present — the whole-frame density, the local one, or both (the
  default); the CSV and the JSON always hold both, so it is a re-plot rather than
  a re-run, and the switch above the chart flips between whatever was reported.
  The chart fills in while the video is still being read,
  the x axis is wall-clock time when `--start-time` says when the recording
  started, and hovering reads both numbers off any moment. Summary cards give
  peak/mean and the time of the peak, and the busiest moments are saved as
  density overlays. `--save-frames overlay|plain` additionally saves *every*
  sampled frame (one per `--sample-seconds`) to `frames/` — `plain` is the frame
  as decoded (nothing drawn on it, so it can go straight to an annotator), and
  `--frame-width` keeps an all-day dump off the disk. An overlay is captioned
  with its clock time, count and peak — sized against the image as saved, so a
  downscaled frame stays readable — and when a `--mask-image` is in play the
  region the model was actually shown is outlined in red, the same frame
  `tools/inference_video.py` draws. When a run saved frames, clicking a point
  on the chart opens that moment's picture. The run also writes `timeline.csv`, `timeline.json` and a
  publication-ready `timeline.png` under `output/video_density/<video>/` —
  **Figure** opens the rendered one, **Save chart** exports what is on screen.
- **Annotations** — the ops from `webui/ops/` (one script per operation;
  live-project ones are prefixed **LS ·** in the picker) followed by the
  `tools/annotations/` file pipeline — merge Label Studio exports → convert to the training JSON →
  drop masked points → split train/val → export back to Label Studio. The
  `--project-id` field gets a dropdown of the projects on the running server.
  In **Density regions**, tick `--send-to-label-studio` to install the matching
  keypoint/region interface and import the generated region-count predictions
  directly into that project.
  **LS · import external annotations** accepts an LS JSON export from another
  machine, matches each task to the local image folder by filename, rewrites
  the foreign image paths, and adds the preserved annotations to matching
  existing project tasks. It creates a new task only when no existing image
  task matches (`--existing-only` disables new-task creation).
- **Data** — starts, stops and opens **Label Studio** and the **ngrok tunnel**,
  each with its own log streamed underneath. The commands match
  `tools/starter.sh label-studio [domain]` exactly (same port, data dir,
  local-files root) and `ngrok http --url=<domain> <port>`. Tick **Public link**
  to target the ngrok domain instead of localhost; that also exports
  `LABEL_STUDIO_HOST`/`CSRF_TRUSTED_ORIGINS` when Label Studio is started.

Both run as *services*, not runs: they do not occupy the one-run-at-a-time slot,
so annotating and training can happen together. Status comes from probing what
actually answers (the URL itself, and ngrok's inspector on :4040), not from what
this UI happens to own — a server you started in a shell shows as running, and
**Stop** works on it too. Before killing a process it did not start, the UI
checks that the command line really is that service, so an unrelated listener on
the port is reported and left alone rather than killed. Quitting the web UI
normally stops the services it started.

Both Label Studio URLs are overridable from `.env`:

```ini
LABEL_STUDIO_URL=http://localhost:8080                        # default: localhost:$LS_PORT (8080)
LABEL_STUDIO_PUBLIC_URL=https://your-domain.ngrok-free.app    # the "Public link" target
LABEL_STUDIO_API_KEY=<token>          # Account & Settings > Access Token; needed by webui/ops/
LABEL_STUDIO_PROJECT_ID=7             # optional default for --project-id
```

`webui/ops/` holds the scripts this UI drives, including the ones that talk to a
running Label Studio over its API — one file per operation, sharing the
`--project-id/--url/--api-key` flags from `_common.py`; see
[its README](webui/ops/README.md) for how to add one. The file-based pipeline
(exports in, training JSON out) stays in `tools/annotations/`. The region-mask
implementation lives in `webui/ops/region_mask_gui.py` but is not registered as
a separate WebUI picker entry.

The first op, `dedupe_annotations.py`, keeps exactly one annotation per task and
deletes the rest — Label Studio adds a second annotation on every re-submit, and
the exporters only ever read the first, so extras silently decide what gets
exported. `--keep most-points` (the default) keeps the most complete labelling
rather than the newest, because a stray empty re-submit is usually the newest
one; `newest`/`oldest` are available. A cancelled annotation never wins over a
real one.

**Deletion is permanent and Label Studio has no undo, so the script reports and
changes nothing unless you pass `--apply`** (the checkbox in the web UI).

Adding another script to the UI is one entry in `ENTRYPOINTS`
(`webui/schema.py`) — no `build_parser()` and no edits to the script itself,
because the extractor falls back to executing the file with `parse_args`
intercepted. Use `pythonpath` for a script that imports a sibling module
(`merge_ls.py` imports `convert_ls_to_coco`); every tool still runs with the
project root as its working directory, so paths you type mean the same thing on
every tab.

One run at a time; **Stop** kills the process together with its dataloader
workers. Every run is tee'd to `logs/webui/<run-id>.log` and the 50 most recent
are restored (metrics and all) when the server restarts, so history survives a
reboot of the UI.

> The UI runs commands on the machine that hosts it. Keep the default
> `127.0.0.1` bind unless you trust the network.

### Hyperparameter sweep

Edit the `GRID` and `COMMON_ARGS` blocks at the top of `tools/sweep.py`:

```python
GRID = {
    "lr": ["1e-5", "5e-5", "1e-4"],
    "waux": ["0.5", "1.0", "2.0"],
    "aux-sigma": ["1.5", "2.0", "3.0"],
}
```

Run:

```bash
python tools/sweep.py --dry-run         # preview commands first
python tools/sweep.py                   # actually run
python tools/summarize_sweep.py ../ckpts/sweep
```

Each combo lands in its own checkpoint subdirectory so the runs don't
overwrite each other; the script is **resume-safe** (re-running skips combos
that already produced a `best_*.pth`). Linux/macOS users can use the
equivalent `tools/sweep.sh` instead.

The sweep uses a smaller `--max-epoch` (200) than the final run (1000) — short
enough to be fast, long enough to differentiate configs. Once the leaderboard
picks a winner, **re-train it from scratch at `--max-epoch 1000`** for the
deployable model.

### Run on a single video

```bash
python tools/inference_video.py \
    -i ../data/clips/sample.mp4 \
    -m ../data/clips/sample_mask.png \
    --video-mode overlay
```

Modes: `overlay` (heatmap blended on original) / `pure` (heatmap only) /
`split` (original | heatmap side-by-side). The mask image marks pixels
to ignore (black = masked). Video output is written next to the input as
`<input>_<mode>.mp4`.

Heatmap colors use an **absolute** density scale (birds per model output cell),
not a per-frame min/max stretch, so the same color means the same density in
every frame and across clips — a quiet frame stays cool instead of having its
own peak painted red. Full scale is `HEATMAP_VMAX` in `utils.py` (0.12, measured
on the current checkpoint); override per run with `--vmax`. The runtime monitor
and `test.py` overlays read the same constant.

Hold-down behaviour: when the predicted count crosses
`--high-hold-seconds` (default 3 s) above the high-density threshold, the
contour color is locked to the high-density red, and any re-trigger inside
the window resets the timer.

To chart crowding over the *length* of a video instead of rendering one, use
`webui/ops/video_density_timeline.py` (the **Video** tab of the web UI):

```bash
python webui/ops/video_density_timeline.py     --video ../data/clips/sample.mp4     --start-time 04:00 --sample-seconds 1 --window 16
```

It samples one frame every `--sample-seconds`, reports the whole-frame count and
the integral of the busiest `--window`×`--window` cell patch (16 cells = 128 px,
about one bird across) — `--report {both,count,peak}` chooses which of the two
the figure and the summary show, `--save-frames overlay|plain` (+ `--frame-width`)
keeps every sampled frame instead of only the busiest few — and writes `timeline.csv`, `timeline.json`, a
publication-ready `timeline.png` and density overlays of the busiest moments to
`output/video_density/<video>/`. `--threshold`/`--threshold-metric` draw an alert
level on the matching series, and `--mask-image` blanks the same regions as
`tools/inference_video.py`. `timeline.png` needs matplotlib; without it the rest
is still written.

### Live multi-camera deployment

```bash
python run.py
```

This is the production loop:

1. `runtime/stream_capture.py` opens each RTSP stream in its own thread and
   writes frames into shared-memory buffers via a FREE→WRITING→READY→READING
   state machine.
1. `runtime/inference_process.py` batches READY frames across all cameras and
   runs the model.
1. `runtime/result_process.py` dispatches the per-frame counts to the
   registered handlers in `runtime/handlers/`:
   - `smart_plug.py` switches a TP-Link Tapo plug above the pile-up threshold.
   - `speaker.py` triggers a voice announcement (HTTP to a TTS service).
   - `monitor.py` reports a live dashboard line.
   - `sms_alarm/` runs the graded pile-up alarm and sends SMS (see below).

Configuration is pydantic-driven (`runtime/config.py`) and reads from
`topology.yaml` + env vars.

### Per-camera region masks

Each camera only counts a fixed slice of its frame; walkways, feeders, the
neighbouring pen and sky are masked out. Point `MASK_DIR` at a folder of PNGs
named by bare MAC — white = count, black = ignore:

```bash
MASK_DIR=../data/images/masks     # B8A44FD51C3C.png, one per camera
```

**This is not cosmetic.** The pile-up thresholds in `configs/alarm.json` were
calibrated on hard-black masked input, and the masks drop 29–67% of the frame
depending on the camera. Running unmasked over-counts by ~36% on average, which
on sampled frames turns 4 genuine threshold crossings into 9.

The mask is applied twice, for two different reasons
(`runtime/inferencer/_masks.py`):

| where                         | why                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| frame, *before* normalization | the model was trained on images with this region literally black, and black normalizes to `(0-mean)/std`, not to 0 — masking after would leave it looking at the ImageNet mean colour |
| density map, before the sum   | clamps what the receptive field bleeds across the boundary, and makes the density written to SHM (so the monitor heatmap) show the region the count actually came from                |

The density weights are an INTER_AREA downsample of the keep-mask, so a density
cell straddling the boundary is weighted by its overlap instead of rounded
wholly in or out. Everything expensive (PNG decode, resize) happens once at
inferencer startup; the hot path is one gather and one multiply per batch.

Streams with no matching mask count the full frame and are named in a startup
warning. Leave `MASK_DIR` empty to disable masking entirely.

### Camera identity: the bare MAC

Two per-camera assets can't be addressed by IP — the region mask and the
pile-up threshold — so both are keyed on the camera's **bare MAC** (12 hex
digits, no separators: `B8A44FD51C3C`). A MAC is burned into the unit, so it
survives re-IPing; an IP-keyed threshold would silently follow the wrong camera
after a DHCP change.

- **Camera mode** — list them in `camera_ids:` in `topology.yaml`, positionally
  aligned with `cameras:`. Entries are normalized, so `axis1/B8A44FD51C3C` and
  `b8:a4:4f:d5:1c:3c` are both accepted; anything that isn't a MAC fails the
  load rather than silently costing that camera its mask. To fill the list in,
  ask the cameras over VAPIX:

  ```bash
  python -m tools.discover_camera_ids            # paste-ready camera_ids: block
  python -m tools.discover_camera_ids --verify   # re-check the file against reality
  python -m tools.discover_camera_ids --write    # rewrite topology.yaml in place
  ```

  It also reports what each camera would actually get once mapped (a threshold,
  a mask, both or neither), and `--verify` exits non-zero when topology.yaml
  disagrees with a live camera — the failure that otherwise runs silently, with
  one camera on another's threshold. The runtime itself never queries a camera:
  discovery at startup would mean a unit that is rebooting loses its mask for
  the whole run, and an unmasked stream over-counts into false alarms.
  **Must run on the farm network.**

- **Video mode** — `camera_ids:` is ignored and the MAC is read from the clip
  filename (`B8A44FD51C3C.mkv`, or the delivery package's
  `.../axisN/axis-<MAC>/clip.mkv` layout).

`configs/alarm.json` keeps the vendor's `axisN/MAC` spelling — those are their
calibrated thresholds and the id is quoted in the SMS body — so `alarm/camera_ids.py`
adds the prefix back at that one boundary. Resolution is in
`runtime/camera_identity.py`; unresolvable streams degrade to "no coverage"
with a startup warning, never a crash.

### Pile-up SMS alarm

Off by default. Watches every mapped camera for a *sustained* threshold breach
and escalates over SMS, with an image of the pile-up attached:

```text
N >= T for 10s          -> Level 1   snapshot + SMS
still breaching + 3min  -> Level 2   snapshot + SMS
still breaching + 3min  -> Level 3   snapshot + SMS, then stop repeating
N < T for 30s           -> Recovery  notification, event closed and reset
```

Counts come from the same batched multi-stream inference pass as everything
else, so covering 20+ cameras costs one extra dict lookup per frame. A
directional-motion filter suppresses the common false positive where a dense
but *moving* flock crosses the frame: it tracks the density centroid and skips
the alarm when the drift is fast, far and consistent enough to be a walk-past
rather than a pile-up. It needs `centroid_x/y` on `InferenceResult`; until the
inferencer emits them the filter reports `missing_centroid` and the alarm falls
back to a pure count rule.

Three things to configure:

1. **Thresholds and rules** — `configs/alarm.json`. These are per-camera and
   independent of the `thresholds:` in `topology.yaml`, which drive
   speaker/smart-plug deterrence: different policy, different numbers.
1. **Camera mapping** — the stream's bare MAC (see above) selects its
   threshold. Streams that resolve to no MAC, or to one absent from
   `configs/alarm.json`, just get no alarm coverage; the handler logs exactly
   which ones at startup.
1. **SMS delivery** — the handler POSTs to the `farm-image-sms-service` API
   (`POST /api/projects/<slug>/images`), which stores the snapshot, produces a
   public link and texts it out.

```bash
# dry run: real payloads, snapshots and logs written to output/alarm/,
# nothing actually sent
ENABLE_SMS_ALARM=1 python run.py

# live SMS
ENABLE_SMS_ALARM=1 SMS_ALARM_REAL_WORKER=1 FARM_SMS_API_KEY=... python run.py
```

Evidence lands under `output/alarm/`: `events.jsonl` and
`notifications.jsonl` at the top level, plus one directory per event holding
`counts.csv`, `pre_window_counts.csv`, the snapshots and `summary.json`. The
layout matches the `chicken_alarm_delivery` package, so its offline HTML report
tooling reads these runs unchanged.

Blocking work — evidence writes, JPEG encoding and the SMS POST (a 20s-timeout
`urlopen`) — runs on background threads. Only the state machine itself is inline
in the consumer loop, since its timing is measured against sample timestamps.

### ONNX export + TensorRT INT8

```bash
# 1. Export trained .pth to ONNX with dynamic batch / spatial axes.
python -m tools.export_onnx \
    --ckpt ../ckpts/best.pth \
    --out ../ckpts/best.onnx \
    --check

# 2. Build a TensorRT INT8 calibration cache from the val set.
python -m tools.calibrate_trt_int8 \
    --onnx ../ckpts/best.onnx \
    --cache ../ckpts/best.int8.cache \
    --data-dir ../data --split val \
    --batch-size 64 --max-batches 200

# 3. Benchmark all four runtimes (FP32 / Fused-AMP / TRT-FP16 / TRT-INT8).
python -m tools.test_inference_time \
    --onnx ../ckpts/best.onnx \
    --int8-cache ../ckpts/best.int8.cache \
    --batch-size 64 --h 720 --w 1080
```

INT8 calibration takes ~1 minute on the val set and is reusable forever (the
cache is written once and reloaded on subsequent ORT sessions).

______________________________________________________________________

## Design notes

A few things worth knowing before changing code:

### Output stride is 8

The model outputs density at `H/8 × W/8`. The dataset's discrete-count
target is sum-pooled to the same resolution. The single source of truth is
`datasets.DOWNSAMPLE_RATIO`; trainer, dataset, and `DMCountLoss` all import
from there. Do not hardcode `8` in new code — re-import.

### U-Net skip — input must be divisible by 8 (not 16)

The model concatenates stage2 (1/8) with size-matched-upsampled stage3 (1/16
→ 1/8). Earlier versions used `scale_factor=2` which silently broke when
input H or W was divisible by 8 but not 16 (off-by-one rounding made
`upsample(stage3) ≠ stage2`). The current implementation uses
`F.interpolate(f3, size=f2.shape[-2:], ...)` so any input divisible by 8
works. Val transform's `PadToMultiple(8)` ensures native-resolution val
frames satisfy this.

### DMCountLoss: OT is averaged over the batch

`OTLoss` accumulates per-sample contributions, so `DMCountLoss` divides by
batch size before applying `wot`. This means **`wot` is portable across
batch sizes** — you can sweep `wot` independently of `--batch-size`.

### EMA evaluation, EMA save

Validation always runs on the EMA shadow copy of the model
(`models.ema.ModelEMA`). The "best" `.pth` saved at validation time is the
EMA state, not the live training state. So `MODEL_PATH` and the inference
runtime are always loading EMA weights. The `_ckpt.tar` per-epoch checkpoint
contains both for resumability.

### Auxiliary density loss

`DMCountLoss` adds a per-pixel L1 between the raw prediction and a
Gaussian-smoothed (σ=2 in H/8 px ≈ chicken-sized) GT. This term is the only
thing supervising **empty regions** — the OT and TV terms are silent there
because they're scaled by `gd_count`. Without the aux term, the model could
hallucinate density on empty floor and only the global count would push back.

### Best-model rotation

Each new validation best is saved as `best_ep<NNNN>_mae<X>_mse<Y>.pth`. The
filename encodes the epoch and metrics so you can read the leaderboard
without loading any tensors. `--max-best-ckpts` (default 5) caps disk use:
older bests are deleted as new ones come in. `tools/summarize_sweep.py` uses
this filename convention to roll up sweep results.

______________________________________________________________________

## Common gotchas

| Symptom                                           | Likely cause                                                               | Fix                                                                                                   |
| ------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `crop_size N not divisible by downsample_ratio 8` | Picked an odd `--crop-size`                                                | Pick a multiple of 8 (default 512 is fine)                                                            |
| Val-time crash in `gen_downsampled_density`       | Val image isn't multiple of 8                                              | Already fixed in `build_val_transform` (PadToMultiple) — make sure you didn't pass a custom transform |
| Concat shape mismatch in forward                  | Custom backbone or input not div by 8                                      | Use the size-matched interpolate (already the default)                                                |
| `KeyError: model_state_dict` on resume            | Loading a `.pth` (best file) where you should load the `.tar` (full state) | `.pth` is model-only; for resume use the `.tar` from the same run dir                                 |
| `FutureWarning: weights_only`                     | Old PyTorch path                                                           | Already silenced via explicit `weights_only=False`; if you see it elsewhere, do the same              |
| Training collapses to predict ≈ 0                 | LR too high / dataset has many empty crops without aux                     | Lower `--lr`, ensure `--waux > 0` and `--aux-sigma > 0`                                               |
| OT loss explodes when batch changes               | (Pre-fix) — should not happen anymore                                      | OT is now batch-averaged in `DMCountLoss`                                                             |
| Loss/F1 looks good in sweep but bad on final      | Sweep used different `--max-epoch` so cosine schedule differs              | Re-tune `--lr` once at the final epoch budget                                                         |

______________________________________________________________________

## Where to read next, in order of "you'll want this first"

1. `train.py` — the CLI surface for training (start with `--help`).
1. `trainer.py` — `Trainer.setup()` and `Trainer.train_epoch()` are the core loop.
1. `losses/dm_count.py` — what the loss actually computes.
1. `models/shufflenet.py` — the architecture and skip connection.
1. `datasets/bird.py` — sample dict shape that everything else expects.
1. `runtime/inference_process.py` — production inference loop.
