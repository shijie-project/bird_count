# birdcount

Density-map bird counting for poultry pile-up detection. One model counts every
camera of the farm's shed ten times a second; the live runtime turns those
counts into pile-up alarms (SMS, zone speakers and smart plugs), and a local web
backend keeps the farm, its history and the model's ratings in view. Everything —
training validation, evaluation, video analysis, replay and the live runtime —
counts through the same inference path, so every number agrees.

```
cameras ─▶ grabber (decode, 10 fps) ─▶ shared-memory ring ─▶ inferencer (mask, fp16, density sum)
        ─▶ count per camera ─▶ alarm rules (hysteresis, evidence) ─▶ SMS, speakers, smart plugs
```

## Highlights

- **Counting, not detection.** A ShuffleNetV2 backbone with a U-Net-style skip
  head regresses a density map at 1/8 resolution; the count is the sum of that
  map inside the camera's region mask, so dense flocks stay countable where
  boxes would merge. Trained with the DM-Count losses (OT + count + TV) plus a
  Gaussian-density auxiliary loss.
- **One inference path.** Validation during training, evaluation, video
  analysis, replay and the live runtime all run `core.inference.DensityPredictor`
  on the same prepared checkpoint (Conv+BN fused, fp16 on CUDA), so a number from
  any of them is the number the alarm saw.
- **Live multi-camera runtime.** A grabber decodes every camera at `FPS` into a
  shared-memory frame ring, a batched inferencer counts them on the GPU, and the
  main process runs the alarm, the devices and the panels without ever loading
  torch. `--check` validates the whole configuration before a run.
- **Alarms that do not flap.** A camera alarms after 10 s above its threshold
  (median of the last 2 s), escalates at 3-minute steps and recovers only below
  0.9× the threshold for 30 s; a directional-motion filter skips dense flocks
  walking across the frame. Rules and per-camera thresholds live in
  `configs/alarm.json`.
- **Thresholds fitted to the farm.** A vision model rates sampled snapshots on a
  five-level density rubric, and each camera's threshold is suggested from where
  half its pictures read "crammed". Recorded days can be replayed through the
  runtime to try other thresholds against your own alarm checks.
- **A farm backend.** Every camera on the shed's floor plan, a camera's day with
  its count curve, alarm events and snapshots, and pages to check alarms, review
  model ratings and redraw masks. It reads only the files the runtime writes, so
  history stays available while the runtime is down.

## Results

The production checkpoint on the validation split — 78 frames, 5,402 birds —
scored exactly as deployed (1280×720, per-camera region masks, fp16, fused EMA
weights):

| Metric | Result |
|---|---|
| **Counting accuracy** — share of all birds counted right (1 − Σ error / Σ birds) | **95.3 %** |
| **Acc@(5, 10 %)** — frames within 5 birds or 10 %, whichever is looser | **88.5 %** |
| MAE / RMSE | 3.28 / 4.97 birds per frame |
| Pile-up detection at a 100-bird threshold — precision / recall / F1 | 0.85 / 1.00 / 0.92 |

The model is trained on 402 labelled frames (324 train / 78 val) in 1,000 epochs
of 512-pixel crops — about 16 minutes on an RTX 5090. Reproduce the table with
`python -m birdcount.training.eval --split val` (it evaluates `MODEL_PATH`, or
the checkpoint given with `--ckpt <run>/best_*.pth`).

## Repository layout

```
birdcount/
├── core/        settings, paths, the model, the inference pipeline, masks, frames
├── deploy/      live runtime (grabber, inferencer, alarm, handlers, monitor),
│                farm backend (webui), threshold calibration, replay tools
└── training/    data, train (and sweeps), eval, video, labelling, control panel
configs/         topology.yaml (the farm) and alarm.json (thresholds and rules)
docs/            the guides linked below
tests/           mirrors the package; pytest
```

`deploy` and `training` both build on `core` and never import each other;
`tests/test_layering.py` keeps it that way.

## Getting started

Python 3.12+ and a CUDA GPU for live counting (the tools also run on CPU, at
fp32 and slower). PyTorch is deliberately not a dependency: install the build
matching the machine's driver first.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -e ".[deploy]"              # the live runtime and farm backend
pip install -e ".[training]"            # training, evaluation, video, labelling
pip install -e ".[dev]"                 # tests, ruff, pre-commit
pre-commit install

cp .env.example .env                    # then edit; .env is git-ignored
```

`.env.example` is the catalogue: every key with what it does. A key in `.env`
that no setting reads is reported at startup (a typo would otherwise silently
do nothing); process environment variables override `.env`, and relative paths
are resolved against the project root. The data layout (`DATA_ROOT`,
`MODEL_PATH`, `OUTPUT_ROOT`) is in
[Training and tools](docs/training.md#data-layout).

## Run

```bash
python -m birdcount.deploy --check      # validate settings, topology, model and masks; list each camera's setup
python -m birdcount.deploy              # count the cameras and raise alarms (birdcount-deploy)
python -m birdcount.deploy.webui        # the farm backend, http://127.0.0.1:8430 (birdcount-farm)
```

```bash
python -m birdcount.training.train      # the production recipe
python -m birdcount.training.eval --split val
python -m birdcount.training.webui      # the training control panel, http://127.0.0.1:8420
```

The runtime and the backend are separate processes: the backend reads only the
files the runtime writes, so it shows the history while the runtime is down and
the live state within a second or two while it runs.

## Documentation

| Guide | Covers |
|---|---|
| [Live deployment](docs/deployment.md) | processes, cameras and zones, region masks, alarm rules and evidence, records, devices, the farm backend |
| [Threshold calibration](docs/calibration.md) | vision-model ratings, how a threshold is suggested, the nightly run |
| [Replaying recordings](docs/replay.md) | replaying recorded days, trying thresholds against your checks, keeping snapshots, mask alignment |
| [Training and tools](docs/training.md) | data layout, training and sweeps, evaluation, video analysis, labelling, the control panel |
| [Design notes](docs/design-notes.md) | the invariants that are easy to break without noticing |
| [Migrating from v2](docs/migrating-from-v2.md) | old commands, settings and outputs mapped to v3 |

## Development

```bash
pytest                                   # everything
pytest -m "not gpu and not slow"         # what the pre-commit hook runs
```

`tests/test_layering.py` enforces the package structure (dependency direction,
private modules, the control panel as a leaf) and `tests/test_import_cost.py`
keeps light modules from loading torch / cv2 / numpy. Ruff formats and lints
(`pre-commit run -a`).
