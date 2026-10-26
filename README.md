# birdcount

Density-map bird counting for poultry pile-up detection: a live multi-camera
deployment (`birdcount.deploy`) and the training / evaluation / labelling
tooling that produces its model (`birdcount.training`), on a shared core
(`birdcount.core`).

## Setup

```bash
# 1. PyTorch built for your CUDA driver (pip would otherwise pick the CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. This package, editable, with dev tools
pip install -e ".[dev]"
pre-commit install

# 3. Configuration
cp .env.example .env      # then edit; .env is git-ignored
```

All relative paths in `.env` are resolved against the project root, so
commands behave the same from any working directory.

## Live runtime

```bash
pip install -e ".[deploy]"
python -m birdcount.deploy --check     # what every camera gets: zone, devices, mask, threshold
python -m birdcount.deploy             # run (also: birdcount-deploy)
```

The farm layout is `configs/topology.yaml`, alarm thresholds and rules
`configs/alarm.json`. Stop with the operator panel's Terminate button, Ctrl+C
or by closing the console: all three switch the devices off and finish the
recordings first. Tools:

```bash
python -m birdcount.deploy.tools.discover_ids --write   # record camera MACs (on the farm network)
python -m birdcount.deploy.tools.mask_editor            # draw each camera's region mask
python -m birdcount.deploy.tools.devices plug status    # switch speakers / plugs by hand
```

## Training and evaluation

```bash
python -m birdcount.training.train                          # the production recipe, ~12 min on an RTX 5090
python -m birdcount.training.train --resume outputs/train/<run>
python -m birdcount.training.train.sweep --grid lr=1e-5,5e-5 -- --epochs 300
python -m birdcount.training.eval --split val --ckpt <best_*.pth>
```

A run folder holds `config.json`, `metrics.jsonl` (one line per epoch),
`summary.json`, `train.log`, the kept `best_*.pth` and `last.tar`.
Validation scores the EMA weights exactly as deployed (720p, region masks,
fused fp16).

## Video analysis

```bash
pip install -e ".[training]"
python -m birdcount.training.video ../data/videos/<clip>.mkv          # count + peak density, one sample a second
python -m birdcount.training.video <clip>.mkv --every 0 --render overlay   # full-rate overlay video
python -m birdcount.training.video ../data/videos --frames boxes --count-threshold 140
```

Counts exactly as deployed: 720p, the camera's region mask by the MAC in the
filename, fused fp16. Each video gets OUTPUT_ROOT/video/<clip>/<time>/ with
`samples.jsonl` (written as the run goes), `timeline.csv`, `summary.json`,
`timeline.png` and overlays of the busiest moments in `peaks/`. A camera's alarm
threshold, to pass as `--count-threshold`, is listed by `python -m birdcount.deploy --check`.

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
only report unless given `--apply`.

## Control panel

```bash
python -m birdcount.training.webui          # http://127.0.0.1:8420
```

A local web page for everything above: forms generated from each tool's own
options, live training curves, evaluation tables and overlays, video
timelines, a dataset browser (annotated points, and an evaluation's prediction
per image), and Label Studio with its ngrok tunnel (`LABEL_STUDIO_DATA_DIR`,
`LABEL_STUDIO_PUBLIC_URL`). Runs land in OUTPUT_ROOT exactly as from a shell,
and runs started from a shell are listed too. Stop asks a run to stop as
Ctrl+C would, so training keeps its state for `--resume`.

## Tests

```bash
pytest                                   # everything
pytest -m "not gpu and not slow"         # what the pre-commit hook runs
```

`tests/test_layering.py` enforces the package structure (dependency direction,
private modules) and `tests/test_import_cost.py` keeps light modules from
loading torch / cv2 / numpy.
