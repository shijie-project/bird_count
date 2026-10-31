# Training and tools

## Data layout

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

## Training

```bash
python -m birdcount.training.train                       # the production recipe, ~16 min on an RTX 5090
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
`python -m birdcount.deploy --check`. `--render` videos are H.264 MP4 that
play in any browser, encoded on the GPU's video engine when there is one
(libx264 otherwise).

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
