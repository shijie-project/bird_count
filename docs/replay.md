# Replaying recordings

```bash
python -m birdcount.deploy.replay G:/ --list                        # cameras and days in the recorder export
python -m birdcount.deploy.replay G:/ --camera B8A44FD51C3C         # every day of one camera (--days to pick)
python -m birdcount.deploy.webui --replay outputs/replay/full       # the farm backend over the replay
```

Recorded days instead of waiting for the farm's: each camera-day of the AXIS
recorder's export goes through the deployed pipeline (model, mask, FPS, alarm
engine, count logger, evidence; nothing is sent) on the recordings' own
clock, from the clips' names, into `OUTPUT_ROOT/replay/<run>/` laid out like
the runtime's files, so every page of the farm backend works on it. Each
camera-day also keeps every sample as counted (`samples/<MAC>/<day>.npz`:
time, count, centroid, alarm level, clip and frame) and a summary
(`summary.csv`, one row per camera-day: hours, count percentiles, seconds
over 60 / 80 / 100 % of the threshold, alarms), for re-running the alarm with
other thresholds and picking the clips worth counting again without decoding.
Days already done are skipped, so a stopped replay continues. A run holds one
model, frame size and FPS; another needs another `--run`. About 900 samples/s
with 5 camera-days at once (CPU decoding and the GPU both near full; the clips
are copied off the USB disk one at a time, see `replay/cache.py`): ~80 min
per camera for 19 days of 05:00–19:00.

```bash
python -m birdcount.deploy.replay.threshold B8A44FD51C3C --try 90..180   # alarms per threshold, scored by your checks
python -m birdcount.deploy.replay.threshold B8A44FD51C3C --set 100       # the run's alarms at 100, pictures and all
```

Other thresholds are tried on what was counted, without counting again: the
alarm engine decides on the kept samples as it did live (a minute for 19
days). `--try` lists each threshold's Level 1 alarms, scored against the
verdicts given on **Check alarms**: an alarm overlapping in time one judged a
real pile-up counts as real, so checking the alarms of the lowest threshold
worth considering scores every higher one. `--set` makes a threshold the
camera's in the run (`thresholds.json`): its evidence with the alarm pictures
decoded again, the count log's alarm levels, the summary. Each threshold keeps
its evidence and verdicts. `configs/alarm.json` is never changed.

```bash
python -m birdcount.deploy.replay.snapshots B8A44FD51C3C                 # DATA_ROOT/snapshots/B8A44FD51C3C/
COUNT_LOG_DIR=outputs/replay/full/count_log ALARM_OUTPUT_DIR=outputs/replay/full/alarm CALIBRATION_DIR=outputs/replay/full/calibration python -m birdcount.deploy.calibrate run --days 19
```

What is kept of the recordings is pictures, not video: per camera and day, 3
moments per 10-bird bin from 40 up, at least 20 min apart, and each peak of
the 2 s median from 80 up (~650 for 19 days of a camera, ~10 min). The frames
as counted (1280×720, unmasked) go to `DATA_ROOT/snapshots/<MAC>/<day>/` with
an `index.csv` (time, count, 2 s median, clip and frame, model and mask), for
counting again under another mask or model, rating and labelling; review-size
copies replace the run's count-log snapshots, so the farm backend shows them
and `deploy.calibrate` rates them with the alarm pictures. With
`--events-at T` every alarm event at threshold T is also pictured once a
second, from 30 s before its Level 1 to its recovery; rated (`calibrate rate
--per-bin` large enough), they let `replay.threshold --try` score each
threshold by the stretches the model saw as crammed (level 5 for 80 % of a
10 s window: its level flickers between 4 and 5 on one scene): how many its
alarms catch, miss, and how many alarms hold none.

```bash
python -m birdcount.deploy.tools.stills                         # one still per camera now: DATA_ROOT/stills/<time>/
python -m birdcount.deploy.replay.alignment B8A44FD51C3C          # does the mask fit every replayed day, and today?
```

A mask is pixels: a camera bumped, re-aimed or zoomed since it was drawn
counts the wrong part of the shed. `replay.alignment` registers each replayed
day's emptiest pictures and every lit still of the camera against the picture
the mask was drawn on (the Masks page keeps it in `MASK_DIR/_reference/`;
else the first day), on the fixed fittings, and flags any off by more than
6 px, 0.5° or 1 % (`alignment.csv`, `alignment.jpg` with the mask's edge
drawn on). Run it before trusting a camera's replayed counts; stills taken
with the lights off are skipped.
