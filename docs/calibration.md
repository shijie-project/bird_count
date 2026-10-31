# Threshold calibration

```bash
python -m birdcount.deploy.calibrate rate --dry-run   # what would be sent, and the tokens it takes
python -m birdcount.deploy.calibrate run              # rate new snapshots, then suggest thresholds
```

A vision model (`CALIBRATION_MODEL`, key `ANTHROPIC_API_KEY`) rates every
sampled snapshot and alarm picture on a fixed five-level density scale
(`calibrate/rubric.py`: sparse, moderate, dense, full, crammed; density only,
whether birds pile up is the count threshold's call), with the
area outside the camera's mask darkened to 15 %. Each camera's 10-bird count bin is
rated until it holds a week of snapshots (21), ~4000 pictures for a first
week, under a dollar with Haiku on the Batch API (rate nightly, or once
after the week: the cost is the same). `propose` then finds, per camera, the count at
which half its pictures are rated crammed (level 5, no floor showing and no
room to move; a full shed with small gaps is level 4) by a monotone
fit over the bins, and writes `CALIBRATION_DIR/latest.json`: the suggestion, how often the
count misjudges a picture at the current and the suggested threshold, how
many alarm pictures were rated crammed. The Thresholds page puts your alarm
checks next to it: real pile-ups and false alarms per camera. Suggestions only: `configs/alarm.json` is never changed. Run it
nightly (Windows Task Scheduler; the module's docstring has the command).
