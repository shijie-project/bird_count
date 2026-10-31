# Migrating from v2

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
