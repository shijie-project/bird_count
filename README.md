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

## Tests

```bash
pytest                                   # everything
pytest -m "not gpu and not slow"         # what the pre-commit hook runs
```

`tests/test_layering.py` enforces the package structure (dependency direction,
private modules) and `tests/test_import_cost.py` keeps light modules from
loading torch / cv2 / numpy.
