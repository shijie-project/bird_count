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

## Tests

```bash
pytest                                   # everything
pytest -m "not gpu and not slow"         # what the pre-commit hook runs
```

`tests/test_layering.py` enforces the package structure (dependency direction,
private modules) and `tests/test_import_cost.py` keeps light modules from
loading torch / cv2 / numpy.
