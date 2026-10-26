"""birdcount — density-map bird counting for poultry pile-up detection.

Three subpackages, with a one-way dependency rule enforced by
`tests/test_layering.py`:

    core       shared by both apps; imports nothing from them
    deploy     the live multi-camera runtime        -> core only
    training   training, evaluation, labelling     -> core only

This module stays empty on purpose: importing `birdcount` must not drag in
torch, cv2 or any subpackage (see `tests/test_import_cost.py`).
"""

__version__ = "3.0.0.dev0"
