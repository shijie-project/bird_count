"""The density model: network definition, checkpoint I/O, inference preparation.

Importing this package imports torch (and nothing heavier: no torchvision,
no cv2).
"""

from birdcount.core.model.loading import build_model, load_inference_model, prepare_inference_model, read_state_dict
from birdcount.core.model.network import OUTPUT_STRIDE, DensityNet, InputOrder


__all__ = [
    "OUTPUT_STRIDE",
    "DensityNet",
    "InputOrder",
    "build_model",
    "load_inference_model",
    "prepare_inference_model",
    "read_state_dict",
]
