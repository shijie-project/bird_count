import logging
import sys

import torch


logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """
    Configure specific logging format for industrial monitoring.
    Includes process name/ID to debug multi-process issues.
    """
    level = logging.DEBUG if debug else logging.INFO
    log_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | PID:%(process)-5d | %(name)-35s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        stream=sys.stdout,
    )

    # Suppress noisy libraries
    for logger_name in ["urllib3", "PIL", "matplotlib", "socket"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_cuda() -> None:
    """
    Configures global CUDA and cuDNN settings for maximum throughput.
    Includes TF32 optimization for Ampere+ architectures.
    """
    if not torch.cuda.is_available():
        return

    # Enable cuDNN auto-tuner to find the best algorithms for the current hardware
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
