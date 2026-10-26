"""What inference hands the consumer: one record per frame, a batch per array.

A numpy structured array crosses the process boundary as one small buffer
(48 bytes a frame), instead of a pickled object per frame. numpy only.
"""

from __future__ import annotations

import numpy as np


RESULT_DTYPE = np.dtype(
    [
        ("sid", np.int32),
        ("slot", np.int32),  # ring slot holding the frame (and its density) until released
        ("frame_idx", np.int64),
        ("captured", np.float64),  # time.time() when the grabber published the frame
        ("inferred", np.float64),  # time.time() when the count was ready
        ("count", np.float32),
        ("cx", np.float32),  # density centroid in 0..1 of the frame; NaN when empty
        ("cy", np.float32),
    ],
    align=True,
)


def empty_batch(n: int) -> np.ndarray:
    return np.zeros(n, dtype=RESULT_DTYPE)
