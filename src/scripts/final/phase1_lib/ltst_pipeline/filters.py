from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def highpass_filter(x, fs: float, cutoff: float = 0.67, order: int = 3):
    """
    Zero-phase Butterworth high-pass filter (default 0.67 Hz).
    Accepts 1D or 2D array; preserves shape.
    """
    x = np.asarray(x, dtype=float)
    b, a = butter(order, cutoff / (0.5 * fs), btype="highpass")
    return filtfilt(b, a, x, axis=0)
