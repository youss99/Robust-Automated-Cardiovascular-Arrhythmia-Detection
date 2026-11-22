from __future__ import annotations

from typing import Optional, List, Dict
import numpy as np
import pandas as pd

try:
    import neurokit2 as nk
except Exception:
    nk = None


def _waves_get_abs(waves, key: str, i_local: int, seg_start: int):
    """Safely fetch waves[key][i_local] and convert to absolute index, else NaN."""
    arr = waves.get(key)
    if arr is None or i_local >= len(arr):
        return np.nan
    val = arr[i_local]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    return int(val) + int(seg_start)


def delineate_long_ecg_chunked(
    lead,
    fs: int = 250,
    W_s: float = 20.0,
    H_s: Optional[float] = None,
    M_s: float = 3.0,
    method: str = "dwt",
    rpeak_method: str = "neurokit",
    out_csv: Optional[str] = None,
    record_name: Optional[str] = None,
    lead_id: int = 0,
    stream_every: int = 5000,
) -> pd.DataFrame:
    """
    Chunked & margin-padded delineation.
    - Detect R-peaks once globally
    - For each window, delineate with margin context
    - Keep beats in a non-overlapping bin to avoid duplicates
    - Optionally stream results to out_csv
    """
    if nk is None:
        raise ImportError("neurokit2 is not available. Install it or ensure it imports correctly.")

    lead = np.asarray(lead, dtype=float).reshape(-1)
    N = len(lead)
    if N == 0:
        return pd.DataFrame()

    W = int(round(W_s * fs))
    H = int(round((W_s / 2.0 if H_s is None else H_s) * fs))
    M = int(round(M_s * fs))

    lead_hp = lead  # already filtered upstream if desired

    # global R-peaks
    _, info_full = nk.ecg_peaks(lead_hp, sampling_rate=fs, method=rpeak_method)
    R_abs = np.asarray(info_full["ECG_R_Peaks"], dtype=int)

    '''_, info_full = nk.ecg_peaks(lead_hp, sampling_rate=fs, method=rpeak_method)
    R_abs = np.asarray(info_full["ECG_R_Peaks"], dtype=int)

    # NEW: refine R-peaks before window loop
    R_abs = refine_rpeaks_local_extreme(
        signal=lead_hp,
        rpeaks=R_abs,
        fs=fs,
        search_ms=80.0,  # +/- 20 samples at 250 Hz
        mode="abs",  # or "abs" if you want robust polarity handling
        neighbor_guard=True
    )'''

    if R_abs.size == 0:
        return pd.DataFrame()

    keys = [
        "ECG_P_Peaks",
        "ECG_P_Onsets",
        "ECG_P_Offsets",
        "ECG_Q_Peaks",
        "ECG_R_Onsets",
        "ECG_R_Offsets",
        "ECG_S_Peaks",
        "ECG_T_Peaks",
        "ECG_T_Onsets",
        "ECG_T_Offsets",
    ]

    rows: List[Dict] = []
    wrote_header = False

    for S in range(0, N, H):
        seg_start = max(0, S - M)
        seg_end = min(N, S + W + M)
        segment = lead_hp[seg_start:seg_end]

        mask_seg = (R_abs >= seg_start) & (R_abs < seg_end)
        r_seg_abs = R_abs[mask_seg]
        if r_seg_abs.size == 0:
            continue
        r_seg_rel = r_seg_abs - seg_start

        _, waves = nk.ecg_delineate(
            segment, rpeaks=r_seg_rel, sampling_rate=fs, method=method, show=False
        )

        bin_right = min(S + H, N)
        keep_mask = (r_seg_abs >= S) & (r_seg_abs < bin_right)
        idx_keep = np.where(keep_mask)[0]
        if idx_keep.size == 0:
            continue

        for i_local, r_abs in zip(idx_keep, r_seg_abs[idx_keep]):
            row = {
                "record_name": record_name,
                "lead": lead_id,
                "fs": fs,
                "win_start": S,
                "win_end": min(S + W, N),
                "r_peak": int(r_abs),
            }
            for k in keys:
                row[k] = _waves_get_abs(waves, k, int(i_local), seg_start)

            qrs_off = row["ECG_R_Offsets"]
            row["J_point"] = qrs_off
            row["J_plus_60ms"] = (
                np.nan if np.isnan(qrs_off) else int(qrs_off + round(0.060 * fs))
            )

            rows.append(row)

        if out_csv and len(rows) >= stream_every:
            pd.DataFrame(rows).to_csv(
                out_csv,
                index=False,
                mode=("a" if wrote_header else "w"),
                header=(not wrote_header),
            )
            rows.clear()
            wrote_header = True

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(
            out_csv,
            index=False,
            mode=("a" if wrote_header else "w"),
            header=(not wrote_header),
        )
    return df


import numpy as np
from typing import Literal, Optional

PeakMode = Literal["max", "min", "abs"]

def refine_rpeaks_local_extreme(
    signal: np.ndarray,
    rpeaks: np.ndarray,
    fs: int = 250,
    search_ms: float = 80.0,
    mode: PeakMode = "max",
    neighbor_guard: bool = True,
) -> np.ndarray:
    """
    Refine each rpeak by searching for a local extreme around it.

    Parameters
    ----------
    signal : 1D array
        ECG lead signal used for r-peak detection (ideally the same one).
    rpeaks : 1D int array
        Initial rpeak indices (sorted).
    fs : int
        Sampling frequency.
    search_ms : float
        Half-window size in milliseconds on each side.
        Example: 80ms -> +/-20 samples at 250Hz.
    mode : {"max","min","abs"}
        Which extreme to snap to within the search window.
    neighbor_guard : bool
        If True, restricts each search window not to cross halfway to neighboring peaks
        to avoid drifting into adjacent beats.

    Returns
    -------
    refined : 1D int array
        Refined rpeak indices (sorted, unique).
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    r = np.asarray(rpeaks, dtype=int)
    if r.size == 0:
        return r
    r = np.sort(r)

    half_w = int(round((search_ms / 1000.0) * fs))
    n = x.size

    refined = np.empty_like(r)

    for i, rp in enumerate(r):
        lo = max(0, rp - half_w)
        hi = min(n, rp + half_w + 1)

        if neighbor_guard:
            # cap the window to the midpoints between neighboring peaks
            if i > 0:
                mid_left = (r[i - 1] + rp) // 2
                lo = max(lo, mid_left)
            if i < r.size - 1:
                mid_right = (rp + r[i + 1]) // 2
                hi = min(hi, mid_right + 1)

        if hi <= lo + 1:
            refined[i] = rp
            continue

        seg = x[lo:hi]

        if mode == "max":
            k = int(np.argmax(seg))
        elif mode == "min":
            k = int(np.argmin(seg))
        else:  # "abs"
            k = int(np.argmax(np.abs(seg)))

        refined[i] = lo + k

    # enforce strictly increasing / uniqueness
    # if refinement causes duplicates or non-increasing, fall back to original at that position
    out = refined.copy()
    out[0] = int(np.clip(out[0], 0, n - 1))
    for i in range(1, out.size):
        out[i] = int(np.clip(out[i], 0, n - 1))
        if out[i] <= out[i - 1]:
            # fallback to original if it restores order; else minimally bump
            cand = r[i]
            if cand > out[i - 1]:
                out[i] = cand
            else:
                out[i] = out[i - 1] + 1

    # clip again and unique
    out = out[(out >= 0) & (out < n)]
    out = np.unique(out)

    return out


