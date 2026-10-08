from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk


# ------------------------------------------------
# Defaults / Config
# ------------------------------------------------
FS_DEFAULT = 250
BASE_PATH_DEFAULT = "ltst_records"

LTSTDB_RECORDS_DEFAULT = [
    "s20011","s20051","s20061","s20071",
    "s20081","s20091","s20101","s20111",
    "s20121","s20131","s20141","s20201",
    "s20211","s20221","s20231","s20241"
]

USE_MANUAL_OMIT_DEFAULT = True
MANUAL_OMIT = {
    ("s20091", "II"),
    ("s20121", "II"),
    ("s20081", "V2"),
    ("s20121", "V2"),
    ("s20221", "V2"),
    ("s20241", "V2"),
}

STTAG_TO_PAPER_LEAD = {"ST0": "II", "ST1": "V2"}
STTAG_TO_LEAD_IDX   = {"ST0": 0, "ST1": 1}  # assumes WFDB signal columns align with ST0/ST1


# ------------------------------------------------
# FSM state machine
# ------------------------------------------------
class S(Enum):
    REST     = auto()
    WU_01    = auto()   # +0.1 mV · 10 min
    WU_02    = auto()   # +0.2 mV · 1 min
    WU_NEG   = auto()   # –0.1 mV · 1 min
    ALARM    = auto()
    COOLDOWN = auto()

T_WU_01, T_WU_02, T_WU_NEG, T_CD = 600, 60, 60, 30  # seconds


@dataclass
class Alarm:
    record: str
    lead_tag: str         # "ST0" / "ST1"
    lead_paper: str       # "II" / "V2"
    onset_breach: int
    onset_expected: int
    onset_actual: int
    end_expected: int
    end_actual: int
    crit: str
    strategy: str = ""    # ST deviation strategy used


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def extract_st_arrays_from_ann(st_ann, lead_prefix: str) -> tuple[list[int], list[float]]:
    """Extract (sample_idx, st_mV) from the '16a' aux_note for ST0/ST1 entries."""
    samp_arr: list[int] = []
    st_arr: list[float] = []

    for samp, aux in zip(st_ann.sample, st_ann.aux_note):
        if not aux:
            continue
        body = aux[3:] if aux.startswith("DMY") else aux  # tolerate DMY-prefixed (manually corrected) beats
        if not body.startswith(lead_prefix):
            continue

        rest = body[len(lead_prefix):]
        if rest.startswith(":") or rest.startswith(","):
            rest = rest[1:]

        fields = rest.split(",")
        if len(fields) > 4:
            try:
                st_mv = float(fields[4]) / 1000.0  # µV -> mV
            except Exception:
                continue

            samp_arr.append(int(samp))
            st_arr.append(float(st_mv))

    return samp_arr, st_arr

def extract_ee_and_j60_location_from_ann(st_ann, lead_prefix: str, fs: float):
    """
    Extract:
      - annotation sample
      - ee value (J+60 amplitude, in mV)
      - absolute J+60 sample index
      - absolute J+60 time in seconds
    """
    samp_arr = []
    ee_mV_arr = []
    j60_sample_arr = []
    j60_time_arr = []

    for samp, aux in zip(st_ann.sample, st_ann.aux_note):
        if not aux:
            continue
        body = aux[3:] if aux.startswith("DMY") else aux  # tolerate DMY-prefixed (manually corrected) beats
        if not body.startswith(lead_prefix):
            continue

        rest = body[len(lead_prefix):]
        if rest.startswith(":") or rest.startswith(","):
            rest = rest[1:]

        fields = rest.split(",")

        if len(fields) > 9:
            try:
                ee_uV = float(fields[4])   # ee = J+60 amplitude in uV
                jp_ms = float(fields[9])   # JP = J-point offset after fiducial in ms

                j60_sample = int(round(samp + ((jp_ms + 60.0) / 1000.0) * fs))
                j60_time_s = j60_sample / fs

            except Exception:
                continue

            samp_arr.append(int(samp))
            ee_mV_arr.append(ee_uV / 1000.0)   # uV -> mV
            j60_sample_arr.append(j60_sample)
            j60_time_arr.append(j60_time_s)

    #return samp_arr, ee_mV_arr, j60_sample_arr, j60_time_arr
    return j60_sample_arr, ee_mV_arr


def aggregate_to_segments(
    samp_arr: list[int],
    st_arr: list[float],
    fs: int,
    segment_duration_s: float = 10.0,
    agg: str = "median",
) -> tuple[list[int], list[float]]:
    """
    Bin per-beat (sample, ST mV) onto a fixed grid of segment_duration_s windows
    starting at t=0. For each non-empty window, emit one (end_sample, agg_st_mV)
    point at the END of the window. Empty windows are skipped (the FSM's dt-based
    timers absorb the gap naturally).

    NaNs are propagated: if any beat in a segment has a NaN ST, the aggregate is
    NaN, and the FSM's `not np.isfinite(st)` gate will skip it.

    agg: "mean" or "median".
    """
    if not samp_arr:
        return [], []

    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")

    samp = np.asarray(samp_arr, dtype=int)
    st = np.asarray(st_arr, dtype=float)

    seg_len = int(round(segment_duration_s * fs))
    if seg_len <= 0:
        raise ValueError("segment_duration_s * fs must be positive.")

    seg_idx = samp // seg_len  # window k covers [k*seg_len, (k+1)*seg_len)

    out_samp: list[int] = []
    out_st: list[float] = []

    for k in np.unique(seg_idx):
        vals = st[seg_idx == k]
        if vals.size == 0:
            continue
        if agg == "median":
            agg_val = float(np.median(vals))
        else:
            agg_val = float(np.mean(vals))
        out_samp.append(int((k + 1) * seg_len))
        out_st.append(agg_val)

    return out_samp, out_st


def aggregate_to_segment_center(
    samp_arr: list[int],
    st_arr: list[float],
    fs: int,
    segment_duration_s: float = 10.0,
) -> tuple[list[int], list[float]]:
    """
    For each non-overlapping segment, find the beat whose sample index is
    closest to the segment midpoint (0.5 * segment_duration_s).
    Emit (end_sample, that_beat_st_mV) for every segment that contains at
    least one beat.
    """
    if not samp_arr:
        return [], []

    samp = np.asarray(samp_arr, dtype=int)
    st = np.asarray(st_arr, dtype=float)

    seg_len = int(round(segment_duration_s * fs))
    mid_offset = seg_len // 2
    seg_idx = samp // seg_len

    out_samp: list[int] = []
    out_st: list[float] = []

    for k in np.unique(seg_idx):
        mask = seg_idx == k
        seg_samps = samp[mask]
        seg_st = st[mask]
        midpoint = int(k * seg_len + mid_offset)
        closest = int(np.argmin(np.abs(seg_samps - midpoint)))
        out_samp.append(int((k + 1) * seg_len))
        out_st.append(float(seg_st[closest]))

    return out_samp, out_st


def aggregate_rolling_window(
    samp_arr: list[int],
    st_arr: list[float],
    fs: int,
    half_window_s: float = 8.0,
    agg: str = "median",
) -> tuple[list[int], list[float]]:
    """
    For each beat i, collect all beats whose sample index falls within
    [samp[i] - half_window_s*fs, samp[i] + half_window_s*fs] and replace
    st_arr[i] with the mean/median of that window.
    Timestamp = original beat sample (beat-indexed).
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    if not samp_arr:
        return [], []

    samp = np.asarray(samp_arr, dtype=int)
    st = np.asarray(st_arr, dtype=float)

    half_w = int(round(half_window_s * fs))
    left_idx = np.searchsorted(samp, samp - half_w, side="left")
    right_idx = np.searchsorted(samp, samp + half_w, side="right")

    smoothed: list[float] = []
    for i in range(len(samp)):
        vals = st[left_idx[i]:right_idx[i]]
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            smoothed.append(float("nan"))
        elif agg == "median":
            smoothed.append(float(np.median(valid)))
        else:
            smoothed.append(float(np.mean(valid)))

    return samp_arr, smoothed


def extract_st_nl_nr_from_ann(
    st_ann,
    lead_prefix: str,
    fs: float,
) -> tuple[list[int], list[float], list[int], list[int]]:
    """
    Extract (j60_sample, ee_mV, NL, NR) from a .16a annotation.
    fields[4]  = ee  (J+60 amplitude in µV)
    fields[9]  = JP  (J-point offset in ms)
    fields[11] = NL  (beats to the left in the average-beat template)
    fields[12] = NR  (beats to the right)
    """
    j60_arr: list[int] = []
    ee_arr: list[float] = []
    nl_arr: list[int] = []
    nr_arr: list[int] = []

    for samp, aux in zip(st_ann.sample, st_ann.aux_note):
        if not aux:
            continue
        body = aux[3:] if aux.startswith("DMY") else aux  # tolerate DMY-prefixed (manually corrected) beats
        if not body.startswith(lead_prefix):
            continue
        rest = body[len(lead_prefix):]
        if rest.startswith(":") or rest.startswith(","):
            rest = rest[1:]
        fields = rest.split(",")
        if len(fields) > 12:
            try:
                ee_uV = float(fields[4])
                jp_ms = float(fields[9])
                nl = int(fields[11])
                nr = int(fields[12])
            except (ValueError, IndexError):
                continue
            j60_sample = int(round(samp + ((jp_ms + 60.0) / 1000.0) * fs))
            j60_arr.append(j60_sample)
            ee_arr.append(ee_uV / 1000.0)
            nl_arr.append(nl)
            nr_arr.append(nr)

    return j60_arr, ee_arr, nl_arr, nr_arr


def aggregate_nl_nr_window(
    samp_arr: list[int],
    st_arr: list[float],
    nl_arr: list[int],
    nr_arr: list[int],
    agg: str = "median",
) -> tuple[list[int], list[float]]:
    """
    For each beat at position i in the sorted array, aggregate the ST values
    of the window [i - NL[i], ..., i, ..., i + NR[i]] (by beat-position index,
    not by time).  Timestamp = original beat sample.
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    if not samp_arr:
        return [], []

    st = np.asarray(st_arr, dtype=float)
    n = len(st)
    smoothed: list[float] = []

    for i in range(n):
        lo = max(0, i - nl_arr[i])
        hi = min(n, i + nr_arr[i] + 1)
        vals = st[lo:hi]
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            smoothed.append(float("nan"))
        elif agg == "median":
            smoothed.append(float(np.median(valid)))
        else:
            smoothed.append(float(np.mean(valid)))

    return samp_arr, smoothed


def get_rpeaks_from_atr(atr_ann) -> np.ndarray:
    """
    Extract beat locations from WFDB atr annotation.
    In LTST this usually corresponds to QRS locations.
    """
    # Most WFDB beat symbols; you can tune this list if needed
    beat_syms = {
        "N","L","R","A","a","J","S","V","F","e","j","E","/","f","Q","?","|"
    }

    samples = np.asarray(atr_ann.sample, dtype=int)
    syms = getattr(atr_ann, "symbol", None)
    if syms is None:
        return samples  # fallback: return all samples

    syms = np.asarray(syms, dtype=object)
    mask = np.array([s in beat_syms for s in syms], dtype=bool)
    return samples[mask]


# ------------------------------------------------
# SQI computation (your function, unchanged)
# ------------------------------------------------
def compute_sqi_last30s(
    r_peaks: np.ndarray,
    sig_filt: np.ndarray,
    current_r_peak: int,
    pre_sec: float = 0.35,
    post_sec: float = 0.35,
    window_sec: float = 30.0,
    fs: float = 250.0,
    min_beats: int = 3,
    eps: float = 1e-12,
) -> float:
    r_peaks = np.asarray(r_peaks, dtype=int)
    sig_filt = np.asarray(sig_filt, dtype=float)

    if sig_filt.ndim != 1:
        raise ValueError("sig_filt must be a 1D array.")
    if r_peaks.ndim != 1:
        raise ValueError("r_peaks must be a 1D array.")
    if r_peaks.size == 0:
        return np.nan

    pre_samp = int(round(pre_sec * fs))
    post_samp = int(round(post_sec * fs))
    beat_len = pre_samp + post_samp + 1

    win_len = int(round(window_sec * fs))
    win_end_excl = int(current_r_peak + post_samp + 1)
    win_start = win_end_excl - win_len

    if win_start < 0 or win_end_excl > len(sig_filt):
        return np.nan

    candidate = r_peaks[r_peaks <= current_r_peak]
    seg_starts = candidate - pre_samp
    seg_ends_excl = candidate + post_samp + 1

    ok = (
        (seg_starts >= win_start) &
        (seg_ends_excl <= win_end_excl) &
        (seg_starts >= 0) &
        (seg_ends_excl <= len(sig_filt))
    )
    used_r = candidate[ok]

    if used_r.size < min_beats:
        return np.nan

    beats = np.empty((used_r.size, beat_len), dtype=float)
    for i, r in enumerate(used_r):
        s0 = int(r - pre_samp)
        s1 = int(r + post_samp + 1)
        beats[i, :] = sig_filt[s0:s1]

    w_avg = beats.mean(axis=0)
    p_sig = float(np.mean(w_avg ** 2))

    diffs = beats - w_avg[None, :]
    p_noise = np.mean(diffs ** 2, axis=1)
    p_noise = np.maximum(p_noise, eps)

    snr_db = 10.0 * np.log10(max(p_sig, eps) / p_noise)
    sqi = float(np.percentile(snr_db, 25))
    return sqi


# ------------------------------------------------
# FSM logic with SQI integrated
# ------------------------------------------------
def run_fsm_on_record(
    rec_id: str,
    lead_tag: str,
    samp_arr: list[int],
    st_arr: list[float],
    ecg_sig_for_sqi: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
    use_sqi: bool = True,
    sqi_threshold_db: float = 5.0,
    sqi_every_s: float = 5.0,
    sqi_window_s: float = 30.0,
    pre_sec: float = 0.35,
    post_sec: float = 0.35,
    min_beats: int = 3,
) -> list[Alarm]:
    """
    FSM over ST samples (samp_arr/st_arr), with optional SQI gating.
    For now: `current_r_peak` for SQI is set to `samp` (your request).

    SQI is computed only every `sqi_every_s` seconds; between checks, last SQI_Good is reused.
    SQI is "good" if sqi_db is finite AND >= sqi_threshold_db.
    """
    lead_paper = STTAG_TO_PAPER_LEAD[lead_tag]

    state = S.REST
    alarms: list[Alarm] = []

    t01 = t02 = tneg = None
    d01 = d02 = dneg = None

    cd_start = None
    cdc = None

    alarm_onset_breach = alarm_onset_expected = alarm_onset_actual = None
    alarm_crit = None

    prev = None

    # SQI gating state
    last_sqi_check_samp: Optional[int] = None
    SQI_Good: bool = True  # default until first check (or always True if use_sqi=False)

    def maybe_update_sqi(now_samp: int) -> None:
        nonlocal last_sqi_check_samp, SQI_Good

        if not use_sqi:
            SQI_Good = True
            return

        if last_sqi_check_samp is None:
            do_check = True
        else:
            do_check = ((now_samp - last_sqi_check_samp) / fs) >= sqi_every_s

        if not do_check:
            return

        # For now we use samp as current_r_peak (as you requested)
        current_r = int(now_samp)
        sqi_db = compute_sqi_last30s(
            r_peaks=r_peaks,
            sig_filt=ecg_sig_for_sqi,
            current_r_peak=current_r,
            pre_sec=pre_sec,
            post_sec=post_sec,
            window_sec=sqi_window_s,
            fs=float(fs),
            min_beats=min_beats,
        )

        # Define "good"
        SQI_Good = (not np.isnan(sqi_db)) and (sqi_db >= sqi_threshold_db)
        last_sqi_check_samp = int(now_samp)

    for samp, st in zip(samp_arr, st_arr):
        samp = int(samp)

        # Update SQI occasionally (every sqi_every_s)
        maybe_update_sqi(samp)

        # --- ST validity gate (treat invalid ST like "no data": freeze FSM timers/transitions)
        if st is None:
            prev = samp
            continue
        st = float(st)
        if not np.isfinite(st):
            prev = samp
            continue

        if state == S.REST:
            if SQI_Good:
                if st >= 0.20:
                    t01 = t02 = samp
                    d01 = d02 = 0
                    state, alarm_crit = S.WU_02, "≥ 0.2 mV"
                elif st >= 0.10:
                    t01 = samp
                    d01 = 0
                    state, alarm_crit = S.WU_01, "≥ 0.1 mV"
                elif st <= -0.10:
                    tneg = samp
                    dneg = 0
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"

        elif state == S.WU_01:
            if SQI_Good and prev is not None:
                if st >= 0.20:
                    d01 = d01 + (samp - prev) / fs
                    if d01 >= T_WU_01:
                        state = S.ALARM
                        alarm_onset_breach = int(t01)
                        alarm_onset_expected = int(t01 + T_WU_01 * fs)
                        alarm_onset_actual = int(samp)
                        alarm_crit = "≥ 0.1 mV · 10 min"
                    else:
                        t02 = samp
                        d02 = 0
                        state, alarm_crit = S.WU_02, "≥ 0.2 mV"

                elif st <= -0.10:
                    tneg = samp
                    dneg = 0
                    t01 = d01 = None
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"

                elif -0.10 < st < 0.10:
                    state = S.REST
                    t01 = d01 = None

                else:
                    d01 = d01 + (samp - prev) / fs
                    if d01 >= T_WU_01:
                        state = S.ALARM
                        alarm_onset_breach = int(t01)
                        alarm_onset_expected = int(t01 + T_WU_01 * fs)
                        alarm_onset_actual = int(samp)
                        alarm_crit = "≥ 0.1 mV · 10 min"

        elif state == S.WU_02:
            if SQI_Good and prev is not None:
                if st <= -0.10:
                    tneg = samp
                    dneg = 0
                    t01 = t02 = None
                    d01 = d02 = None
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"

                elif -0.10 < st < 0.10:
                    state = S.REST
                    t01 = t02 = None
                    d01 = d02 = None

                elif 0.10 <= st < 0.20:
                    d01 = d01 + (samp - prev) / fs
                    done_01 = d01 >= T_WU_01
                    if done_01:
                        state = S.ALARM
                        alarm_onset_breach = int(t01)
                        alarm_onset_expected = int(t01 + T_WU_01 * fs)
                        alarm_onset_actual = int(samp)
                        alarm_crit = "≥ 0.1 mV · 10 min"
                    else:
                        state, alarm_crit = S.WU_01, "≥ 0.1 mV"
                    t02 = d02 = None

                else:
                    d02 = d02 + (samp - prev) / fs
                    done_02 = d02 >= T_WU_02
                    d01 = d01 + (samp - prev) / fs
                    done_01 = d01 >= T_WU_01

                    if done_01 or done_02:
                        crits = []
                        times = []
                        breach_starts = []

                        if done_01:
                            crits.append("≥ 0.1 mV · 10 min")
                            times.append(t01 + int(T_WU_01 * fs))
                            breach_starts.append(t01)
                        if done_02:
                            crits.append("≥ 0.2 mV · 1 min")
                            times.append(t02 + int(T_WU_02 * fs))
                            breach_starts.append(t02)

                        alarm_crit = " & ".join(crits)
                        alarm_onset_expected = int(min(times))
                        alarm_onset_breach = int(min(breach_starts))
                        alarm_onset_actual = int(samp)
                        state = S.ALARM

        elif state == S.WU_NEG:
            if SQI_Good and prev is not None:
                if st >= 0.20:
                    t01 = t02 = samp
                    d01 = d02 = 0
                    tneg = dneg = None
                    state, alarm_crit = S.WU_02, "≥ 0.2 mV"

                elif 0.10 <= st < 0.20:
                    t01 = samp
                    d01 = 0
                    tneg = dneg = None
                    state, alarm_crit = S.WU_01, "≥ 0.1 mV"

                elif -0.10 < st < 0.10:
                    state = S.REST
                    tneg = dneg = None

                else:
                    dneg = dneg + (samp - prev) / fs
                    if dneg >= T_WU_NEG:
                        state = S.ALARM
                        alarm_onset_breach = int(tneg)
                        alarm_onset_expected = int(tneg + T_WU_NEG * fs)
                        alarm_onset_actual = int(samp)
                        alarm_crit = "≤ -0.1 mV · 1 min"

        elif state == S.ALARM:
            if SQI_Good:
                if abs(st) <= 0.05:
                    state = S.COOLDOWN
                    cd_start = samp
                    cdc = 0

        elif state == S.COOLDOWN:
            if SQI_Good and prev is not None:
                if abs(st) >= 0.10:
                    state = S.ALARM
                    cd_start = samp
                    cdc = 0
                else:
                    cdc = cdc + (samp - prev) / fs
                    if cdc >= T_CD:
                        end_expected = int(cd_start + T_CD * fs)
                        end_actual = int(samp)

                        alarms.append(Alarm(
                            record=rec_id,
                            lead_tag=lead_tag,
                            lead_paper=lead_paper,
                            onset_breach=int(alarm_onset_breach),
                            onset_expected=int(alarm_onset_expected),
                            onset_actual=int(alarm_onset_actual),
                            end_expected=end_expected,
                            end_actual=end_actual,
                            crit=alarm_crit
                        ))

                        state = S.REST
                        t01 = t02 = tneg = None
                        d01 = d02 = dneg = None
                        alarm_onset_breach = alarm_onset_expected = alarm_onset_actual = None
                        cd_start = cdc = None
                        alarm_crit = None

        prev = samp

    # EOF flush policy
    if samp_arr:
        last_samp = prev
        if state == S.ALARM and (alarm_onset_expected is not None) and (alarm_onset_actual is not None):
            print(
                f"[EOF flush ALARM] rec={rec_id} lead={lead_tag} "
                f"last_samp={last_samp} "
                f"onset_expected={alarm_onset_expected} "
                f"onset_actual={alarm_onset_actual} "
                f"crit={alarm_crit}"
            )

            alarms.append(Alarm(
                record=rec_id,
                lead_tag=lead_tag,
                lead_paper=lead_paper,
                onset_breach=int(alarm_onset_breach),
                onset_expected=int(alarm_onset_expected),
                onset_actual=int(alarm_onset_actual),
                end_expected=int(last_samp),
                end_actual=int(last_samp),
                crit=alarm_crit
            ))

        elif (
            state == S.COOLDOWN
            and (alarm_onset_expected is not None)
            and (alarm_onset_actual is not None)
            and (cd_start is not None)
        ):
            # Recording ended mid-cooldown: the alarm event already occurred when the FSM
            # entered ALARM. Emit it with end_expected at the projected cooldown end and
            # end_actual at the last available sample.
            end_expected = int(cd_start + T_CD * fs)
            end_actual = int(last_samp)

            print(
                f"[EOF flush COOLDOWN] rec={rec_id} lead={lead_tag} "
                f"last_samp={last_samp} "
                f"onset_expected={alarm_onset_expected} "
                f"onset_actual={alarm_onset_actual} "
                f"end_expected={end_expected} "
                f"end_actual={end_actual} "
                f"cdc={cdc:.2f}s "
                f"crit={alarm_crit}"
            )

            alarms.append(Alarm(
                record=rec_id,
                lead_tag=lead_tag,
                lead_paper=lead_paper,
                onset_breach=int(alarm_onset_breach),
                onset_expected=int(alarm_onset_expected),
                onset_actual=int(alarm_onset_actual),
                end_expected=end_expected,
                end_actual=end_actual,
                crit=alarm_crit
            ))

    return alarms


# ------------------------------------------------
# CLI / Main
# ------------------------------------------------
_VALID_INPUT_MODES = (
    "per_beat",
    "segment_agg",
    "segment_center",
    "rolling_window",
    "nl_nr_window",
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate gold standard alarms with optional SQI gating.")
    p.add_argument("--base_path", type=str, default=BASE_PATH_DEFAULT, help="Path to ltst_records root.")
    p.add_argument("--fs", type=int, default=FS_DEFAULT, help="Sampling frequency (Hz). Default 250.")
    p.add_argument("--out_path", type=str, required=True, help="Output CSV path.")
    p.add_argument("--disable_sqi", action="store_true", help="Disable SQI gating (SQI always good).")
    p.add_argument("--sqi_threshold_db", type=float, default=5.0, help="SQI good threshold (dB).")
    p.add_argument("--sqi_every_s", type=float, default=5.0, help="Compute SQI every N seconds. Default 5s.")
    p.add_argument("--sqi_window_s", type=float, default=30.0, help="SQI trailing window length. Default 30s.")
    p.add_argument("--use_manual_omit", action="store_true", default=USE_MANUAL_OMIT_DEFAULT,
                   help="Use manual omit list (default on).")
    p.add_argument("--input_mode", type=str, default="per_beat",
                   choices=list(_VALID_INPUT_MODES),
                   help=(
                       "ST input granularity: "
                       "per_beat (raw J+60ms per beat), "
                       "segment_agg (mean/median over non-overlapping windows, ts=segment end), "
                       "segment_center (beat closest to segment midpoint, ts=segment end), "
                       "rolling_window (±rolling_half_window_s rolling mean/median, ts=beat), "
                       "nl_nr_window (NL/NR beat-count window from .16a fields, ts=beat)."
                   ))
    p.add_argument("--segment_duration_s", type=float, default=10.0,
                   help="Segment width in seconds (segment_agg / segment_center). Default 10.0.")
    p.add_argument("--segment_agg", type=str, default="median",
                   choices=["mean", "median"],
                   help="Aggregation for segment_agg mode. Default: median.")
    p.add_argument("--rolling_half_window_s", type=float, default=8.0,
                   help="Half-window in seconds for rolling_window mode (±8s → 16s span). Default 8.0.")
    p.add_argument("--rolling_agg", type=str, default="median",
                   choices=["mean", "median"],
                   help="Aggregation for rolling_window and nl_nr_window modes. Default: median.")
    return p


def generate_gold_standard_alarms(
    base_path: str,
    output_csv: str,
    fs: int = 250,
    use_sqi: bool = True,
    sqi_threshold_db: float = 5.0,
    sqi_every_s: float = 5.0,
    sqi_window_s: float = 30.0,
    use_manual_omit: bool = True,
    input_mode: str = "per_beat",
    segment_duration_s: float = 10.0,
    segment_agg: str = "median",
    rolling_half_window_s: float = 8.0,
    rolling_agg: str = "median",
) -> None:
    """
    Generate gold standard alarms from LTST database records.

    input_mode choices
    ------------------
    per_beat        : one ST point per beat, timestamp = J+60ms sample
    segment_agg     : mean/median of all beats in non-overlapping windows,
                      timestamp = segment end sample
    segment_center  : ST of the beat closest to the segment midpoint,
                      timestamp = segment end sample
    rolling_window  : per-beat ±rolling_half_window_s rolling mean/median,
                      timestamp = original beat sample
    nl_nr_window    : per-beat window defined by NL/NR fields in .16a,
                      aggregated with rolling_agg; timestamp = original beat sample
    """
    if input_mode not in _VALID_INPUT_MODES:
        raise ValueError(f"input_mode must be one of {_VALID_INPUT_MODES}, got {input_mode!r}")
    out_path = Path(output_csv)
    all_alarms: list[Alarm] = []

    for record in LTSTDB_RECORDS_DEFAULT:
        print(f"⏳  {record}")
        record_path = f"{base_path}/{record}/{record}"

        # ST annotations
        st_ann = wfdb.rdann(record_path, "16a")

        # Signal + atr for r-peaks
        rec = wfdb.rdrecord(record_path)
        atr = wfdb.rdann(record_path, "atr")

        # r-peaks (same set reused for both leads; fine for now)
        r_peaks = get_rpeaks_from_atr(atr)

        for lead_tag in ("ST0", "ST1"):
            paper_lead = STTAG_TO_PAPER_LEAD[lead_tag]

            if use_manual_omit and ((record, paper_lead) in MANUAL_OMIT):
                print(f"   → {lead_tag} ({paper_lead}): SKIPPED (manual omit)")
                continue

            # choose ECG lead for SQI
            lead_idx = STTAG_TO_LEAD_IDX[lead_tag]
            if rec.p_signal.ndim != 2 or lead_idx >= rec.p_signal.shape[1]:
                print(f"   → {lead_tag} ({paper_lead}): SKIPPED (missing lead idx={lead_idx})")
                continue

            ecg_lead = rec.p_signal[:, lead_idx].astype(float)
            ecg_for_sqi = ecg_lead

            if input_mode == "nl_nr_window":
                j60_sample_arr, ee_mV_arr, nl_arr, nr_arr = extract_st_nl_nr_from_ann(
                    st_ann, lead_tag, fs
                )
            else:
                j60_sample_arr, ee_mV_arr = extract_ee_and_j60_location_from_ann(
                    st_ann, lead_tag, 250
                )

            if input_mode == "segment_agg":
                samp_arr, st_arr = aggregate_to_segments(
                    j60_sample_arr,
                    ee_mV_arr,
                    fs=fs,
                    segment_duration_s=segment_duration_s,
                    agg=segment_agg,
                )
            elif input_mode == "segment_center":
                samp_arr, st_arr = aggregate_to_segment_center(
                    j60_sample_arr,
                    ee_mV_arr,
                    fs=fs,
                    segment_duration_s=segment_duration_s,
                )
            elif input_mode == "rolling_window":
                samp_arr, st_arr = aggregate_rolling_window(
                    j60_sample_arr,
                    ee_mV_arr,
                    fs=fs,
                    half_window_s=rolling_half_window_s,
                    agg=rolling_agg,
                )
            elif input_mode == "nl_nr_window":
                samp_arr, st_arr = aggregate_nl_nr_window(
                    j60_sample_arr,
                    ee_mV_arr,
                    nl_arr,
                    nr_arr,
                    agg=rolling_agg,
                )
            else:  # per_beat
                samp_arr, st_arr = j60_sample_arr, ee_mV_arr

            rec_alarms = run_fsm_on_record(
                rec_id=record,
                lead_tag=lead_tag,
                samp_arr=samp_arr,
                st_arr=st_arr,
                ecg_sig_for_sqi=ecg_for_sqi,
                r_peaks=r_peaks,
                fs=fs,
                use_sqi=use_sqi,
                sqi_threshold_db=sqi_threshold_db,
                sqi_every_s=sqi_every_s,
                sqi_window_s=sqi_window_s,
            )

            print(f"   → {lead_tag} ({paper_lead}): {len(rec_alarms)} alarms "
                  f"[mode={input_mode}, n_points={len(samp_arr)}]")
            all_alarms.extend(rec_alarms)

    print("Total # of alarms:", len(all_alarms))

    df = pd.DataFrame([{
        "record": a.record,
        "lead_tag": a.lead_tag,
        "lead_paper": a.lead_paper,
        "strategy": a.strategy if hasattr(a, 'strategy') else "gold_standard",

        "onset_breach_sample": a.onset_breach,
        "onset_expected_sample": a.onset_expected,
        "onset_actual_sample":   a.onset_actual,
        "end_expected_sample":   a.end_expected,
        "end_actual_sample":     a.end_actual,

        "onset_breach_s": a.onset_breach / fs,
        "onset_expected_s": a.onset_expected / fs,
        "onset_actual_s":   a.onset_actual   / fs,
        "end_expected_s":   a.end_expected   / fs,
        "end_actual_s":     a.end_actual     / fs,

        "duration_expected_s": (a.end_expected - a.onset_expected) / fs,
        "duration_actual_s":   (a.end_actual   - a.onset_actual)   / fs,
        "warmup_duration_s":   (a.onset_actual - a.onset_breach) / fs,

        "criterion": a.crit,
    } for a in all_alarms])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"💾  Saved to {out_path}")


def main():
    args = build_arg_parser().parse_args()

    generate_gold_standard_alarms(
        base_path=args.base_path,
        output_csv=args.out_path,
        fs=int(args.fs),
        use_sqi=not args.disable_sqi,
        sqi_threshold_db=float(args.sqi_threshold_db),
        sqi_every_s=float(args.sqi_every_s),
        sqi_window_s=float(args.sqi_window_s),
        use_manual_omit=args.use_manual_omit,
        input_mode=args.input_mode,
        segment_duration_s=float(args.segment_duration_s),
        segment_agg=args.segment_agg,
        rolling_half_window_s=float(args.rolling_half_window_s),
        rolling_agg=args.rolling_agg,
    )


def process_csv_directory(
    processed_dir: str,
    output_csv: str,
    fs: int = 250,
    use_sqi: bool = False,
    sqi_threshold_db: float = 5.0,
    sqi_every_s: float = 5.0,
    sqi_window_s: float = 30.0,
    records_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Process all CSV files from the processed directory and generate alarms.
    Handles multiple strategy CSV files per record+lead pair.

    Args:
        processed_dir: Base directory containing subdirectories like s20011/lead_0/
        output_csv: Path to save the output alarms CSV
        fs: Sampling frequency (Hz)
        use_sqi: Whether to use SQI gating (default False since we don't have raw ECG)
        sqi_threshold_db: SQI threshold in dB
        sqi_every_s: Compute SQI every N seconds
        sqi_window_s: SQI window length in seconds
    """
    import re

    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"Directory not found: {processed_dir}")

    # Find all CSV files matching pattern (any strategy)
    csv_files = list(processed_path.glob("*/*/*_lead*.csv"))

    if not csv_files:
        print(f"No CSV files found in {processed_dir}")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    all_alarms: list[Alarm] = []

    for csv_path in sorted(csv_files):
        # Extract record, lead, and strategy from filename
        # Example: waves_s20011_lead0_j60_only.csv
        # Example: waves_s20011_lead0_pr_point_prmedian_min10.csv
        filename = csv_path.stem
        match = re.search(r'waves_(\w+)_lead(\d+)_(.+)$', filename)

        if not match:
            print(f"⚠️  Skipping {csv_path.name}: cannot parse record/lead/strategy")
            continue

        record = match.group(1)
        lead_idx = int(match.group(2))
        strategy = match.group(3)
        lead_tag = f"ST{lead_idx}"

        print(f"⏳  Processing {record} lead {lead_idx} strategy={strategy}")

        # Load CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"   ❌ Failed to read CSV: {e}")
            continue

        # --- Detect Path (per-beat or segment) ---
        if 'segment_end_sample' in df.columns and 'st_deviation_agg_mV' in df.columns:
            # New 10s segment pathway
            print(f"   📂 Detected segment-based pathway")
            samp_col = 'segment_end_sample'
            st_col = 'st_deviation_agg_mV'
        elif 'r_peak' in df.columns and 'st_deviation_mV' in df.columns:
            # Current per-beat pathway
            print(f"   📂 Detected per-beat pathway")
            samp_col = 'r_peak'
            st_col = 'st_deviation_mV'
        else:
            print(f"   ⚠️  Missing required columns for either pathway in {csv_path.name}")
            continue

        # Extract arrays - optionally filter NaN values
        # NaN in st_deviation_agg_mV will fail all comparisons, so FSM won't transition states
        # This allows beats/segments with missing ST values to advance time without triggering alarms
        samp_series = pd.to_numeric(df[samp_col], errors='coerce')
        st_series = pd.to_numeric(df[st_col], errors='coerce')

        # Convert to lists - samp must be valid (need sample index)
        # Using pd.notna(samp_series) to avoid potential 'ndarray' attribute error
        valid_mask = pd.notna(samp_series)
        samp_arr = samp_series[valid_mask].astype(int).tolist()
        st_arr = st_series[valid_mask].tolist()

        if not samp_arr:
            print(f"   ⚠️  No data rows with valid {samp_col} and {st_col}")
            continue

        print(f"   📊 Using {len(samp_arr)} valid points (filtered out NaN {samp_col})")

        # For SQI, we need r_peaks and ECG signal
        ecg_sig_for_sqi = None
        r_peaks = None
        use_sqi_actual = use_sqi

        if use_sqi:
            if records_dict and record in records_dict:
                rd = records_dict[record]
                if "sig_used" in rd and "atr" in rd:
                    ecg_sig_for_sqi = rd["sig_used"]
                    # Extract the lead we need
                    if ecg_sig_for_sqi.ndim > 1:
                        if lead_idx < ecg_sig_for_sqi.shape[1]:
                            ecg_sig_for_sqi = ecg_sig_for_sqi[:, lead_idx]
                        else:
                            print(f"   ⚠️  SQI requested but lead {lead_idx} not in sig_used for {record}, disabling SQI")
                            use_sqi_actual = False
                    
                    if use_sqi_actual:
                        # Extract R-peaks from atr
                        r_peaks = get_rpeaks_from_atr(rd["atr"])
                        print(f"   ✅ SQI enabled using loaded signal for {record}")
                else:
                    print(f"   ⚠️  SQI requested but record {record} missing signal/atr in records_dict, disabling SQI")
                    use_sqi_actual = False
            else:
                print(f"   ⚠️  SQI requested but no raw ECG available (no records_dict), disabling SQI for this run")
                use_sqi_actual = False

        if not use_sqi_actual:
            # Dummy arrays for when SQI is disabled
            r_peaks = np.array(samp_arr, dtype=int)
            ecg_sig_for_sqi = np.zeros(max(samp_arr) + 1000, dtype=float)

        # Run FSM
        try:
            rec_alarms = run_fsm_on_record(
                rec_id=record,
                lead_tag=lead_tag,
                samp_arr=samp_arr,
                st_arr=st_arr,
                ecg_sig_for_sqi=ecg_sig_for_sqi,
                r_peaks=r_peaks,
                fs=fs,
                use_sqi=use_sqi_actual,
                sqi_threshold_db=sqi_threshold_db,
                sqi_every_s=sqi_every_s,
                sqi_window_s=sqi_window_s,
            )

            print(f"   ✓ {lead_tag} {strategy}: {len(rec_alarms)} alarms")

            # Add strategy info to each alarm
            for alarm in rec_alarms:
                alarm.strategy = strategy  # We'll need to add this field to Alarm dataclass

            all_alarms.extend(rec_alarms)

        except Exception as e:
            print(f"   ❌ FSM failed: {e}")
            continue

    print(f"\n📊 Total alarms: {len(all_alarms)}")

    # Save to CSV
    df_out = pd.DataFrame([{
        "record": a.record,
        "lead_tag": a.lead_tag,
        "lead_paper": a.lead_paper,
        "strategy": a.strategy,

        "onset_breach_sample": a.onset_breach,
        "onset_expected_sample": a.onset_expected,
        "onset_actual_sample":   a.onset_actual,
        "end_expected_sample":   a.end_expected,
        "end_actual_sample":     a.end_actual,

        "onset_breach_s": a.onset_breach / fs,
        "onset_expected_s": a.onset_expected / fs,
        "onset_actual_s":   a.onset_actual   / fs,
        "end_expected_s":   a.end_expected   / fs,
        "end_actual_s":     a.end_actual     / fs,

        "duration_expected_s": (a.end_expected - a.onset_expected) / fs,
        "duration_actual_s":   (a.end_actual   - a.onset_actual)   / fs,
        "warmup_duration_s":   (a.onset_actual - a.onset_breach) / fs,

        "criterion": a.crit,
    } for a in all_alarms])

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"💾  Saved to {out_path}")

    # Print summary by strategy
    if not df_out.empty and "strategy" in df_out.columns:
        print(f"\n📈 Alarms by strategy:")
        strategy_counts = df_out.groupby("strategy").size().sort_values(ascending=False)
        for strat, count in strategy_counts.items():
            print(f"   {strat}: {count} alarms")


if __name__ == "__main__":
    main()




