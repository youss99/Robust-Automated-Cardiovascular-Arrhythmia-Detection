from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd


# -----------------------------
# FSM states and thresholds
# -----------------------------
class S(Enum):
    REST     = auto()
    WU_01    = auto()   # +0.1 mV · 10 min
    WU_02    = auto()   # +0.2 mV · 1 min
    WU_NEG   = auto()   # –0.1 mV · 1 min
    ALARM    = auto()
    COOLDOWN = auto()

T_WU_01, T_WU_02, T_WU_NEG, T_CD = 600.0, 60.0, 60.0, 30.0  # seconds


@dataclass
class AlarmEpisode:
    record_id: str
    lead_id: str
    crit: str

    onset_expected_s: float
    onset_actual_s: float
    end_expected_s: float
    end_actual_s: float

    # Optional sample indices (filled when possible)
    onset_expected_sample: Optional[int] = None
    onset_actual_sample: Optional[int] = None
    end_expected_sample: Optional[int] = None
    end_actual_sample: Optional[int] = None


TimeAnchor = Literal["t_ref_s", "r_peak_samp", "j60_samp", "samp_ref"]


@dataclass
class SQIConfig:
    enabled: bool = False

    # If using precomputed SQI column:
    use_precomputed_column: bool = True
    sqi_db_col: str = "sqi_db"          # if present, compare to threshold
    quality_ok_col: str = "quality_ok"  # if present, use directly

    # Thresholding:
    sqi_threshold_db: float = 5.0

    # If computing on-the-fly:
    compute_every_s: float = 5.0
    window_s: float = 30.0
    pre_sec: float = 0.35
    post_sec: float = 0.35
    min_beats: int = 3

    # Behavior if SQI can't be computed (missing columns/context)
    fail_open: bool = True  # True => treat as good; False => treat as bad


@dataclass
class FSMConfig:
    st_col: str = "st_dev_mV"
    time_anchor: TimeAnchor = "t_ref_s"
    fs: Optional[float] = 250.0  # required if time_anchor is a *_samp column

    # For turning your SQI gating on/off cleanly
    sqi: SQIConfig = SQIConfig()


# -----------------------------
# SQI computation (unchanged)
# -----------------------------
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

    if sig_filt.ndim != 1 or r_peaks.ndim != 1 or r_peaks.size == 0:
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
    return float(np.percentile(snr_db, 25))


# -----------------------------
# Beat stream normalization
# -----------------------------
def normalize_beat_stream(
    df: pd.DataFrame,
    cfg: FSMConfig,
    record_id: str = "",
    lead_id: str = "",
) -> pd.DataFrame:
    """
    Ensures we have:
      - st_col (float)
      - t_ref_s (float) created from anchor if needed
      - samp_ref (Int64) if an anchor sample column exists
    Sorts by t_ref_s and drops rows with NaNs in required fields.
    """
    out = df.copy()

    if cfg.st_col not in out.columns:
        raise ValueError(f"Missing required ST column: '{cfg.st_col}'")

    out[cfg.st_col] = pd.to_numeric(out[cfg.st_col], errors="coerce")

    # Build t_ref_s
    if cfg.time_anchor == "t_ref_s":
        if "t_ref_s" not in out.columns:
            raise ValueError("cfg.time_anchor='t_ref_s' but 't_ref_s' column is missing.")
        out["t_ref_s"] = pd.to_numeric(out["t_ref_s"], errors="coerce")
        # samp_ref optional here
    else:
        # anchor is a sample column
        if cfg.fs is None or cfg.fs <= 0:
            raise ValueError("cfg.fs must be set when using a sample-based time anchor.")
        if cfg.time_anchor not in out.columns:
            raise ValueError(f"time_anchor='{cfg.time_anchor}' but column missing.")
        anchor = pd.to_numeric(out[cfg.time_anchor], errors="coerce")
        out["t_ref_s"] = anchor / float(cfg.fs)
        out["samp_ref"] = anchor.astype("Int64")

    # Attach labels (optional)
    if "record_id" not in out.columns:
        out["record_id"] = record_id
    if "lead_id" not in out.columns:
        out["lead_id"] = lead_id

    # Drop unusable rows
    out = out.dropna(subset=[cfg.st_col, "t_ref_s"]).copy()
    out = out.sort_values("t_ref_s").reset_index(drop=True)

    # Ensure monotonic time (if duplicates exist, keep stable but allow dt=0)
    return out


# -----------------------------
# SQI gating decision
# -----------------------------
def get_quality_ok(
    row: pd.Series,
    cfg: FSMConfig,
    state: Dict[str, Any],
    *,
    ecg_sig_for_sqi: Optional[np.ndarray] = None,
    r_peaks: Optional[np.ndarray] = None,
) -> bool:
    sqic = cfg.sqi
    if not sqic.enabled:
        return True

    # 1) Precomputed gating
    if sqic.use_precomputed_column:
        if sqic.quality_ok_col in row.index and pd.notna(row.get(sqic.quality_ok_col)):
            return bool(row[sqic.quality_ok_col])

        if sqic.sqi_db_col in row.index and pd.notna(row.get(sqic.sqi_db_col)):
            return float(row[sqic.sqi_db_col]) >= float(sqic.sqi_threshold_db)

        # If enabled but column missing:
        return True if sqic.fail_open else False

    # 2) On-the-fly compute
    # Need signal, r_peaks, and a sample index for "current beat"
    if ecg_sig_for_sqi is None or r_peaks is None:
        return True if sqic.fail_open else False

    # determine current sample
    now_sample = None
    if "r_peak_samp" in row.index and pd.notna(row.get("r_peak_samp")):
        now_sample = int(row["r_peak_samp"])
    elif "samp_ref" in row.index and pd.notna(row.get("samp_ref")):
        now_sample = int(row["samp_ref"])

    if now_sample is None:
        return True if sqic.fail_open else False

    # cadence control
    last_t = state.get("last_sqi_check_t_ref_s", None)
    now_t = float(row["t_ref_s"])
    if last_t is not None and (now_t - last_t) < float(sqic.compute_every_s):
        return bool(state.get("last_quality_ok", True))

    sqi_db = compute_sqi_last30s(
        r_peaks=r_peaks,
        sig_filt=np.asarray(ecg_sig_for_sqi, dtype=float),
        current_r_peak=int(now_sample),
        pre_sec=float(sqic.pre_sec),
        post_sec=float(sqic.post_sec),
        window_sec=float(sqic.window_s),
        fs=float(cfg.fs if cfg.fs is not None else 250.0),
        min_beats=int(sqic.min_beats),
    )

    ok = (not np.isnan(sqi_db)) and (float(sqi_db) >= float(sqic.sqi_threshold_db))
    state["last_sqi_check_t_ref_s"] = now_t
    state["last_quality_ok"] = ok
    return ok


# -----------------------------
# FSM on beat stream
# -----------------------------
def run_fsm_on_beat_stream(
    beats_df: pd.DataFrame,
    cfg: FSMConfig,
    *,
    record_id: str,
    lead_id: str,
    ecg_sig_for_sqi: Optional[np.ndarray] = None,
    r_peaks: Optional[np.ndarray] = None,
) -> list[AlarmEpisode]:
    """
    Dataset-agnostic FSM runner.

    beats_df: normalized beat stream for ONE record+lead (one row per beat).
    Uses:
      - cfg.st_col for ST value
      - cfg.time_anchor for timebase
      - cfg.sqi to gate transitions

    Returns alarms with timestamps in seconds; sample indices filled if possible.
    """
    df = normalize_beat_stream(beats_df, cfg, record_id=record_id, lead_id=lead_id)

    if df.empty:
        return []

    st_col = cfg.st_col

    alarms: list[AlarmEpisode] = []
    state = S.REST

    # warm-up start times (seconds)
    t01_s = t02_s = tneg_s = None
    d01 = d02 = dneg = 0.0

    cd_start_s = None
    cdc = 0.0

    alarm_onset_expected_s = None
    alarm_onset_actual_s = None
    alarm_crit = None

    prev_t = None

    # SQI state
    sqi_state: Dict[str, Any] = {}

    # For optional sample mapping in output
    fs = float(cfg.fs) if cfg.fs else None

    def sample_at_time(row: pd.Series) -> Optional[int]:
        # prefer samp_ref if present
        if "samp_ref" in row.index and pd.notna(row.get("samp_ref")):
            return int(row["samp_ref"])
        # else if time-based and fs exists
        if fs is not None and pd.notna(row.get("t_ref_s")):
            return int(round(float(row["t_ref_s"]) * fs))
        return None

    for _, row in df.iterrows():
        t = float(row["t_ref_s"])
        st = float(row[st_col])

        quality_ok = get_quality_ok(
            row, cfg, sqi_state,
            ecg_sig_for_sqi=ecg_sig_for_sqi,
            r_peaks=r_peaks,
        )

        dt = 0.0 if prev_t is None else max(0.0, t - prev_t)

        if state == S.REST:
            if quality_ok:
                if st >= 0.20:
                    t01_s = t02_s = t
                    d01 = d02 = 0.0
                    state, alarm_crit = S.WU_02, "≥ 0.2 mV"
                elif st >= 0.10:
                    t01_s = t
                    d01 = 0.0
                    state, alarm_crit = S.WU_01, "≥ 0.1 mV"
                elif st <= -0.10:
                    tneg_s = t
                    dneg = 0.0
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"

        elif state == S.WU_01:
            if quality_ok:
                if st >= 0.20:
                    d01 += dt
                    if d01 >= T_WU_01:
                        state = S.ALARM
                        alarm_onset_expected_s = float(t01_s + T_WU_01)
                        alarm_onset_actual_s = float(t)
                        alarm_crit = "≥ 0.1 mV · 10 min"
                    else:
                        t02_s = t
                        d02 = 0.0
                        state, alarm_crit = S.WU_02, "≥ 0.2 mV"
                elif st <= -0.10:
                    tneg_s = t
                    dneg = 0.0
                    t01_s = None; d01 = 0.0
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"
                elif -0.10 < st < 0.10:
                    state = S.REST
                    t01_s = None; d01 = 0.0
                else:
                    d01 += dt
                    if d01 >= T_WU_01:
                        state = S.ALARM
                        alarm_onset_expected_s = float(t01_s + T_WU_01)
                        alarm_onset_actual_s = float(t)
                        alarm_crit = "≥ 0.1 mV · 10 min"

        elif state == S.WU_02:
            if quality_ok:
                if st <= -0.10:
                    tneg_s = t
                    dneg = 0.0
                    t01_s = None; t02_s = None
                    d01 = 0.0; d02 = 0.0
                    state, alarm_crit = S.WU_NEG, "≤ -0.1 mV"
                elif -0.10 < st < 0.10:
                    state = S.REST
                    t01_s = None; t02_s = None
                    d01 = 0.0; d02 = 0.0
                elif 0.10 <= st < 0.20:
                    d01 += dt
                    if d01 >= T_WU_01:
                        state = S.ALARM
                        alarm_onset_expected_s = float(t01_s + T_WU_01)
                        alarm_onset_actual_s = float(t)
                        alarm_crit = "≥ 0.1 mV · 10 min"
                    else:
                        state, alarm_crit = S.WU_01, "≥ 0.1 mV"
                else:
                    d02 += dt
                    d01 += dt
                    done_02 = d02 >= T_WU_02
                    done_01 = d01 >= T_WU_01
                    if done_01 or done_02:
                        crits = []
                        times = []
                        if done_01:
                            crits.append("≥ 0.1 mV · 10 min")
                            times.append(float(t01_s + T_WU_01))
                        if done_02:
                            crits.append("≥ 0.2 mV · 1 min")
                            times.append(float(t02_s + T_WU_02))
                        alarm_crit = " & ".join(crits)
                        alarm_onset_expected_s = float(min(times))
                        alarm_onset_actual_s = float(t)
                        state = S.ALARM

        elif state == S.WU_NEG:
            if quality_ok:
                if st >= 0.20:
                    t01_s = t02_s = t
                    d01 = d02 = 0.0
                    tneg_s = None; dneg = 0.0
                    state, alarm_crit = S.WU_02, "≥ 0.2 mV"
                elif 0.10 <= st < 0.20:
                    t01_s = t
                    d01 = 0.0
                    tneg_s = None; dneg = 0.0
                    state, alarm_crit = S.WU_01, "≥ 0.1 mV"
                elif -0.10 < st < 0.10:
                    state = S.REST
                    tneg_s = None; dneg = 0.0
                else:
                    dneg += dt
                    if dneg >= T_WU_NEG:
                        state = S.ALARM
                        alarm_onset_expected_s = float(tneg_s + T_WU_NEG)
                        alarm_onset_actual_s = float(t)
                        alarm_crit = "≤ -0.1 mV · 1 min"

        elif state == S.ALARM:
            if quality_ok:
                if abs(st) <= 0.05:
                    state = S.COOLDOWN
                    cd_start_s = t
                    cdc = 0.0

        elif state == S.COOLDOWN:
            if quality_ok:
                if abs(st) >= 0.10:
                    state = S.ALARM
                    cd_start_s = t
                    cdc = 0.0
                else:
                    cdc += dt
                    if cdc >= T_CD:
                        end_expected_s = float(cd_start_s + T_CD)
                        end_actual_s = float(t)

                        # sample mapping
                        onset_exp_samp = int(round(alarm_onset_expected_s * fs)) if fs else None
                        onset_act_samp = None
                        end_exp_samp = int(round(end_expected_s * fs)) if fs else None
                        end_act_samp = sample_at_time(row)

                        alarms.append(AlarmEpisode(
                            record_id=record_id,
                            lead_id=lead_id,
                            crit=str(alarm_crit),

                            onset_expected_s=float(alarm_onset_expected_s),
                            onset_actual_s=float(alarm_onset_actual_s),
                            end_expected_s=end_expected_s,
                            end_actual_s=end_actual_s,

                            onset_expected_sample=onset_exp_samp,
                            onset_actual_sample=onset_act_samp,
                            end_expected_sample=end_exp_samp,
                            end_actual_sample=end_act_samp,
                        ))

                        # reset
                        state = S.REST
                        t01_s = t02_s = tneg_s = None
                        d01 = d02 = dneg = 0.0
                        alarm_onset_expected_s = alarm_onset_actual_s = None
                        cd_start_s = None; cdc = 0.0
                        alarm_crit = None

        prev_t = t

    # EOF flush: if in ALARM at end, cut at last time
    if len(df) and state == S.ALARM and alarm_onset_expected_s is not None and alarm_onset_actual_s is not None:
        last_row = df.iloc[-1]
        last_t = float(last_row["t_ref_s"])
        last_samp = sample_at_time(last_row)

        alarms.append(AlarmEpisode(
            record_id=record_id,
            lead_id=lead_id,
            crit=str(alarm_crit),

            onset_expected_s=float(alarm_onset_expected_s),
            onset_actual_s=float(alarm_onset_actual_s),
            end_expected_s=float(last_t),
            end_actual_s=float(last_t),

            onset_expected_sample=int(round(alarm_onset_expected_s * fs)) if fs else None,
            onset_actual_sample=None,
            end_expected_sample=last_samp,
            end_actual_sample=last_samp,
        ))

    return alarms
