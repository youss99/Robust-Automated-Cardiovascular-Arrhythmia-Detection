from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
import wfdb


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
    onset_expected: int
    onset_actual: int
    end_expected: int
    end_actual: int
    crit: str


@dataclass
class FsmLeadInput:
    record: str
    lead_tag: str            # "ST0" / "ST1"
    samp_arr: list[int]      # from 16a
    st_arr: list[float]      # from 16a
    ecg_for_sqi: np.ndarray  # 1D
    r_peaks: np.ndarray      # 1D


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def extract_st_arrays_from_ann(st_ann, lead_prefix: str) -> tuple[list[int], list[float]]:
    """Extract (sample_idx, st_mV) from the '16a' aux_note for ST0/ST1 entries."""
    samp_arr: list[int] = []
    st_arr: list[float] = []

    for samp, aux in zip(st_ann.sample, st_ann.aux_note):
        if not aux or not aux.startswith(lead_prefix):
            continue

        rest = aux[len(lead_prefix):]
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


def get_rpeaks_from_atr(atr_ann) -> np.ndarray:
    """
    Extract beat locations from WFDB atr annotation.
    In LTST this usually corresponds to QRS locations.
    """
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


def alarms_to_df(alarms: list[Alarm], fs: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "record": a.record,
        "lead_tag": a.lead_tag,
        "lead_paper": a.lead_paper,

        "onset_expected_sample": a.onset_expected,
        "onset_actual_sample":   a.onset_actual,
        "end_expected_sample":   a.end_expected,
        "end_actual_sample":     a.end_actual,

        "onset_expected_s": a.onset_expected / fs,
        "onset_actual_s":   a.onset_actual   / fs,
        "end_expected_s":   a.end_expected   / fs,
        "end_actual_s":     a.end_actual     / fs,

        "duration_expected_s": (a.end_expected - a.onset_expected) / fs,
        "duration_actual_s":   (a.end_actual   - a.onset_actual)   / fs,

        "criterion": a.crit,
    } for a in alarms])


def alarms_count_df(alarms: list[Alarm]) -> pd.DataFrame:
    if not alarms:
        return pd.DataFrame(columns=["record", "lead_tag", "lead_paper", "n_alarms"])
    df = pd.DataFrame([{
        "record": a.record,
        "lead_tag": a.lead_tag,
        "lead_paper": a.lead_paper,
    } for a in alarms])
    out = (
        df.value_counts()
          .rename("n_alarms")
          .reset_index()
          .sort_values(["record", "lead_tag"], kind="mergesort")
          .reset_index(drop=True)
    )
    return out


# ------------------------------------------------
# SQI computation (unchanged)
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
    lead_paper = STTAG_TO_PAPER_LEAD[lead_tag]

    state = S.REST
    alarms: list[Alarm] = []

    t01 = t02 = tneg = None
    d01 = d02 = dneg = None

    cd_start = None
    cdc = None

    alarm_onset_expected = None
    alarm_onset_actual = None
    alarm_crit = None

    prev = None

    last_sqi_check_samp: Optional[int] = None
    SQI_Good: bool = True

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

        # For now: use samp as "current_r_peak"
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

        SQI_Good = (not np.isnan(sqi_db)) and (sqi_db >= sqi_threshold_db)
        last_sqi_check_samp = int(now_samp)

    for samp, st in zip(samp_arr, st_arr):
        samp = int(samp)

        maybe_update_sqi(samp)

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
                        alarm_onset_expected = int(t01 + T_WU_01 * fs)
                        alarm_onset_actual = int(samp)
                        alarm_crit = "≥ 0.1 mV · 10 min"
                    else:
                        state, alarm_crit = S.WU_01, "≥ 0.1 mV"

                else:
                    d02 = d02 + (samp - prev) / fs
                    done_02 = d02 >= T_WU_02
                    d01 = d01 + (samp - prev) / fs
                    done_01 = d01 >= T_WU_01

                    if done_01 or done_02:
                        crits = []
                        times = []

                        if done_01:
                            crits.append("≥ 0.1 mV · 10 min")
                            times.append(t01 + int(T_WU_01 * fs))
                        if done_02:
                            crits.append("≥ 0.2 mV · 1 min")
                            times.append(t02 + int(T_WU_02 * fs))

                        alarm_crit = " & ".join(crits)
                        alarm_onset_expected = int(min(times))
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
                            onset_expected=int(alarm_onset_expected),
                            onset_actual=int(alarm_onset_actual),
                            end_expected=end_expected,
                            end_actual=end_actual,
                            crit=alarm_crit
                        ))

                        state = S.REST
                        t01 = t02 = tneg = None
                        d01 = d02 = dneg = None
                        alarm_onset_expected = alarm_onset_actual = None
                        cd_start = cdc = None
                        alarm_crit = None

        prev = samp

    # EOF flush policy
    if samp_arr:
        last_samp = prev
        if state == S.ALARM and (alarm_onset_expected is not None) and (alarm_onset_actual is not None):
            alarms.append(Alarm(
                record=rec_id,
                lead_tag=lead_tag,
                lead_paper=lead_paper,
                onset_expected=int(alarm_onset_expected),
                onset_actual=int(alarm_onset_actual),
                end_expected=int(last_samp),
                end_actual=int(last_samp),
                crit=alarm_crit
            ))

    return alarms


# ------------------------------------------------
# Build inputs (gold-standard WFDB extraction)
# ------------------------------------------------
def build_inputs_from_wfdb(
    base_path: str,
    records: list[str],
    use_manual_omit: bool,
) -> list[FsmLeadInput]:
    inputs: list[FsmLeadInput] = []

    for record in records:
        print(f"⏳  {record}")
        record_path = f"{base_path}/{record}/{record}"

        st_ann = wfdb.rdann(record_path, "16a")
        rec = wfdb.rdrecord(record_path)
        atr = wfdb.rdann(record_path, "atr")

        r_peaks = get_rpeaks_from_atr(atr)

        for lead_tag in ("ST0", "ST1"):
            paper_lead = STTAG_TO_PAPER_LEAD[lead_tag]

            if use_manual_omit and ((record, paper_lead) in MANUAL_OMIT):
                print(f"   → {lead_tag} ({paper_lead}): SKIPPED (manual omit)")
                continue

            lead_idx = STTAG_TO_LEAD_IDX[lead_tag]
            if rec.p_signal.ndim != 2 or lead_idx >= rec.p_signal.shape[1]:
                print(f"   → {lead_tag} ({paper_lead}): SKIPPED (missing lead idx={lead_idx})")
                continue

            ecg_lead = rec.p_signal[:, lead_idx].astype(float)
            ecg_for_sqi = ecg_lead  # (optional) filter later

            samp_arr, st_arr = extract_st_arrays_from_ann(st_ann, lead_tag)

            inputs.append(FsmLeadInput(
                record=record,
                lead_tag=lead_tag,
                samp_arr=samp_arr,
                st_arr=st_arr,
                ecg_for_sqi=ecg_for_sqi,
                r_peaks=r_peaks,
            ))

    return inputs


# ------------------------------------------------
# Batch runner (prints per record+lead)
# ------------------------------------------------
def run_fsm_batch(
    inputs: Iterable[FsmLeadInput],
    fs: int,
    use_sqi: bool = True,
    sqi_threshold_db: float = 5.0,
    sqi_every_s: float = 5.0,
    sqi_window_s: float = 30.0,
    pre_sec: float = 0.35,
    post_sec: float = 0.35,
    min_beats: int = 3,
    verbose: bool = True,   # NEW
) -> list[Alarm]:
    all_alarms: list[Alarm] = []

    for item in inputs:
        rec_alarms = run_fsm_on_record(
            rec_id=item.record,
            lead_tag=item.lead_tag,
            samp_arr=item.samp_arr,
            st_arr=item.st_arr,
            ecg_sig_for_sqi=item.ecg_for_sqi,
            r_peaks=item.r_peaks,
            fs=fs,
            use_sqi=use_sqi,
            sqi_threshold_db=sqi_threshold_db,
            sqi_every_s=sqi_every_s,
            sqi_window_s=sqi_window_s,
            pre_sec=pre_sec,
            post_sec=post_sec,
            min_beats=min_beats,
        )

        if verbose:
            paper = STTAG_TO_PAPER_LEAD.get(item.lead_tag, item.lead_tag)
            print(f"   → {item.record} {item.lead_tag} ({paper}): {len(rec_alarms)} alarms")

        all_alarms.extend(rec_alarms)

    return all_alarms



# ------------------------------------------------
# CLI / Main
# ------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate alarms with optional SQI gating.")

    p.add_argument(
        "--gold_standard_mode",
        action="store_true",
        help="If set, build inputs from WFDB (16a + atr + signal). If not set, "
             "this module expects inputs to be provided programmatically (from your pipeline)."
    )

    p.add_argument("--base_path", type=str, default=BASE_PATH_DEFAULT, help="Path to ltst_records root.")
    p.add_argument("--fs", type=int, default=FS_DEFAULT, help="Sampling frequency (Hz). Default 250.")
    p.add_argument("--out_path", type=str, required=True, help="Output CSV path.")

    p.add_argument("--disable_sqi", action="store_true", help="Disable SQI gating (SQI always good).")
    p.add_argument("--sqi_threshold_db", type=float, default=5.0, help="SQI good threshold (dB).")
    p.add_argument("--sqi_every_s", type=float, default=5.0, help="Compute SQI every N seconds. Default 5s.")
    p.add_argument("--sqi_window_s", type=float, default=30.0, help="SQI trailing window length. Default 30s.")

    p.add_argument(
        "--no_manual_omit",
        action="store_true",
        help="If set, do NOT apply manual omit list."
    )

    return p


def main():
    args = build_arg_parser().parse_args()

    fs = int(args.fs)
    out_path = Path(args.out_path)

    use_sqi = not args.disable_sqi
    sqi_threshold_db = float(args.sqi_threshold_db)
    sqi_every_s = float(args.sqi_every_s)
    sqi_window_s = float(args.sqi_window_s)

    use_manual_omit = (not args.no_manual_omit)

    if not args.gold_standard_mode:
        raise SystemExit(
            "gold_standard_mode is OFF. This CLI does not build inputs.\n"
            "Call run_fsm_batch(inputs=..., fs=..., ...) from your pipeline instead."
        )

    # 1) Build inputs from WFDB (gold-standard extraction)
    inputs = build_inputs_from_wfdb(
        base_path=args.base_path,
        records=LTSTDB_RECORDS_DEFAULT,
        use_manual_omit=use_manual_omit,
    )

    print(f"\nBuilt {len(inputs)} (record, lead) inputs.")

    # 2) Run FSM batch (prints per pair)
    all_alarms = run_fsm_batch(
        inputs=inputs,
        fs=fs,
        use_sqi=use_sqi,
        sqi_threshold_db=sqi_threshold_db,
        sqi_every_s=sqi_every_s,
        sqi_window_s=sqi_window_s,
        verbose=True,
    )

    print("\n=== Alarms per record + lead ===")
    counts = alarms_count_df(all_alarms)
    if len(counts):
        print(counts.to_string(index=False))
    else:
        print("(no alarms)")

    print("\nTotal # of alarms:", len(all_alarms))

    # 3) Write output CSV
    df = alarms_to_df(all_alarms, fs=fs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n💾  Saved to {out_path}")


if __name__ == "__main__":
    main()
