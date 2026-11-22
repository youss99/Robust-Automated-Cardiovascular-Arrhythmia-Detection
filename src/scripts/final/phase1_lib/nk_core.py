#!/usr/bin/env python3
"""
SPEC 1 — shared scoring core for the NeuroKit fiducial baseline.

Everything here is a faithful port of the evaluators the LEARNED arm was scored
with, so the two arms are directly comparable:

  episode matcher  <- ltst_pipeline/evaluation.py :: evaluate_file  (lines 184-329)
  time-based       <- transfer_bundle/scripts/evaluate_time_based.py :: score

Nothing is tuned (SPEC 1 G1); this module only scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# SPEC 1 §A conventions
# ----------------------------------------------------------------------------
RECORDS_15 = [
    "s20011", "s20051", "s20061", "s20071", "s20081", "s20091", "s20101",
    "s20111", "s20131", "s20141", "s20201", "s20211", "s20221", "s20231",
    "s20241",
]
EXCLUDED_RECORD = "s20121"          # A1: 0 GS episodes, pinned to train
LEADS = ["II", "V2"]

# A8 — fold partition, identical to the learned arm
FOLDS = {
    "fold01": ["s20051", "s20211", "s20221"],
    "fold02": ["s20081", "s20101", "s20141"],
    "fold03": ["s20061", "s20091", "s20241"],
    "fold04": ["s20111", "s20131", "s20231"],
    "fold05": ["s20011", "s20071", "s20201"],
}

# A3 — the omitted-lead mask.
# MANUAL_OMIT is the force-omit list in ltst_pipeline/old_fsm_gs.py:29-36. It is
# applied STRUCTURALLY by the learned-arm alarm generator (new_fsm_ml.py:156-158),
# which emits no alarms at all on these pairs. Scoring NeuroKit without it made
# every NeuroKit alarm there an automatic FP — the known prior error.
MANUAL_OMIT = {
    ("s20091", "II"), ("s20121", "II"),
    ("s20081", "V2"), ("s20121", "V2"), ("s20221", "V2"), ("s20241", "V2"),
}

DDOF = 1  # A9 — sample std everywhere


# ----------------------------------------------------------------------------
# Gold standard
# ----------------------------------------------------------------------------
def load_gs(path: str) -> pd.DataFrame:
    gs = pd.read_csv(path)
    gs = gs[gs["record"] != EXCLUDED_RECORD].copy()
    gs["lead"] = gs["lead_paper"].astype(str)
    gs["record"] = gs["record"].astype(str)
    return gs.reset_index(drop=True)


def gs_zero_episode_pairs(gs: pd.DataFrame) -> set[tuple[str, str]]:
    """(record, lead) pairs among the 15 evaluated records with no GS episode."""
    have = {(r, l) for r, l in zip(gs["record"], gs["lead"])}
    return {(r, l) for r in RECORDS_15 for l in LEADS if (r, l) not in have}


# ----------------------------------------------------------------------------
# Alarm loading + masking
# ----------------------------------------------------------------------------
def load_alarms(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # a zero-byte alarm file legitimately means "no alarms at this threshold"
        return pd.DataFrame(columns=["record", "lead", "onset_actual_s", "end_actual_s"])
    if df.empty:
        return pd.DataFrame(columns=["record", "lead", "onset_actual_s", "end_actual_s"])
    df["record"] = df["record"].astype(str)
    df["lead"] = df["lead_paper"].astype(str)
    return df.reset_index(drop=True)


def apply_mask(df: pd.DataFrame, mask: set[tuple[str, str]]) -> pd.DataFrame:
    """Drop s20121 (A1) and every masked (record, lead) pair (A3)."""
    if df.empty:
        return df
    keep = df["record"].isin(RECORDS_15)
    pair = list(zip(df["record"], df["lead"]))
    keep &= pd.Series([p not in mask for p in pair], index=df.index)
    return df[keep].reset_index(drop=True)


def scope_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """FULL keeps everything; own-lead scopes keep one lead."""
    if scope == "FULL":
        return df
    if scope in ("own_II", "own_V2"):
        lead = scope.split("_")[1]
        return df[df["lead"] == lead].reset_index(drop=True) if not df.empty else df
    raise ValueError(scope)


# ----------------------------------------------------------------------------
# Episode-level matcher — port of evaluation.py::evaluate_file
# ----------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    tp: int
    fp: int
    fn: int
    ignored: int
    n_pred: int
    n_gs: int
    precision: float
    recall: float
    f1: float
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)


def match_episodes(gs: pd.DataFrame, pred: pd.DataFrame) -> EpisodeResult:
    """
    Asymmetric matching, exactly as evaluation.py:184-329:
      TP      = GS episodes hit by >=1 prediction (first-by-onset pred credited)
      FP      = predictions overlapping no GS episode
      FN      = GS episodes hit by no prediction
      Ignored = redundant pred<->GS overlap pairs on an already-credited GS
    Grouping is by (record, lead) — a lead-II prediction can never match a
    lead-V2 GS episode (SPEC 1 F4).
    """
    detected: dict[int, list[dict]] = {}
    pred_hit: set[int] = set()
    rows: list[dict] = []

    if not pred.empty:
        for pi, pr in pred.iterrows():
            ps, pe = float(pr["onset_actual_s"]), float(pr["end_actual_s"])
            pdur = pe - ps
            for gi, gr in gs.iterrows():
                if pr["record"] != gr["record"] or pr["lead"] != gr["lead"]:
                    continue
                gsx, gex = float(gr["onset_actual_s"]), float(gr["end_actual_s"])
                gdur = gex - gsx
                os_, oe = max(ps, gsx), min(pe, gex)
                if oe <= os_:
                    continue
                ov = oe - os_
                # evaluation.py uses overlap_threshold=0.0 -> strict >
                frac_min = ov / min(pdur, gdur) if min(pdur, gdur) > 0 else 0.0
                if frac_min <= 0.0:
                    continue
                union = pdur + gdur - ov
                pred_hit.add(pi)
                info = {
                    "pred_idx": pi, "gs_idx": gi,
                    "record": gr["record"], "lead": gr["lead"],
                    "pred_onset_s": ps, "pred_end_s": pe, "pred_duration_s": pdur,
                    "gs_onset_s": gsx, "gs_end_s": gex, "gs_duration_s": gdur,
                    "match_type": "TP",
                    "overlap_duration_s": ov,
                    "jaccard": ov / union if union > 0 else 0.0,
                    "onset_error_s": ps - gsx,
                    "end_error_s": pe - gex,
                    "duration_diff_s": pdur - gdur,
                }
                detected.setdefault(gi, []).append(info)
                rows.append(info)

    ignored = 0
    for gi, prs in detected.items():
        if len(prs) <= 1:
            continue
        prs.sort(key=lambda p: p["pred_onset_s"])
        for extra in prs[1:]:
            extra["match_type"] = "ignored"
            ignored += 1

    n_pred = len(pred)
    tp = len(detected)
    fp = n_pred - len(pred_hit)
    fn = len(gs) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return EpisodeResult(tp, fp, fn, ignored, n_pred, len(gs), p, r, f1,
                         pd.DataFrame(rows))


# ----------------------------------------------------------------------------
# Time-based scorer — port of evaluate_time_based.py::score
# ----------------------------------------------------------------------------
def merge_intervals(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    iv = sorted((a, b) for a, b in iv if b > a)
    if not iv:
        return []
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_len(iv) -> float:
    return float(sum(b - a for a, b in iv))


def intersect_len(x, y) -> float:
    i = j = 0
    tot = 0.0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            tot += hi - lo
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def intervals_by_group(df: pd.DataFrame) -> dict[tuple[str, str], list]:
    g: dict[tuple[str, str], list] = {}
    if df.empty:
        return g
    for _, r in df.iterrows():
        a, b = float(r["onset_actual_s"]), float(r["end_actual_s"])
        if np.isfinite(a) and np.isfinite(b) and b > a:
            g.setdefault((r["record"], r["lead"]), []).append((a, b))
    return {k: merge_intervals(v) for k, v in g.items()}


def score_time(gs: pd.DataFrame, pred: pd.DataFrame, durations: dict[str, float],
               universe: list[tuple[str, str]]) -> dict:
    """
    Duration confusion matrix over merged intervals (C2).
    `universe` fixes the (record, lead) pairs that contribute TN — this is the
    specificity denominator and it is what the mask (A3) controls.
    """
    gi, pi = intervals_by_group(gs), intervals_by_group(pred)
    TP = FP = FN = TN = 0.0
    gs_raw = pred_raw = 0.0
    for rec, lead in universe:
        dur = durations.get(rec)
        if dur is None:
            continue
        g, p = gi.get((rec, lead), []), pi.get((rec, lead), [])
        gl, pl = total_len(g), total_len(p)
        inter = intersect_len(g, p)
        TP += inter
        FP += pl - inter
        FN += gl - inter
        TN += max(0.0, dur - (gl + pl - inter))
        gs_raw += gl
        pred_raw += pl

    sens = TP / (TP + FN) if TP + FN > 0 else np.nan
    spec = TN / (TN + FP) if TN + FP > 0 else np.nan
    ppv = TP / (TP + FP) if TP + FP > 0 else np.nan
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan
    f1t = (2 * ppv * sens / (ppv + sens)) if (np.isfinite(ppv) and np.isfinite(sens)
                                              and (ppv + sens) > 0) else np.nan
    return {
        "time_TP_s": TP, "time_FP_s": FP, "time_FN_s": FN, "time_TN_s": TN,
        "time_F1": f1t, "sensitivity": sens, "PPV_time": ppv,
        "specificity": spec,
        "balanced_acc": (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan,
        "IoU_whole_record": iou,
        # A9: coverage = total predicted alarm duration / total GS duration.
        # _merged uses the union of intervals (C2 basis); _raw is the plain sum
        # of alarm durations, which is what the learned arm's summary.csv used.
        "coverage_merged": pred_raw / gs_raw if gs_raw > 0 else np.nan,
        "gs_duration_s": gs_raw, "pred_duration_s": pred_raw,
    }


def coverage_raw(gs: pd.DataFrame, pred: pd.DataFrame) -> float:
    gd = float((gs["end_actual_s"] - gs["onset_actual_s"]).sum()) if len(gs) else 0.0
    pdur = float((pred["end_actual_s"] - pred["onset_actual_s"]).sum()) if len(pred) else 0.0
    return pdur / gd if gd > 0 else np.nan


# ----------------------------------------------------------------------------
# Universes
# ----------------------------------------------------------------------------
def universe_pairs(scope: str, mask: set[tuple[str, str]],
                   records: list[str] | None = None) -> list[tuple[str, str]]:
    recs = records if records is not None else RECORDS_15
    leads = LEADS if scope == "FULL" else [scope.split("_")[1]]
    return [(r, l) for r in recs for l in leads if (r, l) not in mask]


# ----------------------------------------------------------------------------
# Temporal quality (C3) — over matched TP pairs only (biased subset)
# ----------------------------------------------------------------------------
def temporal_stats(pairs: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    tp = pairs[pairs["match_type"] == "TP"] if len(pairs) else pairs
    for col, name in [("jaccard", "jaccard"), ("onset_error_s", "onset_error_s"),
                      ("end_error_s", "end_error_s"), ("duration_diff_s", "duration_diff_s")]:
        s = tp[col].dropna() if len(tp) else pd.Series(dtype=float)
        out[f"median_{name}"] = float(s.median()) if len(s) else np.nan
        out[f"mean_{name}"] = float(s.mean()) if len(s) else np.nan
    out["n_tp_pairs"] = int(len(tp))
    return out


def mean_sd(vals: list[float]) -> tuple[float, float]:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return np.nan, np.nan
    sd = float(np.std(a, ddof=DDOF)) if a.size > 1 else np.nan
    return float(np.mean(a)), sd
