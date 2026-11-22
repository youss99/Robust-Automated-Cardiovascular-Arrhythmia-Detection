#!/usr/bin/env python3
"""
SPEC 3 — alarm performance and temporal quality for the learned arms.

Scores ALREADY-GENERATED alarm CSVs; no inference is re-run (the alarm CSVs are
the pipeline's own output, and SPEC 2 C4 established that re-deriving the
predictions reproduces their counts exactly). No PTB-XL arm.

Sources
  clean, noisy      transfer_bundle/results/alarm_eval_results_cv5_optB_thesis_small/
  noisy + SQI 5 dB  sqi_results/noisy_sqi_FILTERED/          <- paper-correct gate
  noisy + SQI (old) sqi_results/superseded_UNFILTERED/noisy_sqi/  <- for D1 continuity only
  SQI sweeps        sqi_results/sweep_FILTERED{,_union,_lead_II,_lead_V2}/ (pretrained)
                    sqi_results/sweep_FILTERED_supervised/                 (supervised)
                    -> scored by sweeps_all.py, which overwrites C6_sqi_sweep_curves.csv
  NeuroKit          SPEC 1 outputs, for the C9(b) comparison

Evaluators are the SPEC 1 core (results/neurokit/_code/nk_core.py) so every number
here is on exactly the same basis as NEUROKIT_ARM_FINAL.md.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/student/GIT/results/alarms")
OUT.mkdir(parents=True, exist_ok=True)
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/results/neurokit/_code")
import nk_core as K  # noqa: E402

TB = Path("/home/student/GIT/transfer_bundle/results/alarm_eval_results_cv5_optB_thesis_small")
SQI = Path("/home/student/GIT/sqi_results")
NKD = Path("/home/student/Desktop/Research Data/DATASET NK/NeuroKit Alarms")
NK_SWEEP = Path("/home/student/GIT/results/neurokit/_sweep_faithful")
GS_PATH = "/home/student/GIT/transfer_bundle/gs_per_beat.csv"
DUR_CSV = "/home/student/GIT/results/neurokit/_sweep/record_durations.csv"

# mode -> (transfer_bundle dir name, scopes)   A6: two_lead/union FULL only
MODES = {
    "two_lead": ("two_lead", ["FULL"]),
    "lead_II": ("lead_II", ["FULL", "own_II"]),
    "lead_V2": ("lead_V2", ["FULL", "own_V2"]),
    "union": ("single_lead_union", ["FULL"]),
}
ARMS = {  # arm -> mode -> variant dir
    "supervised": {"two_lead": "supervised", "lead_II": "supervised",
                   "lead_V2": "supervised", "union": "supervised"},
    "Unified-pretrained": {"two_lead": "pretrained_chapman_ii_v2",
                           "lead_II": "pretrained_chapman_ii_only",
                           "lead_V2": "pretrained_chapman_v2_only",
                           "union": "pretrained"},
}
CONDS = ["clean", "noisy", "noisy_sqi_5dB", "noisy_sqi_6dB"]
SEED = 20260805
N_BOOT = 10_000
_LOG: list[str] = []


def log(m: str = "") -> None:
    print(m, flush=True)
    _LOG.append(m)


def alarm_csv(arm: str, mode: str, cond: str) -> Path | None:
    vd = ARMS[arm][mode]
    md = MODES[mode][0]
    if cond in ("clean", "noisy"):
        base = TB / md / vd / cond
    elif cond == "noisy_sqi_5dB":
        base = SQI / "noisy_sqi_FILTERED" / md / vd / "noisy_sqi"
    elif cond == "noisy_sqi_5dB_UNFILTERED":
        base = SQI / "superseded_UNFILTERED" / "noisy_sqi" / md / vd / "noisy_sqi"
        if not base.exists():
            base = SQI / "noisy_sqi_UNFILTERED_old" / md / vd / "noisy_sqi"
    else:
        return None
    for n in ("ml_alarms_pooled.csv", "ml_alarms_union.csv"):
        if (base / n).exists():
            return base / n
    return None


def alarm_frame(arm: str, mode: str, cond: str) -> pd.DataFrame | None:
    """Raw (unmasked) alarm set for a condition.

    `noisy_sqi_6dB` — Abdelazez's published operating point, and the paper's PRIMARY
    matched comparison — has no fixed-threshold run. It is taken from the corrected
    sweeps for BOTH arms via sweeps_all.alarms(), which fold-filters and concatenates
    where a mode has no directly-generated file.
    """
    if cond == "noisy_sqi_6dB":
        from sweeps_all import alarms as sweep_alarms
        return sweep_alarms(arm, mode, 6)
    ac = alarm_csv(arm, mode, cond)
    return None if ac is None else K.load_alarms(str(ac))


def gs_scope(gs: pd.DataFrame, scope: str) -> pd.DataFrame:
    return gs if scope == "FULL" else gs[gs.lead == scope.split("_")[1]].reset_index(drop=True)


def per_record_components(gs: pd.DataFrame, pred: pd.DataFrame, durations, scope, mask,
                          records) -> pd.DataFrame:
    """Additive per-record components — the basis for the C9 bootstrap/jackknife."""
    rows = []
    for rec in records:
        g = gs[gs.record == rec].reset_index(drop=True)
        p = pred[pred.record == rec].reset_index(drop=True) if not pred.empty else pred
        r = K.match_episodes(g, p)
        tm = K.score_time(g, p, durations, K.universe_pairs(scope, mask, records=[rec]))
        rows.append({"record": rec, "n_GS": r.n_gs, "TP": r.tp, "FP": r.fp, "FN": r.fn,
                     "Ignored": r.ignored, "n_pred": r.n_pred,
                     "tTP": tm["time_TP_s"], "tFP": tm["time_FP_s"],
                     "tFN": tm["time_FN_s"], "tTN": tm["time_TN_s"],
                     "gs_dur": tm["gs_duration_s"], "pred_dur": tm["pred_duration_s"]})
    return pd.DataFrame(rows)


def metrics_from_components(c: pd.DataFrame) -> dict:
    TP, FP, FN = c.TP.sum(), c.FP.sum(), c.FN.sum()
    p = TP / (TP + FP) if TP + FP else 0.0
    r = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    tTP, tFP, tFN, tTN = c.tTP.sum(), c.tFP.sum(), c.tFN.sum(), c.tTN.sum()
    sens = tTP / (tTP + tFN) if tTP + tFN else np.nan
    ppv = tTP / (tTP + tFP) if tTP + tFP else np.nan
    spec = tTN / (tTN + tFP) if tTN + tFP else np.nan
    tf1 = 2 * ppv * sens / (ppv + sens) if np.isfinite(ppv) and np.isfinite(sens) and ppv + sens else np.nan
    return {"TP": int(TP), "FP": int(FP), "FN": int(FN), "Ignored": int(c.Ignored.sum()),
            "n_predicted_alarms": int(c.n_pred.sum()),
            "n_notifications": int(TP + FP + c.Ignored.sum()),
            "precision": p, "recall": r, "F1": f1,
            "time_F1": tf1, "sensitivity": sens, "PPV_time": ppv, "specificity": spec,
            "balanced_acc": (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan,
            "IoU_whole_record": tTP / (tTP + tFP + tFN) if (tTP + tFP + tFN) else np.nan,
            "coverage": c.pred_dur.sum() / c.gs_dur.sum() if c.gs_dur.sum() else np.nan}


def main() -> None:
    gs = K.load_gs(GS_PATH)
    durations = {r["record"]: float(r["duration_s"])
                 for _, r in pd.read_csv(DUR_CSV).iterrows()}
    mask = set(K.MANUAL_OMIT)
    total_h = sum(durations[r] for r in K.RECORDS_15) / 3600.0

    log("=" * 78)
    log("SPEC 3 — learned arms: alarm performance and temporal quality")
    log("=" * 78)
    log(f"GS sha256 {hashlib.sha256(Path(GS_PATH).read_bytes()).hexdigest()[:16]}… — "
        f"{len(gs)} episodes; monitored {total_h:.4f} h")
    log(f"mask (A3): {sorted(f'{r}/{l}' for r, l in mask)}")

    # ---------------------------------------------------------------- scoring
    comps: dict[tuple, pd.DataFrame] = {}
    c1, c2, c3, c4, c7, c8 = [], [], [], [], [], []
    c1f, c2f = [], []

    for arm in ARMS:
        for mode, (_md, scopes) in MODES.items():
            for cond in CONDS + ["noisy_sqi_5dB_UNFILTERED"]:
                fr = alarm_frame(arm, mode, cond)
                if fr is None:
                    continue
                raw = K.apply_mask(fr, mask)
                for scope in scopes:
                    g = gs_scope(gs, scope)
                    p = K.scope_filter(raw, scope)
                    comp = per_record_components(g, p, durations, scope, mask, K.RECORDS_15)
                    comps[(arm, mode, cond, scope)] = comp
                    m = metrics_from_components(comp)
                    base = {"arm": arm, "mode": mode, "condition": cond, "scope": scope}
                    r = K.match_episodes(g, p)
                    ts = K.temporal_stats(r.pairs)
                    # C4(b): predicted alarms that are not the credited-first TP
                    burden_b = r.n_pred - r.tp if r.n_pred >= r.tp else np.nan
                    c1.append({**base, "aggregation": "pooled", **{k: m[k] for k in
                               ("TP", "FP", "FN", "Ignored", "n_predicted_alarms",
                                "n_notifications", "precision", "recall", "F1")},
                               "n_gs_episodes": r.n_gs})
                    c2.append({**base, "aggregation": "pooled", **{k: m[k] for k in
                               ("time_F1", "sensitivity", "PPV_time", "specificity",
                                "balanced_acc", "IoU_whole_record", "coverage")},
                               "record_duration_def": "wfdb header sig_len/fs"})
                    c3.append({**base, **ts})
                    c4.append({**base, "FP": m["FP"], "Ignored": m["Ignored"],
                               "burden_a_FP_plus_Ignored": m["FP"] + m["Ignored"],
                               "burden_b_alarms_not_credited_TP": burden_b,
                               "burden_reconcile_a_minus_b": (m["FP"] + m["Ignored"]) - burden_b,
                               "TP": m["TP"], "n_predicted_alarms": m["n_predicted_alarms"],
                               "TP_per_burden": m["TP"] / (m["FP"] + m["Ignored"])
                               if (m["FP"] + m["Ignored"]) else np.nan,
                               "Ignored_per_TP": m["Ignored"] / m["TP"] if m["TP"] else np.nan})
                    notif = m["n_notifications"]
                    c8.append({**base, "monitored_h": total_h,
                               "false_alarms_per_24h_per_patient": m["FP"] * 24.0 / total_h,
                               "notifications_per_24h_per_patient": notif * 24.0 / total_h,
                               "notifications_per_true_episode": notif / r.n_gs if r.n_gs else np.nan})
                    for _, rr in comp.iterrows():
                        c7.append({**base, "record": rr.record, "n_GS": int(rr.n_GS),
                                   "TP": int(rr.TP), "FP": int(rr.FP), "FN": int(rr.FN),
                                   "Ignored": int(rr.Ignored),
                                   "n_predicted_alarms": int(rr.n_pred),
                                   "precision": rr.TP / (rr.TP + rr.FP) if rr.TP + rr.FP else 0.0,
                                   "recall": rr.TP / (rr.TP + rr.FN) if rr.TP + rr.FN else 0.0})
                    # per-fold (A5/A7) — fold episode count carried on every row
                    for f, recs in K.FOLDS.items():
                        sub = comp[comp.record.isin(recs)]
                        mf = metrics_from_components(sub)
                        bf = {**base, "aggregation": "per_fold", "fold": f,
                              "n_gs_episodes_in_fold": int(sub.n_GS.sum()),
                              "test_records": ";".join(recs)}
                        c1f.append({**bf, **{k: mf[k] for k in
                                    ("TP", "FP", "FN", "Ignored", "n_predicted_alarms",
                                     "n_notifications", "precision", "recall", "F1")}})
                        c2f.append({**bf, **{k: mf[k] for k in
                                    ("time_F1", "sensitivity", "PPV_time", "specificity",
                                     "balanced_acc", "IoU_whole_record", "coverage")}})
        log(f"  scored {arm}")

    pd.DataFrame(c1).to_csv(OUT / "C1_alarm_metrics_pooled.csv", index=False)
    pd.DataFrame(c1f).to_csv(OUT / "C1_alarm_metrics_per_fold.csv", index=False)
    pd.DataFrame(c2).to_csv(OUT / "C2_time_metrics_pooled.csv", index=False)
    pd.DataFrame(c2f).to_csv(OUT / "C2_time_metrics_per_fold.csv", index=False)
    pd.DataFrame(c3).to_csv(OUT / "C3_temporal_quality.csv", index=False)
    pd.DataFrame(c4).to_csv(OUT / "C4_burden.csv", index=False)
    pd.DataFrame(c7).to_csv(OUT / "C7_per_record_breakdown.csv", index=False)
    pd.DataFrame(c8).to_csv(OUT / "C8_clinical_units.csv", index=False)

    # per-fold summary, ddof=1
    F1 = pd.DataFrame(c1f)
    F2 = pd.DataFrame(c2f)
    rows = []
    for keys, g in F1.groupby(["arm", "mode", "condition", "scope"], sort=False):
        g2 = F2
        for k, v in zip(["arm", "mode", "condition", "scope"], keys):
            g2 = g2[g2[k] == v]
        row = dict(zip(["arm", "mode", "condition", "scope"], keys))
        row |= {"aggregation": "per_fold_summary", "n_folds": len(g), "ddof": 1,
                "fold_episode_counts": ";".join(
                    f"{r.fold}:{r.n_gs_episodes_in_fold}" for r in g.itertuples())}
        for col, src in [("F1", g.F1), ("recall", g.recall), ("precision", g.precision),
                         ("time_F1", g2.time_F1), ("IoU_whole_record", g2.IoU_whole_record)]:
            mu, sd = K.mean_sd(list(src))
            row[f"{col}_mean"], row[f"{col}_sd_ddof1"] = mu, sd
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "C1_C2_per_fold_summary.csv", index=False)

    # ------------------------------------------------- C5 per-lead breakdown
    c5 = []
    for arm in ARMS:
        for mode in MODES:
            for cond in CONDS:
                fr = alarm_frame(arm, mode, cond)
                if fr is None:
                    continue
                raw = K.apply_mask(fr, mask)
                for lead in ("II", "V2"):
                    g = gs[gs.lead == lead].reset_index(drop=True)
                    p = raw[raw.lead == lead].reset_index(drop=True) if not raw.empty else raw
                    r = K.match_episodes(g, p)
                    tm = K.score_time(g, p, durations,
                                      K.universe_pairs(f"own_{lead}", mask))
                    c5.append({"arm": arm, "mode": mode, "condition": cond,
                               "gs_lead": lead, "scope": f"own_{lead}",
                               "n_gs_episodes": r.n_gs, "TP": r.tp, "FP": r.fp,
                               "FN": r.fn, "Ignored": r.ignored,
                               "n_predicted_alarms": r.n_pred,
                               "precision": r.precision, "recall": r.recall, "F1": r.f1,
                               "time_F1": tm["time_F1"], "sensitivity": tm["sensitivity"],
                               "PPV_time": tm["PPV_time"],
                               "IoU_whole_record": tm["IoU_whole_record"]})
    pd.DataFrame(c5).to_csv(OUT / "C5_per_lead_breakdown.csv", index=False)

    # --------------------------------------------------------- C6 sweeps
    src = Path("/home/student/GIT/results/neurokit/C5c_learned_sqi_sweeps_all_modes.csv")
    if src.exists():
        sw = pd.read_csv(src)
        sw.insert(0, "arm", "Unified-pretrained")
        sw.to_csv(OUT / "C6_sqi_sweep_curves.csv", index=False)

    # per-record components feed the C9 bootstrap/jackknife in confidence_intervals.py
    import pickle
    with open(OUT / "_components.pkl", "wb") as fh:
        pickle.dump({"comps": comps, "total_h": total_h}, fh)
    (OUT / "run_log.txt").write_text("\n".join(_LOG))
    log("")
    log(f"wrote {len(list(OUT.glob('*.csv')))} CSVs")
    return comps, gs, durations, mask, total_h


if __name__ == "__main__":
    main()
