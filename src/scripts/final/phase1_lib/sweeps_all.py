#!/usr/bin/env python3
"""
SPEC 3 C6 — SQI sweep curves for BOTH arms, all four modes.

The supervised sweeps have now been imported, closing the gap flagged in the
first pass. Sources:

  Unified-pretrained
    two_lead  sweep_FILTERED/_pooled/pooled_sqi<T>.csv          (already pooled)
    union     sweep_FILTERED_union/union_sqi<T>.csv             (already pooled)
    lead_II   sweep_FILTERED_lead_II/fold0N/ml_alarms_sqi<T>.csv    (fold-filter here)
    lead_V2   sweep_FILTERED_lead_V2/fold0N/ml_alarms_sqi<T>.csv    (fold-filter here)
  supervised
    two_lead  sweep_FILTERED_supervised/two_lead/fold0N/ml_alarms_sqi<T>.csv
    lead_II   sweep_FILTERED_supervised/lead_II/fold0N/...
    lead_V2   sweep_FILTERED_supervised/lead_V2/fold0N/...
    union     lead_II u lead_V2, concatenated after fold-filtering (D4 shows this
              reproduces a directly-generated union exactly)

Per-fold files hold alarms for ALL 15 records because each fold's model is run
over everything, so each fold is filtered to its own held-out patients before
concatenation. Every curve is cross-checked against the bundle's own
sqi_sweep_<mode>_pooled_metrics.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/student/GIT/results/alarms")
SQI = Path("/home/student/GIT/sqi_results")
pass  # vendored (phase1_lib is on sys.path); was: sys.path.insert(0, "/home/student/GIT/results/neurokit/_code")
import nk_core as K  # noqa: E402

GS_PATH = "/home/student/GIT/transfer_bundle/gs_per_beat.csv"
DUR_CSV = "/home/student/GIT/results/neurokit/_sweep/record_durations.csv"
SUP = SQI / "sweep_FILTERED_supervised"

SCOPE = {"two_lead": "FULL", "union": "FULL", "lead_II": "own_II", "lead_V2": "own_V2"}
REF = {
    ("Unified-pretrained", "two_lead"): SQI / "sweep_FILTERED/_pooled/sqi_sweep_pooled_metrics.csv",
    ("Unified-pretrained", "union"): SQI / "sweep_FILTERED_union/sqi_sweep_union_pooled_metrics.csv",
    ("Unified-pretrained", "lead_II"): SQI / "sweep_FILTERED_union/sqi_sweep_lead_II_pooled_metrics.csv",
    ("Unified-pretrained", "lead_V2"): SQI / "sweep_FILTERED_union/sqi_sweep_lead_V2_pooled_metrics.csv",
    ("supervised", "two_lead"): SUP / "_metrics/sqi_sweep_two_lead_pooled_metrics.csv",
    ("supervised", "union"): SUP / "_metrics/sqi_sweep_union_pooled_metrics.csv",
    ("supervised", "lead_II"): SUP / "_metrics/sqi_sweep_lead_II_pooled_metrics.csv",
    ("supervised", "lead_V2"): SUP / "_metrics/sqi_sweep_lead_V2_pooled_metrics.csv",
}


def fold_concat(dirs: list[Path], t: int) -> pd.DataFrame | None:
    parts = []
    for d in dirs:
        for fold, recs in K.FOLDS.items():
            p = d / fold / f"ml_alarms_sqi{t}.csv"
            if not p.exists():
                return None
            df = K.load_alarms(str(p))
            if not df.empty:
                parts.append(df[df.record.isin(recs)])
    if not parts:
        return pd.DataFrame(columns=["record", "lead", "onset_actual_s", "end_actual_s"])
    return pd.concat(parts, ignore_index=True)


def alarms(arm: str, mode: str, t: int) -> pd.DataFrame | None:
    if arm == "Unified-pretrained":
        if mode == "two_lead":
            p = SQI / f"sweep_FILTERED/_pooled/pooled_sqi{t}.csv"
            return K.load_alarms(str(p)) if p.exists() else None
        if mode == "union":
            p = SQI / f"sweep_FILTERED_union/union_sqi{t}.csv"
            return K.load_alarms(str(p)) if p.exists() else None
        return fold_concat([SQI / f"sweep_FILTERED_{mode}"], t)
    # supervised
    if mode == "union":
        return fold_concat([SUP / "lead_II", SUP / "lead_V2"], t)
    return fold_concat([SUP / mode], t)


def main() -> None:
    gs = K.load_gs(GS_PATH)
    durations = {r["record"]: float(r["duration_s"]) for _, r in pd.read_csv(DUR_CSV).iterrows()}
    mask = set(K.MANUAL_OMIT)

    rows, checks = [], []
    for arm in ("Unified-pretrained", "supervised"):
        for mode in ("two_lead", "union", "lead_II", "lead_V2"):
            scope = SCOPE[mode]
            g = gs if scope == "FULL" else gs[gs.lead == scope.split("_")[1]].reset_index(drop=True)
            uni = K.universe_pairs(scope, mask)
            refp = REF.get((arm, mode))
            ref = pd.read_csv(refp).set_index("sqi_threshold_db") if refp and refp.exists() else None
            n_ok = n_cmp = 0
            for t in range(-20, 26):
                pred = alarms(arm, mode, t)
                if pred is None:
                    continue
                pred = K.scope_filter(K.apply_mask(pred, mask), scope)
                r = K.match_episodes(g, pred)
                tm = K.score_time(g, pred, durations, uni)
                rows.append({"arm": arm, "mode": mode, "scope": scope, "threshold_dB": t,
                             "TP": r.tp, "FP": r.fp, "FN": r.fn, "Ignored": r.ignored,
                             "n_predicted_alarms": r.n_pred, "precision": r.precision,
                             "recall": r.recall, "F1": r.f1, "time_F1": tm["time_F1"],
                             "sensitivity": tm["sensitivity"], "PPV_time": tm["PPV_time"],
                             "IoU_whole_record": tm["IoU_whole_record"],
                             "coverage": tm["coverage_merged"],
                             "burden_FP_plus_Ignored": r.fp + r.ignored})
                if ref is not None and t in ref.index:
                    n_cmp += 1
                    n_ok += int(int(ref.loc[t, "TP"]) == r.tp and int(ref.loc[t, "FP"]) == r.fp)
            if ref is not None:
                checks.append({"arm": arm, "mode": mode, "thresholds_compared": n_cmp,
                               "thresholds_matching": n_ok, "exact": n_ok == n_cmp})

    S = pd.DataFrame(rows)
    S.to_csv(OUT / "C6_sqi_sweep_curves.csv", index=False)
    C = pd.DataFrame(checks)
    C.to_csv(OUT / "C6_sweep_crosscheck.csv", index=False)

    print(f"scored {len(S)} (arm, mode, threshold) cells\n")
    print("cross-check vs the bundles' own pooled metrics:")
    for r in C.itertuples():
        print(f"  {r.arm:19s} {r.mode:9s} {r.thresholds_matching}/{r.thresholds_compared} "
              f"{'EXACT' if r.exact else '*** MISMATCH ***'}")

    print("\noperating points (episode F1):")
    print(f"{'arm':20s}{'mode':10s}{'gate off':>9s}{'5 dB':>8s}{'6 dB':>8s}"
          f"{'Abdelazez rule':>16s}{'T* (oracle)':>14s}")
    ops = []
    for (arm, mode), s in S.groupby(["arm", "mode"], sort=False):
        s = s.sort_values("threshold_dB")
        off = s[s.threshold_dB == -20].iloc[0]
        keep = s[s.TP >= off.TP]
        abd = keep.loc[keep.precision.idxmax()] if len(keep) else None
        best = s.loc[s.F1.idxmax()]
        f5 = s[s.threshold_dB == 5].F1.iloc[0]
        f6 = s[s.threshold_dB == 6].F1.iloc[0]
        dom = s[(s.TP >= s[s.threshold_dB == 5].TP.iloc[0]) &
                (s.FP <= s[s.threshold_dB == 5].FP.iloc[0]) & (s.threshold_dB != 5) &
                ((s.TP > s[s.threshold_dB == 5].TP.iloc[0]) |
                 (s.FP < s[s.threshold_dB == 5].FP.iloc[0]))]
        print(f"{arm:20s}{mode:10s}{off.F1:>9.4f}{f5:>8.4f}{f6:>8.4f}"
              f"{int(abd.threshold_dB):>10d} dB {abd.F1:>.4f}"
              f"{int(best.threshold_dB):>8d} dB {best.F1:.4f}")
        ops.append({"arm": arm, "mode": mode, "scope": SCOPE[mode],
                    "F1_gate_off": off.F1, "F1_5dB": f5, "F1_6dB": f6,
                    "abdelazez_rule_dB": int(abd.threshold_dB), "abdelazez_rule_F1": abd.F1,
                    "Tstar_dB": int(best.threshold_dB), "Tstar_F1": best.F1,
                    "thresholds_dominating_5dB": ";".join(map(str, sorted(dom.threshold_dB))) or "none"})
    pd.DataFrame(ops).to_csv(OUT / "C6_operating_points.csv", index=False)


if __name__ == "__main__":
    main()
