#!/usr/bin/env python3
"""
Trace-level diagnosis of WHY pretraining changes alarm behaviour (two-lead model).

Tests two hypotheses that the alarm-level metrics could only suggest indirectly:

  H1 SHRINKAGE — MSE training pulls predictions toward the conditional mean,
     compressing the ST excursions the FSM thresholds on. Measured by the OLS
     slope of pred~GT (<1 == shrinkage), the std ratio, and the retained
     amplitude inside the excursion bands (|GT| >= 0.1 / 0.2 mV).

  H2 SMOOTHNESS — masked-patch pretraining biases the model toward smooth,
     temporally-correlated output, so the trace crosses thresholds earlier,
     holds longer, and dips out less often. Measured by lag-1 autocorrelation
     of the ST trace, mean |Δ| between consecutive segments, and the number of
     threshold crossings / fraction of time above threshold.

CV-honest: each fold's model is run ONLY on that fold's held-out test patients,
matching how the pooled alarm metrics were produced.

Ground truth ST comes from the LTSTDB '.16a' annotations (same source the
gold-standard FSM uses), nearest-neighbour aligned to each 10 s segment.

Output: one row per (variant, lead) with all statistics, plus a per-record CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

NEUROKIT_ROOT = "/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline"
PATCHECG_ROOT = "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection"
sys.path.insert(0, NEUROKIT_ROOT)

from ltst_pipeline.ml_inference import (  # noqa: E402
    SEGMENT_SECONDS,
    load_patchecg_model,
    predict_st_per_segment,
)
from ltst_pipeline.fsm_gold_standard import extract_st_arrays_from_ann  # noqa: E402
from ltst_pipeline.old_fsm_gs import (  # noqa: E402
    MANUAL_OMIT,
    STTAG_TO_LEAD_IDX,
    STTAG_TO_PAPER_LEAD,
)

CKPT_ROOT = f"{PATCHECG_ROOT}/results/fine-tune/saved_models/patient-segment-regression"
MANIFEST_DIR = f"{PATCHECG_ROOT}/src/scripts/custom_dataset/cv5_optB_fold_manifests"
LTST_RECORDS = "/lustre06/project/6039801/youss99/Project_1/Datasets/ltst_records"
FS = 250
FOLDS = ["01", "02", "03", "04", "05"]

VARIANTS = {
    "supervised":   "ltstdb_median_cv5_optB_two_lead_regression_supervised_fold{f}_thesis_small",
    "pre_unified":  "ltstdb_median_cv5_optB_two_lead_regression_pretrained_chapman_ii_v2_fold{f}_thesis_small",
    "pre_ptbxl":    "ltstdb_median_cv5_optB_two_lead_regression_pretrained_ptbxl_ii_v2_fold{f}_thesis_small",
}


def fold_test_patients() -> dict[str, list[str]]:
    out = {}
    for f in FOLDS:
        df = pd.read_csv(f"{MANIFEST_DIR}/fold{f}.csv")
        out[f] = df[df["split"].str.lower() == "test"]["patient_id"].astype(str).tolist()
    return out


def gt_st_for_segments(record: str, lead_tag: str, samp_arr: np.ndarray) -> np.ndarray:
    """Nearest-neighbour align the .16a ground-truth ST series onto segment anchors."""
    ann = wfdb.rdann(f"{LTST_RECORDS}/{record}/{record}", "16a")
    g_samp, g_st = extract_st_arrays_from_ann(ann, lead_tag)
    if not g_samp:
        return np.full(len(samp_arr), np.nan)
    g_samp = np.asarray(g_samp, dtype=float)
    g_st = np.asarray(g_st, dtype=float)
    order = np.argsort(g_samp)
    g_samp, g_st = g_samp[order], g_st[order]
    idx = np.searchsorted(g_samp, samp_arr)
    idx = np.clip(idx, 1, len(g_samp) - 1)
    left, right = g_samp[idx - 1], g_samp[idx]
    pick = np.where(np.abs(samp_arr - left) <= np.abs(samp_arr - right), idx - 1, idx)
    out = g_st[pick]
    # Reject anchors with no GT beat within 30 s -> unusable
    too_far = np.abs(samp_arr - g_samp[pick]) > 30 * FS
    out[too_far] = np.nan
    return out


def trace_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    m = np.isfinite(pred) & np.isfinite(gt)
    p, g = pred[m], gt[m]
    d = {"n": int(m.sum())}
    if d["n"] < 50:
        return d
    # --- H1 shrinkage -------------------------------------------------
    d["std_pred"], d["std_gt"] = float(p.std()), float(g.std())
    d["std_ratio"] = d["std_pred"] / d["std_gt"] if d["std_gt"] > 0 else np.nan
    if g.std() > 0:
        d["slope"] = float(np.polyfit(g, p, 1)[0])          # <1 => shrinkage
        d["corr"] = float(np.corrcoef(g, p)[0, 1])
    for lo, name in ((0.1, "exc10"), (0.2, "exc20")):
        sel = np.abs(g) >= lo
        d[f"{name}_n"] = int(sel.sum())
        if sel.sum() >= 10:
            # amplitude retained inside the excursion band
            d[f"{name}_ratio"] = float(np.abs(p[sel]).mean() / np.abs(g[sel]).mean())
            # of GT excursions, how many does the model also place past threshold?
            d[f"{name}_recall"] = float((np.abs(p[sel]) >= lo).mean())
    # --- H2 smoothness / threshold dynamics ---------------------------
    dp = np.diff(p)
    d["roughness_mean_abs_diff"] = float(np.abs(dp).mean())
    d["lag1_autocorr"] = float(np.corrcoef(p[:-1], p[1:])[0, 1]) if len(p) > 2 else np.nan
    dg = np.diff(g)
    d["gt_roughness"] = float(np.abs(dg).mean())
    d["gt_lag1_autocorr"] = float(np.corrcoef(g[:-1], g[1:])[0, 1]) if len(g) > 2 else np.nan
    for lo, name in ((0.1, "t10"), (0.2, "t20")):
        above_p = np.abs(p) >= lo
        above_g = np.abs(g) >= lo
        d[f"{name}_frac_pred"] = float(above_p.mean())
        d[f"{name}_frac_gt"] = float(above_g.mean())
        # crossings = how often the trace enters/leaves the alarm band
        d[f"{name}_cross_pred"] = int(np.abs(np.diff(above_p.astype(int))).sum())
        d[f"{name}_cross_gt"] = int(np.abs(np.diff(above_g.astype(int))).sum())
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", default=f"{NEUROKIT_ROOT}/st_trace_analysis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ftest = fold_test_patients()
    rows = []

    for vname, pat in VARIANTS.items():
        for f in FOLDS:
            model_name = pat.format(f=f)
            ckpt = f"{CKPT_ROOT}/{model_name}/{model_name}_best.pt"
            if not Path(ckpt).exists():
                print(f"SKIP {vname} fold{f}: missing {ckpt}")
                continue
            print(f"\n=== {vname} fold{f} -> test {ftest[f]}")
            model, revin, dev = load_patchecg_model(
                ckpt_path=ckpt, device=args.device, lead_mode="two_lead")
            for record in ftest[f]:
                try:
                    rec = wfdb.rdrecord(f"{LTST_RECORDS}/{record}/{record}")
                except Exception as exc:
                    print(f"  {record}: read error {exc}"); continue
                sig = rec.p_signal[:, [STTAG_TO_LEAD_IDX["ST0"],
                                       STTAG_TO_LEAD_IDX["ST1"]]].astype(np.float32)
                samp_arr, st_by_lead = predict_st_per_segment(
                    signal_2lead=sig, fs=FS, model=model, revin=revin, device=dev,
                    lead_order=("II", "V2"), batch_size=args.batch_size,
                    lead_mode="two_lead")
                if len(samp_arr) == 0:
                    continue
                for lead_tag in ("ST0", "ST1"):
                    paper = STTAG_TO_PAPER_LEAD[lead_tag]
                    if (record, paper) in MANUAL_OMIT or paper not in st_by_lead:
                        continue
                    pred = np.asarray(st_by_lead[paper], dtype=float)
                    gt = gt_st_for_segments(record, lead_tag, np.asarray(samp_arr, dtype=float))
                    s = trace_stats(pred, gt)
                    s.update(variant=vname, fold=f, record=record, lead=paper)
                    rows.append(s)
                    if s.get("n", 0) >= 50:
                        print(f"  {record} {paper}: n={s['n']} slope={s.get('slope',float('nan')):.3f} "
                              f"stdratio={s.get('std_ratio',float('nan')):.3f} "
                              f"exc10_ratio={s.get('exc10_ratio',float('nan')):.3f} "
                              f"lag1={s.get('lag1_autocorr',float('nan')):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "st_trace_per_record.csv", index=False)

    num = df.select_dtypes(include=[np.number]).columns
    agg = df.groupby("variant")[list(num)].mean(numeric_only=True)
    agg.to_csv(out_dir / "st_trace_summary.csv")
    print(f"\nWrote {out_dir}/st_trace_per_record.csv and st_trace_summary.csv")
    cols = ["slope", "std_ratio", "corr", "exc10_ratio", "exc10_recall", "exc20_ratio",
            "lag1_autocorr", "roughness_mean_abs_diff", "gt_lag1_autocorr", "gt_roughness",
            "t10_frac_pred", "t10_frac_gt", "t10_cross_pred", "t10_cross_gt"]
    print(agg[[c for c in cols if c in agg.columns]].to_string())


if __name__ == "__main__":
    main()
