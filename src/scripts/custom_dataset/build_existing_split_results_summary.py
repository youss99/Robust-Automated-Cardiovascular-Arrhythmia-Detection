#!/usr/bin/env python3
"""
Build a single human-readable results.txt summarizing the EXISTING-SPLIT
experiments (CV runs are intentionally excluded).

Sections:
  1. Regression metrics — per trial
  2. Regression metrics — mean ± std across trials per (lead_variant, mode)
  3. Per-output-lead regression metrics (mean ± std on test)
  4. Best trial per variant (lowest valid_loss)
  5. Alarm-eval — best trial per variant under {clean, noisy} (+ noisy_sqi when present)
  6. Two-lead noisy + SQI sweep — best threshold per mode by F1

Reads from:
  results/fine-tune/saved_models/patient-segment-regression/ltstdb_median_3trial_summary/  (aggregator output)
  Neurokit2_Pipeline/alarm_eval_results/                                                    (single-threshold alarm-eval)
  Neurokit2_Pipeline/alarm_eval_results_sweep/                                              (noisy_sqi sweep)

Writes to:
  results/fine-tune/saved_models/patient-segment-regression/ltstdb_median_3trial_summary/existing_split_results.txt

Run AFTER the 5-trial aggregator + alarm-eval aggregator + sweep have all run.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PATCHECG_RESULTS = Path(
    "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection/results/fine-tune/saved_models/patient-segment-regression"
)
AGG_DIR = PATCHECG_RESULTS / "ltstdb_median_3trial_summary"
ALARM_ROOT = Path(
    "/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline/alarm_eval_results"
)
SWEEP_ROOT = Path(
    "/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline/alarm_eval_results_sweep"
)
OUT_TXT = AGG_DIR / "existing_split_results.txt"

VARIANT_ORDER = [
    ("two_lead", "supervised"),
    ("two_lead", "pretrained"),
    ("lead_II",  "supervised"),
    ("lead_II",  "pretrained"),
    ("lead_V2",  "supervised"),
    ("lead_V2",  "pretrained"),
]

DATASET_META_CSV = Path(
    "/scratch/youss99/ltstdb_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress/modified_dataset/modified_dataset.csv"
)
SPLIT_LABEL = {1: "TRAIN", 2: "TEST", 3: "VALIDATION"}
ABN_THRESHOLD_MV = 0.1


def df_to_text(df: pd.DataFrame, float_fmt="%.4f") -> str:
    if df is None or df.empty:
        return "  (no data)\n"
    return df.to_string(index=False, float_format=lambda v: float_fmt % v) + "\n"


def fmt_section(title: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}\n"


def section_0_split_stats(out_lines: list[str]) -> None:
    import numpy as np
    out_lines.append(fmt_section("SECTION 0 — Dataset split stats (existing 12/2/2 split)"))
    if not DATASET_META_CSV.exists():
        out_lines.append(f"  (missing: {DATASET_META_CSV})\n")
        return
    md = pd.read_csv(DATASET_META_CSV)
    ii = md["target_lead0_mV"].to_numpy(float)
    v2 = md["target_lead1_mV"].to_numpy(float)
    global_ii = float((np.abs(ii) >= ABN_THRESHOLD_MV).mean())
    global_v2 = float((np.abs(v2) >= ABN_THRESHOLD_MV).mean())
    out_lines.append(f"Abnormal segment threshold: |ST| >= {ABN_THRESHOLD_MV} mV\n")
    out_lines.append(f"Global abnormal fractions: II = {global_ii:.4f}    V2 = {global_v2:.4f}\n\n")

    rows = []
    for sp, sub in md.groupby("strat_fold_annotated"):
        ii = sub["target_lead0_mV"].to_numpy(float)
        v2 = sub["target_lead1_mV"].to_numpy(float)
        ii_f = float((np.abs(ii) >= ABN_THRESHOLD_MV).mean())
        v2_f = float((np.abs(v2) >= ABN_THRESHOLD_MV).mean())
        rows.append({
            "split": SPLIT_LABEL.get(int(sp), str(sp)),
            "n_patients": sub["patient_id"].nunique(),
            "n_seg": len(sub),
            "II_abn_frac":   round(ii_f, 4),
            "II_dev_global": round(ii_f - global_ii, 4),
            "V2_abn_frac":   round(v2_f, 4),
            "V2_dev_global": round(v2_f - global_v2, 4),
            "patients": ", ".join(sorted(sub["patient_id"].unique())),
        })
    df = pd.DataFrame(rows).sort_values("split")
    out_lines.append(df_to_text(df, "%.4f"))
    out_lines.append("\nMANUAL_OMIT (alarm-pipeline-omitted record+lead pairs): "
                     "(s20081,V2), (s20091,II), (s20121,II), (s20121,V2), (s20221,V2), (s20241,V2)\n")
    out_lines.append("These 5 records are forced into TRAIN by `--force_omit_to_train` and never reach val/test.\n")


def section_1_per_trial(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 1 — Regression metrics, per trial"))
    p = AGG_DIR / "all_trial_metrics.csv"
    if not p.exists():
        out_lines.append(f"  (missing: {p})\n")
        return
    df = pd.read_csv(p)
    keep = [c for c in ["lead_tag", "mode", "trial", "best_epoch", "best_valid_loss",
                        "test_MSE", "test_MAE", "test_RMSE", "test_R2"] if c in df.columns]
    df = df[keep].sort_values(["lead_tag", "mode", "trial"])
    out_lines.append(df_to_text(df, "%.4f"))


def section_2_mean_std(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 2 — Regression metrics, mean ± std across trials per (lead_variant, mode)"))
    p = AGG_DIR / "aggregate_metric_summary.csv"
    if not p.exists():
        out_lines.append(f"  (missing: {p})\n")
        return
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    for c in ("lead_tag", "mode"):
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip()

    pairs = [("test_MAE", "test_MAE_mean", "test_MAE_std"),
             ("test_RMSE", "test_RMSE_mean", "test_RMSE_std"),
             ("test_R2",   "test_R2_mean",   "test_R2_std"),
             ("best_valid_loss", "best_valid_loss_mean", "best_valid_loss_std"),
             ("best_valid_R2",  "best_valid_R2_mean",  "best_valid_R2_std")]
    rows = []
    for lead_tag, mode in VARIANT_ORDER:
        sub = df[(df["lead_tag"] == lead_tag) & (df["mode"] == mode)]
        if sub.empty: continue
        r = sub.iloc[0]
        row = {"lead_variant": lead_tag, "mode": mode}
        for label, mcol, scol in pairs:
            if mcol in r and scol in r and pd.notna(r[mcol]) and pd.notna(r[scol]):
                row[label] = f"{float(r[mcol]):.4f} ± {float(r[scol]):.4f}"
        rows.append(row)
    out_lines.append(df_to_text(pd.DataFrame(rows), "%.4f"))


def section_3_per_lead(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 3 — Per-output-lead regression metrics (test, mean ± std)"))
    p = AGG_DIR / "per_lead_metric_summary.csv"
    if not p.exists():
        out_lines.append(f"  (missing: {p})\n")
        return
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    for c in ("lead_tag", "mode", "split", "label"):
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip()
    df = df[df["split"] == "test"].copy()
    rows = []
    for lead_tag, mode in VARIANT_ORDER:
        sub = df[(df["lead_tag"] == lead_tag) & (df["mode"] == mode)]
        for _, r in sub.iterrows():
            rows.append({
                "lead_variant": lead_tag,
                "mode": mode,
                "output_lead": r["label"],
                "MAE":  f"{float(r['MAE_mean']):.4f} ± {float(r['MAE_std']):.4f}",
                "RMSE": f"{float(r['RMSE_mean']):.4f} ± {float(r['RMSE_std']):.4f}",
                "R2":   f"{float(r['R2_mean']):.4f} ± {float(r['R2_std']):.4f}",
            })
    out_lines.append(df_to_text(pd.DataFrame(rows), "%.4f"))


def section_4_best_trials(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 4 — Best trial per variant (lowest valid_loss)"))
    p = AGG_DIR / "best_trial_per_variant.csv"
    if not p.exists():
        out_lines.append(f"  (missing: {p})\n")
        return
    df = pd.read_csv(p)
    keep = [c for c in ["lead_tag", "mode", "trial", "best_valid_loss", "best_epoch", "checkpoint_exists"]
            if c in df.columns]
    out_lines.append(df_to_text(df[keep], "%.6f"))


def section_5_alarm_eval(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 5 — Alarm-eval (CLEAN condition only) — best trial per variant"))
    p = ALARM_ROOT / "_summary" / "per_run_alarm_metrics.csv"
    if not p.exists():
        out_lines.append(f"  (missing: {p})\n")
        return
    df = pd.read_csv(p)
    df = df[df["condition"] == "clean"].copy()

    # Restrict to the *best* trial per (lead_tag, mode) from the regression aggregator,
    # so the same checkpoint that the regression best-trial table calls "best" is the
    # one shown here.
    best_csv = AGG_DIR / "best_trial_per_variant.csv"
    if best_csv.exists():
        best = pd.read_csv(best_csv)
        # trial column type-coerce
        best["trial_int"] = best["trial"].astype(int)
        df["trial_int"] = df["trial"].astype(int)
        keep_pairs = set()
        for _, r in best.iterrows():
            keep_pairs.add((r["lead_tag"], r["mode"], int(r["trial"])))
        df = df[df.apply(lambda r: (r["lead_tag"], r["mode"], int(r["trial"])) in keep_pairs, axis=1)]
        df = df.drop(columns=["trial_int"], errors="ignore")

    keep = [c for c in ["lead_tag", "mode", "trial",
                        "tp", "fp", "fn", "precision", "recall", "f1",
                        "total_pred_alarms", "total_gs_alarms",
                        "median_onset_error_s", "median_jaccard"] if c in df.columns]
    df = df[keep].sort_values(["lead_tag", "mode"])
    out_lines.append("Clean ECG, SQI disabled, no noise. GS alarms = 101. One row per (lead_variant, mode), best trial by valid_loss.\n\n")
    out_lines.append(df_to_text(df, "%.3f"))


def section_6_sweep(out_lines: list[str]) -> None:
    out_lines.append(fmt_section("SECTION 6 — Two-lead noisy + SQI sweep (best threshold per mode by F1)"))
    if not SWEEP_ROOT.exists():
        out_lines.append(f"  (sweep root not found: {SWEEP_ROOT})\n")
        return

    summary_rows = []
    for model_dir in sorted(SWEEP_ROOT.iterdir()):
        if not model_dir.is_dir(): continue
        eval_dir = model_dir / "noisy_sqi" / "eval_outputs"
        if not eval_dir.is_dir(): continue
        # The pipeline produces per-threshold *_summary_stats.csv under eval_outputs/
        per_thr = list(eval_dir.glob("*_summary_stats.csv"))
        if not per_thr: continue

        rows = []
        for f in per_thr:
            try:
                r = pd.read_csv(f).iloc[0].to_dict()
            except Exception:
                continue
            label = r.get("label", f.stem.replace("_summary_stats", ""))
            # threshold lives in the label like "..._sqi-20" / "..._sqi5"
            thr = None
            for tok in str(label).split("_"):
                if tok.startswith("sqi") and tok != "sqi":
                    try: thr = int(tok.replace("sqi", ""))
                    except ValueError: pass
            if thr is None:
                continue
            rows.append({
                "model_name": model_dir.name,
                "sqi_threshold_dB": thr,
                "TP": int(r.get("tp", 0)),
                "FP": int(r.get("fp", 0)),
                "FN": int(r.get("fn", 0)),
                "precision": float(r.get("precision", 0)),
                "recall":    float(r.get("recall",    0)),
                "f1":        float(r.get("f1",        0)),
                "pred_n":    int(r.get("total_pred_alarms", 0)),
            })
        if not rows:
            continue
        cur = pd.DataFrame(rows).sort_values("sqi_threshold_dB")
        best = cur.loc[cur["f1"].idxmax()]
        summary_rows.append({
            "model_name": model_dir.name,
            "best_sqi_threshold_dB": int(best["sqi_threshold_dB"]),
            "TP_at_best":  int(best["TP"]),
            "FP_at_best":  int(best["FP"]),
            "FN_at_best":  int(best["FN"]),
            "precision":   round(float(best["precision"]), 3),
            "recall":      round(float(best["recall"]),    3),
            "f1":          round(float(best["f1"]),        3),
            "pred_n":      int(best["pred_n"]),
        })

        # Also dump the full curve into a per-model CSV next to the eval_outputs
        curve_path = model_dir / "noisy_sqi" / "sweep_curve.csv"
        cur.to_csv(curve_path, index=False)

    if not summary_rows:
        out_lines.append("  (no sweep results yet — submit the two noisy_sqi sweep jobs and re-run this)\n")
        return

    out_lines.append("Best threshold per model (by F1); full per-threshold curves saved under each model's noisy_sqi/sweep_curve.csv\n\n")
    out_lines.append(df_to_text(pd.DataFrame(summary_rows), "%.3f"))


def main() -> int:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    out_lines.append("=" * 78 + "\n")
    out_lines.append("LTSTDB Median Regression — Existing-split Summary (CV intentionally excluded)\n")
    out_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    out_lines.append("Split: 12 train / 2 val / 2 test, force_omit_to_train applied (5 MANUAL_OMIT records always in train).\n")
    out_lines.append("=" * 78 + "\n")

    section_0_split_stats(out_lines)
    section_1_per_trial(out_lines)
    section_2_mean_std(out_lines)
    section_3_per_lead(out_lines)
    section_4_best_trials(out_lines)
    section_5_alarm_eval(out_lines)
    # noise + noisy_sqi sweep intentionally omitted — to be added later when needed

    OUT_TXT.write_text("".join(out_lines))
    print(f"Wrote: {OUT_TXT}")
    # Also echo to stdout
    print()
    sys.stdout.write("".join(out_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
