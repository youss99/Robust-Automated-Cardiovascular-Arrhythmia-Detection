#!/usr/bin/env python3
"""
Build the two-lead clean vs noisy vs best-noisy_sqi summary table + two plots.

Reads:
  - Single-threshold alarm-eval (clean + noisy):
      Neurokit2_Pipeline/alarm_eval_results/<model>/<cond>/eval_outputs/ml_alarms_vs_gs_summary_stats.csv
  - SQI sweep (noisy + SQI on, -20..25 dB step 1):
      Neurokit2_Pipeline/alarm_eval_results_sweep/<model>/noisy_sqi/eval_outputs/<label>_summary_stats.csv
      (one file per integer threshold; threshold embedded in `label`, e.g. "..._sqi-12")

Writes:
  - <results_summary_dir>/two_lead_sqi_sweep_summary.txt   (table)
  - <results_summary_dir>/two_lead_sqi_sweep_curves.csv    (per-threshold curves)
  - <results_summary_dir>/plot_two_lead_f1_vs_sqi.png
  - <results_summary_dir>/plot_two_lead_pred_alarms_vs_sqi.png

Two-lead models considered (best per the 5-trial aggregator):
  supervised  -> trial05
  pretrained  -> trial02
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALARM_ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline/alarm_eval_results")
SWEEP_ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline/alarm_eval_results_sweep")
OUT_DIR = Path(
    "/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection/results/fine-tune/saved_models/patient-segment-regression/ltstdb_median_3trial_summary"
)

MODELS = {
    "supervised":  "ltstdb_median_full16_12_2_2_omit_train_two_lead_regression_supervised_trial05",
    "pretrained (fine-tune)": "ltstdb_median_full16_12_2_2_omit_train_two_lead_regression_pretrained_chapman_ii_v2_trial02",
}


def _count_ignored(detailed_csv: Path) -> int:
    if not detailed_csv.exists():
        return 0
    try:
        df = pd.read_csv(detailed_csv, usecols=["match_type"])
    except Exception:
        return 0
    return int((df["match_type"] == "ignored").sum())


def read_single(model_name: str, cond: str) -> dict | None:
    eval_dir = ALARM_ROOT / model_name / cond / "eval_outputs"
    p = eval_dir / "ml_alarms_vs_gs_summary_stats.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p).iloc[0].to_dict()
    r["ignored"] = _count_ignored(eval_dir / "ml_alarms_vs_gs_detailed_matches.csv")
    return r


def read_sweep_curve(model_name: str) -> pd.DataFrame:
    """Return DataFrame: rows per integer threshold."""
    eval_dir = SWEEP_ROOT / model_name / "noisy_sqi" / "eval_outputs"
    if not eval_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for f in eval_dir.glob("*_summary_stats.csv"):
        try:
            r = pd.read_csv(f).iloc[0].to_dict()
        except Exception:
            continue
        label = str(r.get("label", f.stem.replace("_summary_stats", "")))
        thr = None
        for tok in label.split("_"):
            if tok.startswith("sqi") and tok != "sqi":
                try:
                    thr = int(tok.replace("sqi", ""))
                except ValueError:
                    pass
        if thr is None:
            continue
        # Detailed matches sits next to the summary_stats; recompute Ignored.
        ignored = _count_ignored(f.parent / f.name.replace("_summary_stats.csv",
                                                            "_detailed_matches.csv"))
        rows.append({
            "sqi_threshold_dB": thr,
            "TP": int(r.get("tp", 0)),
            "FP": int(r.get("fp", 0)),
            "FN": int(r.get("fn", 0)),
            "Ignored": ignored,
            "precision": float(r.get("precision", 0)),
            "recall":    float(r.get("recall",    0)),
            "f1":        float(r.get("f1",        0)),
            "pred_n":    int(r.get("total_pred_alarms", 0)),
            "median_onset_err_s":    float(r.get("median_onset_error_s",    float("nan"))),
            "median_end_err_s":      float(r.get("median_end_error_s",      float("nan"))),
            "median_jaccard":        float(r.get("median_jaccard",          float("nan"))),
            "median_duration_diff_s":float(r.get("median_duration_diff_s",  float("nan"))),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("sqi_threshold_dB").reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    curves = {}
    for mode, model in MODELS.items():
        clean = read_single(model, "clean")
        noisy = read_single(model, "noisy")
        curve = read_sweep_curve(model)

        # Best SQI threshold by F1
        best_thr_row = None
        if not curve.empty:
            best_thr_row = curve.loc[curve["f1"].idxmax()]

        for label, r in [("clean", clean), ("noisy (SQI off)", noisy)]:
            if r is None:
                rows.append({"mode": mode, "condition": label, "note": "missing"})
                continue
            rows.append({
                "mode": mode,
                "condition": label,
                "sqi_threshold_dB": "",
                "TP": int(r.get("tp", 0)),
                "FP": int(r.get("fp", 0)),
                "FN": int(r.get("fn", 0)),
                "Ignored":   int(r.get("ignored", 0)),
                "precision": round(float(r.get("precision", 0)), 3),
                "recall":    round(float(r.get("recall",    0)), 3),
                "F1":        round(float(r.get("f1",        0)), 3),
                "pred_n":    int(r.get("total_pred_alarms", 0)),
                "gs_n":      int(r.get("total_gs_alarms", 101)),
                "med_onset_err_s":  round(float(r.get("median_onset_error_s",    0.0)), 2),
                "med_end_err_s":    round(float(r.get("median_end_error_s",      0.0)), 2),
                "med_jaccard":      round(float(r.get("median_jaccard",          0.0)), 3),
                "med_duration_diff_s": round(float(r.get("median_duration_diff_s", 0.0)), 2),
            })

        if best_thr_row is not None:
            rows.append({
                "mode": mode,
                "condition": "noisy + SQI (best)",
                "sqi_threshold_dB": int(best_thr_row["sqi_threshold_dB"]),
                "TP": int(best_thr_row["TP"]),
                "FP": int(best_thr_row["FP"]),
                "FN": int(best_thr_row["FN"]),
                "Ignored":   int(best_thr_row.get("Ignored", 0)),
                "precision": round(float(best_thr_row["precision"]), 3),
                "recall":    round(float(best_thr_row["recall"]),    3),
                "F1":        round(float(best_thr_row["f1"]),        3),
                "pred_n":    int(best_thr_row["pred_n"]),
                "gs_n":      101,
                "med_onset_err_s":  round(float(best_thr_row.get("median_onset_err_s",    0.0)), 2),
                "med_end_err_s":    round(float(best_thr_row.get("median_end_err_s",      0.0)), 2),
                "med_jaccard":      round(float(best_thr_row.get("median_jaccard",        0.0)), 3),
                "med_duration_diff_s": round(float(best_thr_row.get("median_duration_diff_s", 0.0)), 2),
            })
            curves[mode] = curve
        else:
            rows.append({"mode": mode, "condition": "noisy + SQI (best)", "note": "sweep not done"})

    table = pd.DataFrame(rows)

    # Write table to text + save curves CSV
    out_txt = OUT_DIR / "two_lead_sqi_sweep_summary.txt"
    out_csv = OUT_DIR / "two_lead_sqi_sweep_curves.csv"

    with open(out_txt, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("Two-lead clean vs noisy vs noisy+SQI (best threshold)\n")
        f.write("Best by F1 over SQI sweep [-20, 25] dB (1 dB step). GS = 101 alarms.\n")
        f.write("=" * 78 + "\n\n")
        f.write(table.to_string(index=False) + "\n")

    # Save per-threshold curves (one row per mode/threshold)
    long_rows = []
    for mode, curve in curves.items():
        for _, r in curve.iterrows():
            long_rows.append({"mode": mode, **r.to_dict()})
    if long_rows:
        pd.DataFrame(long_rows).to_csv(out_csv, index=False)

    # ---- Plots ----
    if curves:
        # PLOT 1: F1 vs SQI threshold, with horizontal reference lines
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, curve in curves.items():
            ax.plot(curve["sqi_threshold_dB"], curve["f1"], marker="o", markersize=3, label=f"{mode} (noisy + SQI)")
            # reference: clean + noisy F1 for the same mode
            model = MODELS[mode]
            clean = read_single(model, "clean") or {}
            noisy = read_single(model, "noisy") or {}
            if "f1" in clean:
                ax.axhline(float(clean["f1"]), linestyle="--", linewidth=1, alpha=0.6,
                           color=ax.lines[-1].get_color(), label=f"{mode} clean F1 = {float(clean['f1']):.3f}")
            if "f1" in noisy:
                ax.axhline(float(noisy["f1"]), linestyle=":",  linewidth=1, alpha=0.6,
                           color=ax.lines[-3].get_color() if len(ax.lines) >= 3 else None,
                           label=f"{mode} noisy F1 = {float(noisy['f1']):.3f}")
        ax.set_xlabel("SQI threshold (dB)")
        ax.set_ylabel("Alarm-level F1")
        ax.set_title("Two-lead: F1 vs SQI threshold (noisy ECG)\nDashed = clean baseline, dotted = noisy w/o SQI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "plot_two_lead_f1_vs_sqi.png", dpi=140)
        plt.close(fig)

        # PLOT 2: Total predicted alarms vs SQI threshold, with horizontal reference lines
        fig, ax = plt.subplots(figsize=(8, 5))
        for mode, curve in curves.items():
            line, = ax.plot(curve["sqi_threshold_dB"], curve["pred_n"], marker="o", markersize=3,
                            label=f"{mode} (noisy + SQI) — pred_n")
            color = line.get_color()
            model = MODELS[mode]
            clean = read_single(model, "clean") or {}
            noisy = read_single(model, "noisy") or {}
            if "total_pred_alarms" in clean:
                ax.axhline(float(clean["total_pred_alarms"]), linestyle="--", linewidth=1,
                           alpha=0.6, color=color,
                           label=f"{mode} clean pred_n = {int(clean['total_pred_alarms'])}")
            if "total_pred_alarms" in noisy:
                ax.axhline(float(noisy["total_pred_alarms"]), linestyle=":", linewidth=1,
                           alpha=0.6, color=color,
                           label=f"{mode} noisy pred_n = {int(noisy['total_pred_alarms'])}")
        # Reference: GS alarm count
        ax.axhline(101, color="black", linestyle="-", linewidth=1, alpha=0.5, label="GS = 101")
        ax.set_xlabel("SQI threshold (dB)")
        ax.set_ylabel("Total predicted alarms")
        ax.set_title("Two-lead: total predicted alarms vs SQI threshold (noisy ECG)\nDashed = clean baseline, dotted = noisy w/o SQI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "plot_two_lead_pred_alarms_vs_sqi.png", dpi=140)
        plt.close(fig)

        # PLOT 3 & 4: One plot per model — TP, FP, FN vs SQI threshold,
        # with horizontal references for clean and noisy (SQI off) baselines.
        for mode, curve in curves.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(curve["sqi_threshold_dB"], curve["TP"], marker="o", markersize=3,
                    color="tab:green", label="TP")
            ax.plot(curve["sqi_threshold_dB"], curve["FP"], marker="o", markersize=3,
                    color="tab:red",   label="FP")
            ax.plot(curve["sqi_threshold_dB"], curve["FN"], marker="o", markersize=3,
                    color="tab:blue",  label="FN")
            if "Ignored" in curve.columns:
                ax.plot(curve["sqi_threshold_dB"], curve["Ignored"], marker="o", markersize=3,
                        color="tab:orange", label="Ignored (fragmentation)")

            # Best-F1 threshold marker
            best_idx = curve["f1"].idxmax()
            best_thr = int(curve.loc[best_idx, "sqi_threshold_dB"])
            ax.axvline(best_thr, color="black", linestyle="--", linewidth=1, alpha=0.6,
                       label=f"best F1 @ SQI = {best_thr} dB")

            # Horizontal references for clean + noisy
            model = MODELS[mode]
            clean = read_single(model, "clean") or {}
            noisy = read_single(model, "noisy") or {}
            for ref_label, ref, ls in [("clean", clean, "--"), ("noisy", noisy, ":")]:
                if not ref: continue
                ax.axhline(int(ref.get("tp", 0)), color="tab:green", linestyle=ls, alpha=0.4,
                           label=f"TP {ref_label} = {int(ref.get('tp', 0))}")
                ax.axhline(int(ref.get("fp", 0)), color="tab:red",   linestyle=ls, alpha=0.4,
                           label=f"FP {ref_label} = {int(ref.get('fp', 0))}")
                ax.axhline(int(ref.get("fn", 0)), color="tab:blue",  linestyle=ls, alpha=0.4,
                           label=f"FN {ref_label} = {int(ref.get('fn', 0))}")
                ax.axhline(int(ref.get("ignored", 0)), color="tab:orange", linestyle=ls, alpha=0.4,
                           label=f"Ignored {ref_label} = {int(ref.get('ignored', 0))}")

            ax.set_xlabel("SQI threshold (dB)")
            ax.set_ylabel("Count")
            ax.set_title(f"Two-lead {mode}: TP / FP / FN vs SQI threshold (noisy ECG, GS=101)\n"
                         f"Dashed refs = clean baseline, dotted refs = noisy (SQI off)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=7, ncol=2)
            fig.tight_layout()
            # Make a filesystem-safe filename
            safe = mode.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            fig.savefig(OUT_DIR / f"plot_two_lead_{safe}_tp_fp_fn_vs_sqi.png", dpi=140)
            plt.close(fig)

    # Echo
    print(table.to_string(index=False))
    print(f"\nWrote table:  {out_txt}")
    if long_rows:
        print(f"Wrote curves: {out_csv}")
        print(f"Wrote plots:  {OUT_DIR / 'plot_two_lead_f1_vs_sqi.png'}")
        print(f"              {OUT_DIR / 'plot_two_lead_pred_alarms_vs_sqi.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
