"""
Compare CV5 Option B two-lead model performance on CLEAN vs NOISY signal.

Focus: does the pretrained two-lead model (which has better regression R^2 and
lower fold-to-fold variance) also produce more alarm-eval-robust predictions
when the input ECG is contaminated with NSTDB motion-artifact noise?

Reads:
  - clean:  alarm_eval_results_cv5_optB/two_lead/<mode>/clean/
            (ml_alarms_pooled.csv + eval_outputs/*summary_stats.csv)
  - noisy:  alarm_eval_results_cv5_optB/two_lead/<mode>/noisy/

Outputs (under alarm_eval_results_cv5_optB/_summary/):
  cv5_optB_two_lead_clean_vs_noisy.csv
  cv5_optB_two_lead_clean_vs_noisy.txt   (human-readable side-by-side + deltas)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

NEUROKIT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline")
ROOT = NEUROKIT / "alarm_eval_results_cv5_optB" / "two_lead"
OUT_DIR = NEUROKIT / "alarm_eval_results_cv5_optB" / "_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (mode_label, mode_dir_name)
MODES = [
    ("supervised", "supervised"),
    ("pretrained", "pretrained_chapman_ii_v2"),
]
CONDS = ["clean", "noisy"]


def read_stats(mode_dir: str, cond: str) -> dict | None:
    d = ROOT / mode_dir / cond
    pooled = d / "ml_alarms_pooled.csv"
    eval_dir = d / "eval_outputs"
    if not pooled.exists():
        return None
    cands = list(eval_dir.glob("*summary_stats.csv"))
    if not cands:
        return None
    stats = pd.read_csv(cands[0]).iloc[0]
    n_alarms = len(pd.read_csv(pooled))
    return {
        "n_pooled_alarms": n_alarms,
        "n_pred_alarms": int(stats["total_pred_alarms"]),
        "n_gs_alarms": int(stats["total_gs_alarms"]),
        "TP": int(stats["tp"]),
        "FP": int(stats["fp"]),
        "FN": int(stats["fn"]),
        "f1": float(stats["f1"]),
        "precision": float(stats["precision"]),
        "recall": float(stats["recall"]),
        "jaccard_med": float(stats["median_jaccard"]),
        "overlap_frac_pred_med": float(stats["median_overlap_frac_pred"]),
        "overlap_frac_gs_med": float(stats["median_overlap_frac_gs"]),
        "onset_err_med_s": float(stats["median_onset_error_s"]),
        "end_err_med_s": float(stats["median_end_error_s"]),
        "duration_diff_med_s": float(stats["median_duration_diff_s"]),
        "pred_total_dur_h": float(stats["total_pred_duration_s"]) / 3600.0,
        "gs_total_dur_h": float(stats["total_gs_duration_s"]) / 3600.0,
    }


def main():
    rows = []
    missing = []
    for mode_label, mode_dir in MODES:
        per_cond = {}
        for cond in CONDS:
            s = read_stats(mode_dir, cond)
            if s is None:
                missing.append(f"two_lead/{mode_dir}/{cond}")
            per_cond[cond] = s
        if per_cond["clean"] is None or per_cond["noisy"] is None:
            continue
        # Long-form for CSV
        for cond in CONDS:
            row = {"mode": mode_label, "condition": cond, **per_cond[cond]}
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data: jobs may still be pending. Missing:")
        for m in missing:
            print(f"  - {m}")
        return

    csv_path = OUT_DIR / "cv5_optB_two_lead_clean_vs_noisy.csv"
    df.to_csv(csv_path, index=False)

    # Build the human-readable text summary
    lines = []
    W = 92
    lines.append("=" * W)
    lines.append("CV5 Option B two-lead: CLEAN vs NOISY alarm-eval")
    lines.append("=" * W)
    lines.append("")
    lines.append("Pool: 15 records (s20121 always trains). Same GS file in both conditions.")
    lines.append("Noise: NSTDB 'em' motion artifact applied to ECG before ML inference.")
    lines.append("SQI gating disabled in both conditions (apples-to-apples).")
    lines.append("")

    # Side-by-side per metric per mode
    for mode_label, _ in MODES:
        sub = df[df["mode"] == mode_label]
        if sub.empty: continue
        clean = sub[sub.condition == "clean"].iloc[0]
        noisy = sub[sub.condition == "noisy"].iloc[0]
        lines.append("-" * W)
        lines.append(f"Mode: {mode_label}")
        lines.append("-" * W)
        h = f"{'metric':<28} {'clean':>14} {'noisy':>14} {'delta':>14} {'%change':>10}"
        lines.append(h)

        def row(metric, lo_better=False, digits=3, suffix=""):
            c = clean[metric]; n = noisy[metric]
            d = n - c
            pct = (d / abs(c) * 100.0) if c != 0 else float('nan')
            arrow = ""
            if not pd.isna(pct):
                if abs(pct) >= 1.0:
                    # Improvement = wanted direction
                    if (d > 0 and not lo_better) or (d < 0 and lo_better):
                        arrow = "  better"
                    else:
                        arrow = "  worse"
            lines.append(
                f"{metric:<28} {c:>14.{digits}f}{suffix:<1} {n:>14.{digits}f}{suffix:<1} "
                f"{d:>+14.{digits}f}{suffix:<1} {pct:>+9.1f}%{arrow}"
            )

        row("n_pred_alarms", lo_better=True, digits=0)
        row("TP",            lo_better=False, digits=0)
        row("FP",            lo_better=True,  digits=0)
        row("FN",            lo_better=True,  digits=0)
        row("f1",            lo_better=False, digits=3)
        row("precision",     lo_better=False, digits=3)
        row("recall",        lo_better=False, digits=3)
        row("jaccard_med",   lo_better=False, digits=3)
        row("onset_err_med_s", lo_better=False, digits=1)
        row("end_err_med_s",   lo_better=False, digits=1)
        row("pred_total_dur_h", lo_better=True, digits=1)
        lines.append("")

    # Head-to-head: which model is more robust to noise?
    lines.append("=" * W)
    lines.append("Robustness head-to-head: which model degrades less from clean to noisy?")
    lines.append("=" * W)
    lines.append("")
    h = f"{'metric':<28} {'sup clean':>10} {'sup noisy':>10} {'pre clean':>10} {'pre noisy':>10} {'winner':>20}"
    lines.append(h)
    sup = {c: df[(df["mode"] == "supervised") & (df.condition == c)] for c in CONDS}
    pre = {c: df[(df["mode"] == "pretrained") & (df.condition == c)] for c in CONDS}

    def vs_row(metric, lo_better=False, digits=3, fmt="{:>10.3f}"):
        sc = sup["clean"].iloc[0][metric]; sn = sup["noisy"].iloc[0][metric]
        pc = pre["clean"].iloc[0][metric]; pn = pre["noisy"].iloc[0][metric]
        sup_drop = (sn - sc)
        pre_drop = (pn - pc)
        # Which model loses less? "loses less" = smaller magnitude of drop in the
        # bad direction.
        if lo_better:
            sup_bad = max(0, sup_drop); pre_bad = max(0, pre_drop)
        else:
            sup_bad = max(0, -sup_drop); pre_bad = max(0, -pre_drop)
        if sup_bad == pre_bad:
            winner = "tie"
        elif sup_bad < pre_bad:
            winner = f"supervised (-{pre_bad - sup_bad:.3f})"
        else:
            winner = f"pretrained (-{sup_bad - pre_bad:.3f})"
        if digits == 0:
            cellfmt = "{:>10.0f}"
        elif digits == 1:
            cellfmt = "{:>10.1f}"
        else:
            cellfmt = "{:>10.3f}"
        lines.append(
            f"{metric:<28} {cellfmt.format(sc)} {cellfmt.format(sn)} "
            f"{cellfmt.format(pc)} {cellfmt.format(pn)} {winner:>20}"
        )

    vs_row("f1",            lo_better=False, digits=3)
    vs_row("precision",     lo_better=False, digits=3)
    vs_row("recall",        lo_better=False, digits=3)
    vs_row("TP",            lo_better=False, digits=0)
    vs_row("FP",            lo_better=True,  digits=0)
    vs_row("FN",            lo_better=True,  digits=0)
    vs_row("jaccard_med",   lo_better=False, digits=3)
    vs_row("n_pred_alarms", lo_better=True,  digits=0)
    lines.append("")
    lines.append("Notes:")
    lines.append("  - delta = noisy - clean; '%change' relative to clean.")
    lines.append("  - 'winner' column: model whose absolute degradation (in the BAD direction)")
    lines.append("    is smaller.  Difference shown in parens is the magnitude of the gap.")
    lines.append("")

    txt = "\n".join(lines)
    txt_path = OUT_DIR / "cv5_optB_two_lead_clean_vs_noisy.txt"
    txt_path.write_text(txt)
    print(txt)
    print(f"\nWrote:\n  {csv_path}\n  {txt_path}")
    if missing:
        print("\nNote: some inputs were missing (jobs probably still running):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
