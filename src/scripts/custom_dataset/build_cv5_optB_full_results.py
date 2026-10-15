"""
Build a single comprehensive results text file for the CV5 Option B experiment.

Pulls together:
  - Regression metrics (overall + per-class) for all sub-designs x modes (CV5)
  - Existing-split baseline (12/2/2 single split, 5 trials per variant)
  - Alarm-eval results (clean signal) for the same CV5 runs
  - Existing-split alarm-eval baseline (best trials per variant)
  - Cross-cut R^2 + F1 table

Output: cv5_optB_summary/cv5_optB_full_results.txt
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PATCHECG = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
NEUROKIT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline")
REG_DIR = PATCHECG / "results/fine-tune/saved_models/patient-segment-regression"

CV5_SUMMARY_DIR = REG_DIR / "cv5_optB_summary"
ALARM_SUMMARY_DIR = NEUROKIT / "alarm_eval_results_cv5_optB" / "_summary"
EX_ALARM_CSV = NEUROKIT / "alarm_eval_results" / "_summary" / "per_run_alarm_metrics.csv"

OUT = CV5_SUMMARY_DIR / "cv5_optB_full_results.txt"

W = 102  # column width


def hr(ch="=", w=W) -> str:
    return ch * w


def fmt_pm(m: float, s: float, d: int = 3) -> str:
    return f"{m:.{d}f} +/- {s:.{d}f}" if pd.notna(m) else "n/a"


def section(title: str) -> str:
    return f"\n{hr('=')}\n{title}\n{hr('=')}\n"


def subsection(title: str) -> str:
    return f"\n{title}\n{hr('-')}\n"


def render_cv5_regression_overall() -> str:
    pf = pd.read_csv(CV5_SUMMARY_DIR / "cv5_optB_per_fold_overall.csv")
    summary = pd.read_csv(CV5_SUMMARY_DIR / "cv5_optB_summary_overall.csv")
    out = []
    out.append(subsection("CV5 Option B — Regression (overall, mean +/- std across 5 folds)"))
    h = f"{'sub_design':<12} {'mode':<11} {'MSE':<22} {'MAE':<22} {'RMSE':<22} {'R2':<22}"
    out.append(h)
    for _, r in summary.iterrows():
        out.append(
            f"{r['sub_design']:<12} {r['mode']:<11} "
            f"{fmt_pm(r['test_MSE_mean'],  r['test_MSE_std'], 4):<22} "
            f"{fmt_pm(r['test_MAE_mean'],  r['test_MAE_std'], 4):<22} "
            f"{fmt_pm(r['test_RMSE_mean'], r['test_RMSE_std'], 4):<22} "
            f"{fmt_pm(r['test_R2_mean'],   r['test_R2_std'], 3):<22}"
        )

    out.append(subsection("CV5 Option B — Regression (per fold, two_lead only)"))
    two_pf = pf[pf.sub_design == "two_lead"].copy()
    h = f"{'mode':<11} {'fold':>4}  {'MSE':>8}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}"
    out.append(h)
    for _, r in two_pf.sort_values(["mode", "fold"]).iterrows():
        out.append(
            f"{r['mode']:<11} {r['fold']:>4}  "
            f"{r['test_MSE']:>8.4f}  {r['test_MAE']:>8.4f}  "
            f"{r['test_RMSE']:>8.4f}  {r['test_R2']:>8.3f}"
        )
    return "\n".join(out) + "\n"


def render_cv5_regression_per_class() -> str:
    pc_summary = pd.read_csv(CV5_SUMMARY_DIR / "cv5_optB_summary_per_class.csv")
    out = [subsection("CV5 Option B — Regression (per-lead, mean +/- std across 5 folds)")]
    h = f"{'sub_design':<12} {'mode':<11} {'lead':<16} {'MSE':<22} {'MAE':<22} {'R2':<22}"
    out.append(h)
    for _, r in pc_summary.iterrows():
        out.append(
            f"{r['sub_design']:<12} {r['mode']:<11} {r['label']:<16} "
            f"{fmt_pm(r['MSE_mean'], r['MSE_std'], 4):<22} "
            f"{fmt_pm(r['MAE_mean'], r['MAE_std'], 4):<22} "
            f"{fmt_pm(r['R2_mean'],  r['R2_std'], 3):<22}"
        )
    return "\n".join(out) + "\n"


def render_existing_split_regression() -> str:
    """5-trial existing-split two-lead regression rows."""
    rows_o, rows_pc = [], []
    for mode, tag in [("supervised", None), ("pretrained", "chapman_ii_v2")]:
        for t in range(1, 6):
            name = (f"ltstdb_median_full16_12_2_2_omit_train_two_lead_regression_"
                    f"{'pretrained_'+tag if tag else 'supervised'}_trial{t:02d}")
            d = REG_DIR / name
            ov = d / f"_test_metrics_{name}.csv"
            pc = d / f"_test_metrics_{name}_per_class.csv"
            if ov.exists():
                df = pd.read_csv(ov); df.insert(0, "trial", t); df.insert(0, "mode", mode); rows_o.append(df)
            if pc.exists():
                df = pd.read_csv(pc); df.insert(0, "trial", t); df.insert(0, "mode", mode); rows_pc.append(df)
    ov = pd.concat(rows_o, ignore_index=True) if rows_o else pd.DataFrame()
    pc = pd.concat(rows_pc, ignore_index=True) if rows_pc else pd.DataFrame()

    out = []
    out.append(subsection("Existing-split (12/2/2) — Two-lead regression, per-trial"))
    h = f"{'mode':<11} {'trial':>5}  {'MSE':>8}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}"
    out.append(h)
    for _, r in ov.sort_values(["mode", "trial"]).iterrows():
        out.append(f"{r['mode']:<11} {r['trial']:>5}  "
                   f"{r['test_MSE']:>8.4f}  {r['test_MAE']:>8.4f}  "
                   f"{r['test_RMSE']:>8.4f}  {r['test_R2']:>8.3f}")

    out.append(subsection("Existing-split — Two-lead regression, mean +/- std across 5 trials"))
    out.append(f"{'mode':<11} {'MSE':<22} {'MAE':<22} {'RMSE':<22} {'R2':<22}")
    for mode in ("supervised", "pretrained"):
        sub = ov[ov["mode"] == mode]
        if sub.empty: continue
        out.append(
            f"{mode:<11} "
            f"{fmt_pm(sub.test_MSE.mean(),  sub.test_MSE.std(), 4):<22} "
            f"{fmt_pm(sub.test_MAE.mean(),  sub.test_MAE.std(), 4):<22} "
            f"{fmt_pm(sub.test_RMSE.mean(), sub.test_RMSE.std(), 4):<22} "
            f"{fmt_pm(sub.test_R2.mean(),   sub.test_R2.std(), 3):<22}"
        )

    if not pc.empty:
        out.append(subsection("Existing-split — Two-lead regression, per-lead mean +/- std"))
        out.append(f"{'mode':<11} {'lead':<16} {'MSE':<22} {'MAE':<22} {'R2':<22}")
        for mode in ("supervised", "pretrained"):
            for lead in ("target_lead0_mV", "target_lead1_mV"):
                sub = pc[(pc["mode"] == mode) & (pc["label"] == lead)]
                if sub.empty: continue
                out.append(
                    f"{mode:<11} {lead:<16} "
                    f"{fmt_pm(sub.MSE.mean(), sub.MSE.std(), 4):<22} "
                    f"{fmt_pm(sub.MAE.mean(), sub.MAE.std(), 4):<22} "
                    f"{fmt_pm(sub.R2.mean(),  sub.R2.std(), 3):<22}"
                )
    return "\n".join(out) + "\n", ov, pc


def render_cv5_alarms() -> str:
    df = pd.read_csv(ALARM_SUMMARY_DIR / "cv5_optB_alarm_metrics.csv")
    out = [subsection("CV5 Option B — Alarm-eval (clean signal, pooled across 5 folds = 15 records)")]
    h = (f"{'sub_design':<22} {'mode':<11} {'n_pred':>7} {'n_gs':>5} "
         f"{'TP':>4} {'FP':>4} {'FN':>4} {'F1':>7} {'P':>7} {'R':>7}")
    out.append(h)
    for _, r in df.iterrows():
        out.append(
            f"{r['sub_design']:<22} {r['mode']:<11} {r['n_pred_alarms']:>7} {r['n_gs_alarms']:>5} "
            f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4} {r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f}"
        )
    return "\n".join(out) + "\n", df


def render_cv5_alarms_temporal() -> str:
    df = pd.read_csv(ALARM_SUMMARY_DIR / "cv5_optB_alarm_temporal.csv")
    out = [subsection("CV5 Option B — Alarm temporal overlap (on the TPs)")]
    h = (f"{'sub_design':<22} {'mode':<11} {'TP':>4} "
         f"{'Jaccard':>8} {'frac_pred':>10} {'frac_gs':>9} "
         f"{'onset_s':>9} {'end_s':>8} {'durdiff_s':>10} {'dur_pred_h':>11} {'dur_ratio':>10}")
    out.append(h)
    for _, r in df.iterrows():
        out.append(
            f"{r['sub_design']:<22} {r['mode']:<11} {r['TP']:>4} "
            f"{r['jaccard_med']:>8.3f} {r['overlap_frac_pred_med']:>10.3f} {r['overlap_frac_gs_med']:>9.3f} "
            f"{r['onset_err_med_s']:>9.1f} {r['end_err_med_s']:>8.1f} {r['duration_diff_med_s']:>10.1f} "
            f"{r['pred_total_dur_h']:>11.1f} {r['dur_ratio']:>10.3f}"
        )
    return "\n".join(out) + "\n"


def render_existing_split_alarms() -> str:
    ex = pd.read_csv(EX_ALARM_CSV)
    two = ex[(ex.lead_tag == "two_lead") & (ex.condition == "clean")].copy()
    out = [subsection("Existing-split — Two-lead alarm-eval, per-trial (clean)")]
    h = (f"{'mode':<11} {'trial':>5} {'n_pred':>7} {'n_gs':>5} "
         f"{'TP':>4} {'FP':>4} {'FN':>4} {'F1':>7} {'P':>7} {'R':>7} "
         f"{'Jaccard':>8} {'onset_s':>9} {'end_s':>8} {'dur_pred_h':>11}")
    out.append(h)
    for _, r in two.sort_values(["mode", "trial"]).iterrows():
        dur_h = r['total_pred_duration_s'] / 3600.0
        out.append(
            f"{r['mode']:<11} {int(r['trial']):>5} {int(r['total_pred_alarms']):>7} {int(r['total_gs_alarms']):>5} "
            f"{int(r['tp']):>4} {int(r['fp']):>4} {int(r['fn']):>4} {r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} "
            f"{r['median_jaccard']:>8.3f} {r['median_onset_error_s']:>9.1f} {r['median_end_error_s']:>8.1f} {dur_h:>11.1f}"
        )
    return "\n".join(out) + "\n", two


def render_cross_cut_two_lead(cv5_alarms_df: pd.DataFrame, ex_alarms_df: pd.DataFrame,
                              cv5_overall: pd.DataFrame, ex_overall: pd.DataFrame) -> str:
    """One table: R^2 + F1 + Jaccard + onset, focused on two-lead."""
    out = [subsection("Cross-cut: Two-lead regression R^2 + alarm F1 + temporal (the key table)")]
    out.append(
        f"{'setup':<28} {'mode':<11} {'R2':<20} {'F1':<8} {'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'Jaccard':>8} {'onset_s':>9}"
    )
    # CV5 rows
    for _, ar in cv5_alarms_df[cv5_alarms_df.sub_design == "two_lead"].iterrows():
        mode = ar["mode"]
        rr = cv5_overall[(cv5_overall.sub_design == "two_lead") & (cv5_overall["mode"] == mode)].iloc[0]
        r2_field = fmt_pm(rr['test_R2_mean'], rr['test_R2_std'], 3)
        out.append(
            f"{'CV5 optB (pooled)':<28} {mode:<11} {r2_field:<20} {ar['f1']:>8.3f} "
            f"{int(ar['TP']):>4} {int(ar['FP']):>4} {int(ar['FN']):>4} n/a n/a"
        )
    # Existing-split rows
    for mode in ("supervised", "pretrained"):
        sub_alarm = ex_alarms_df[ex_alarms_df["mode"] == mode]
        sub_reg = ex_overall[ex_overall["mode"] == mode]
        if sub_alarm.empty or sub_reg.empty:
            continue
        r2_field = fmt_pm(sub_reg.test_R2.mean(), sub_reg.test_R2.std(), 3)
        # Best trial = highest F1
        best = sub_alarm.loc[sub_alarm.f1.idxmax()]
        out.append(
            f"{'Existing-split (best)':<28} {mode:<11} {r2_field:<20} {best['f1']:>8.3f} "
            f"{int(best['tp']):>4} {int(best['fp']):>4} {int(best['fn']):>4} "
            f"{best['median_jaccard']:>8.3f} {best['median_onset_error_s']:>9.1f}"
        )
        out.append(
            f"{'  best is trial':<28} {f'trial{int(best.trial):02d}':<11}"
        )
    out.append("")
    out.append("Notes:")
    out.append("  - Existing-split R^2 is mean +/- std across 5 trials (same test patients, varying seeds).")
    out.append("  - CV5 R^2 is mean +/- std across 5 folds (different test patients each fold).")
    out.append("  - Existing-split alarms run on all 16 records, but the model trained on 14 of them ")
    out.append("    -> ~88 of 101 GS alarms come from patients the model saw at training time.")
    out.append("  - CV5 alarms are pooled across 5 folds; each test patient's alarms come only ")
    out.append("    from a fold-model that NEVER saw that patient. Strict group K-fold compliance.")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    pieces = []
    pieces.append(hr("="))
    pieces.append("CV5 OPTION B — FULL RESULTS (regression + alarm-eval)")
    pieces.append(hr("="))
    pieces.append("")
    pieces.append("Designs: 3 sub-designs (two_lead, lead_II, lead_V2) x 2 modes ")
    pieces.append("         (supervised, pretrained) x 5 folds = 30 CV5 training runs.")
    pieces.append("         Plus single_lead_union = concat(lead_II + lead_V2) alarm CSVs ")
    pieces.append("         re-evaluated against the GS (2 union rows in alarm-eval table).")
    pieces.append("         All evaluated on the 15-record optB pool (s20121 always trains).")
    pieces.append("")

    pieces.append(section("SECTION 1 — REGRESSION RESULTS (test set, per-fold)"))
    pieces.append(render_cv5_regression_overall())
    pieces.append(render_cv5_regression_per_class())

    ex_reg_txt, ex_overall, ex_pc = render_existing_split_regression()
    pieces.append(section("SECTION 2 — EXISTING-SPLIT BASELINE (12/2/2 single split, 5 trials)"))
    pieces.append(ex_reg_txt)

    pieces.append(section("SECTION 3 — ALARM-EVAL RESULTS (clean signal, pooled CV5)"))
    cv5_alarms_txt, cv5_alarms_df = render_cv5_alarms()
    pieces.append(cv5_alarms_txt)
    pieces.append(render_cv5_alarms_temporal())

    pieces.append(section("SECTION 4 — EXISTING-SPLIT ALARM-EVAL BASELINE"))
    ex_alarms_txt, ex_alarms_df = render_existing_split_alarms()
    pieces.append(ex_alarms_txt)

    cv5_overall = pd.read_csv(CV5_SUMMARY_DIR / "cv5_optB_summary_overall.csv")
    pieces.append(section("SECTION 5 — TWO-LEAD CROSS-CUT (regression R^2 + alarm F1)"))
    pieces.append(render_cross_cut_two_lead(cv5_alarms_df, ex_alarms_df, cv5_overall, ex_overall))

    pieces.append(section("SECTION 6 — METHODOLOGY NOTES"))
    pieces.append("""
GS coverage:
  Total (record, lead) pairs in LTSTDB:   16 records * 2 leads = 32
  MANUAL_OMIT (skipped on both sides):     6 pairs
  Evaluable (record, lead) pairs:         26
    s20121 is fully omitted (both leads)
    Partial: (s20081, II only), (s20091, V2 only), (s20221, II only), (s20241, II only)

Both setups use the same GS file (gs_per_beat.csv).
Both setups have 101 GS alarms across the evaluable (record, lead) pairs.
The only difference is leakage:
  Existing-split: model trained on 12/14 of the contributing records;
                  alarm-eval runs on all 16 -> ~88 of 101 GS alarms
                  come from patients the model saw at training time.
  CV5 optB:       each fold's model never saw its own test patients;
                  pooled CSV concatenates 5 folds -> 15 records
                  with strict no-leakage discipline.

Two-lead model usage at alarm-eval:
  --alarms_ml_lead_mode=two_lead
  Single forward pass predicts ST values for both Lead II and Lead V2.
  FSM converts predicted ST -> alarm intervals per (record, lead).
  MANUAL_OMIT is applied symmetrically on prediction and GS sides.
""")

    OUT.write_text("\n".join(pieces))
    print(f"Wrote {OUT}")
    print(f"({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
