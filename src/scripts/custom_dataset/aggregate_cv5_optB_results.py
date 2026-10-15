"""
Aggregate regression metrics across the 5-fold CV Option B experiment.

Reads per-fold test_metrics CSVs produced by patchECG_finetune.py for all six
(sub-design, train-mode) combos:

  optB_two_lead  x  {supervised, pretrained_chapman_ii_v2}
  optB_lead_II   x  {supervised, pretrained_chapman_ii_only}
  optB_lead_V2   x  {supervised, pretrained_chapman_v2_only}

Outputs (under .../patient-segment-regression/cv5_optB_summary/):
  cv5_optB_per_fold_overall.csv     - one row per (sub_design, mode, fold)
  cv5_optB_per_fold_per_class.csv   - one row per (sub_design, mode, fold, lead)
  cv5_optB_summary_overall.csv      - mean +/- std across 5 folds per variant
  cv5_optB_summary_per_class.csv    - mean +/- std per (variant, lead)
  cv5_optB_results.txt              - human-readable summary with comparison
                                       against the existing-split (5-trial)
                                       baseline.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection/results/fine-tune/saved_models/patient-segment-regression")
OUT_DIR = ROOT / "cv5_optB_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUB_DESIGNS = [
    ("two_lead", "supervised",  None),
    ("two_lead", "pretrained",  "chapman_ii_v2"),
    ("lead_II",  "supervised",  None),
    ("lead_II",  "pretrained",  "chapman_ii_only"),
    ("lead_V2",  "supervised",  None),
    ("lead_V2",  "pretrained",  "chapman_v2_only"),
]

# Existing-split (single 12/2/2 split, 5-trial) baseline name pattern,
# for the comparison block at the bottom of the txt summary.
EXISTING_SUB_DESIGNS = [
    ("two_lead", "supervised",  None),
    ("two_lead", "pretrained",  "chapman_ii_v2"),
    ("lead_II",  "supervised",  None),
    ("lead_II",  "pretrained",  "chapman_ii_only"),
    ("lead_V2",  "supervised",  None),
    ("lead_V2",  "pretrained",  "chapman_v2_only"),
]


def cv5_model_name(sub_design: str, mode: str, ckpt_tag: str | None, fold: int) -> str:
    tag = f"pretrained_{ckpt_tag}" if mode == "pretrained" else "supervised"
    return f"ltstdb_median_cv5_optB_{sub_design}_regression_{tag}_fold{fold:02d}"


# Single-lead "native" CV designs: smaller record pools where the omitted lead
# is pinned to train. Used for the native-vs-optB head-to-head.
NATIVE_SUB_DESIGNS = [
    ("lead_II", "supervised",  None),
    ("lead_II", "pretrained",  "chapman_ii_only"),
    ("lead_V2", "supervised",  None),
    ("lead_V2", "pretrained",  "chapman_v2_only"),
]


def cv5_native_model_name(sub_design: str, mode: str, ckpt_tag: str | None, fold: int) -> str:
    tag = f"pretrained_{ckpt_tag}" if mode == "pretrained" else "supervised"
    return f"ltstdb_median_cv5_{sub_design}_regression_{tag}_fold{fold:02d}"


def existing_model_name(sub_design: str, mode: str, ckpt_tag: str | None, trial: int) -> str:
    tag = f"pretrained_{ckpt_tag}" if mode == "pretrained" else "supervised"
    return (f"ltstdb_median_full16_12_2_2_omit_train_{sub_design}_regression_"
            f"{tag}_trial{trial:02d}")


def read_pair(model_dir: Path, model_name: str):
    """Return (overall_df, per_class_df) for one run, or (None, None) if missing."""
    ov = model_dir / f"_test_metrics_{model_name}.csv"
    pc = model_dir / f"_test_metrics_{model_name}_per_class.csv"
    o = pd.read_csv(ov) if ov.exists() else None
    p = pd.read_csv(pc) if pc.exists() else None
    return o, p


def collect_cv5():
    per_fold_overall, per_fold_per_class = [], []
    missing = []
    for sub_design, mode, ckpt_tag in SUB_DESIGNS:
        for fold in range(1, 6):
            name = cv5_model_name(sub_design, mode, ckpt_tag, fold)
            d = ROOT / name
            o, p = read_pair(d, name)
            if o is None or p is None:
                missing.append(name)
                continue
            for df in (o, p):
                df.insert(0, "fold", fold)
                df.insert(0, "mode", mode)
                df.insert(0, "sub_design", sub_design)
            per_fold_overall.append(o)
            per_fold_per_class.append(p)
    return (
        (pd.concat(per_fold_overall, ignore_index=True) if per_fold_overall else pd.DataFrame()),
        (pd.concat(per_fold_per_class, ignore_index=True) if per_fold_per_class else pd.DataFrame()),
        missing,
    )


def collect_native():
    per_fold_overall, per_fold_per_class = [], []
    missing = []
    for sub_design, mode, ckpt_tag in NATIVE_SUB_DESIGNS:
        for fold in range(1, 6):
            name = cv5_native_model_name(sub_design, mode, ckpt_tag, fold)
            d = ROOT / name
            o, p = read_pair(d, name)
            if o is None or p is None:
                missing.append(name)
                continue
            for df in (o, p):
                df.insert(0, "fold", fold)
                df.insert(0, "mode", mode)
                df.insert(0, "sub_design", sub_design)
            per_fold_overall.append(o)
            per_fold_per_class.append(p)
    return (
        (pd.concat(per_fold_overall, ignore_index=True) if per_fold_overall else pd.DataFrame()),
        (pd.concat(per_fold_per_class, ignore_index=True) if per_fold_per_class else pd.DataFrame()),
        missing,
    )


def collect_existing():
    rows_o, rows_p = [], []
    for sub_design, mode, ckpt_tag in EXISTING_SUB_DESIGNS:
        for trial in range(1, 6):
            name = existing_model_name(sub_design, mode, ckpt_tag, trial)
            d = ROOT / name
            o, p = read_pair(d, name)
            if o is None or p is None:
                continue
            for df in (o, p):
                df.insert(0, "trial", trial)
                df.insert(0, "mode", mode)
                df.insert(0, "sub_design", sub_design)
            rows_o.append(o)
            rows_p.append(p)
    return (
        (pd.concat(rows_o, ignore_index=True) if rows_o else pd.DataFrame()),
        (pd.concat(rows_p, ignore_index=True) if rows_p else pd.DataFrame()),
    )


def summarize_overall(per_fold_overall: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    metric_cols = [c for c in per_fold_overall.columns
                   if c not in ("sub_design", "mode", fold_col)]
    grouped = per_fold_overall.groupby(["sub_design", "mode"])[metric_cols]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def summarize_per_class(per_fold_pc: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    metric_cols = ["MSE", "MAE", "RMSE", "R2"]
    grouped = per_fold_pc.groupby(["sub_design", "mode", "label"])[metric_cols]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def render_summary_txt(cv5_overall: pd.DataFrame, cv5_pc: pd.DataFrame,
                       ex_overall: pd.DataFrame, ex_pc: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("CV5 Option B regression results — 3 sub-designs x 2 modes x 5 folds")
    lines.append("=" * 78)
    lines.append("")

    # Overall block
    lines.append("OVERALL test metrics (mean +/- std across 5 folds)")
    lines.append("-" * 78)
    header = f"{'sub_design':<10} {'mode':<11} {'MSE':<24} {'MAE':<24} {'RMSE':<24} {'R2':<24}"
    lines.append(header)
    for _, row in cv5_overall.iterrows():
        lines.append(
            f"{row['sub_design']:<10} {row['mode']:<11} "
            f"{fmt_pm(row['test_MSE_mean'],  row['test_MSE_std']):<24} "
            f"{fmt_pm(row['test_MAE_mean'],  row['test_MAE_std']):<24} "
            f"{fmt_pm(row['test_RMSE_mean'], row['test_RMSE_std']):<24} "
            f"{fmt_pm(row['test_R2_mean'],   row['test_R2_std'], digits=3):<24}"
        )
    lines.append("")

    # Per-class block
    lines.append("PER-LEAD test metrics (mean +/- std across 5 folds)")
    lines.append("-" * 78)
    header = (f"{'sub_design':<10} {'mode':<11} {'lead':<18} "
              f"{'MSE':<22} {'MAE':<22} {'R2':<22}")
    lines.append(header)
    for _, row in cv5_pc.iterrows():
        lines.append(
            f"{row['sub_design']:<10} {row['mode']:<11} {row['label']:<18} "
            f"{fmt_pm(row['MSE_mean'], row['MSE_std']):<22} "
            f"{fmt_pm(row['MAE_mean'], row['MAE_std']):<22} "
            f"{fmt_pm(row['R2_mean'],  row['R2_std'], digits=3):<22}"
        )
    lines.append("")

    # Single-lead model vs two-lead model's per-lead head
    lines.append("=" * 78)
    lines.append("Single-lead model vs two-lead model's per-lead head (apples-to-apples)")
    lines.append("=" * 78)
    lines.append("Lead II: single-lead lead_II  vs  two-lead head target_lead0_mV")
    lines.append("Lead V2: single-lead lead_V2  vs  two-lead head target_lead1_mV")
    lines.append("")
    # Build (mode, lead_view) -> {single_lead_R2: (m,s), two_lead_R2: (m,s)}
    pc = cv5_pc.set_index(["sub_design", "mode", "label"])
    rows = []
    for mode in ("supervised", "pretrained"):
        # Lead II
        if ("lead_II", mode, "target_lead0_mV") in pc.index and \
           ("two_lead", mode, "target_lead0_mV") in pc.index:
            sl = pc.loc[("lead_II",  mode, "target_lead0_mV")]
            tl = pc.loc[("two_lead", mode, "target_lead0_mV")]
            rows.append(("Lead II", mode,
                         sl["R2_mean"], sl["R2_std"],
                         tl["R2_mean"], tl["R2_std"],
                         sl["MSE_mean"], tl["MSE_mean"]))
        # Lead V2
        if ("lead_V2", mode, "target_lead1_mV") in pc.index and \
           ("two_lead", mode, "target_lead1_mV") in pc.index:
            sl = pc.loc[("lead_V2",  mode, "target_lead1_mV")]
            tl = pc.loc[("two_lead", mode, "target_lead1_mV")]
            rows.append(("Lead V2", mode,
                         sl["R2_mean"], sl["R2_std"],
                         tl["R2_mean"], tl["R2_std"],
                         sl["MSE_mean"], tl["MSE_mean"]))
    header = f"{'lead':<8} {'mode':<11} {'R2 single-lead':<22} {'R2 two-lead head':<22} {'MSE SL':<10} {'MSE 2L':<10} {'winner':<10}"
    lines.append(header)
    for lead, mode, sl_m, sl_s, tl_m, tl_s, sl_mse, tl_mse in rows:
        winner = "single" if sl_m > tl_m else ("two-lead" if tl_m > sl_m else "tie")
        lines.append(
            f"{lead:<8} {mode:<11} "
            f"{fmt_pm(sl_m, sl_s, digits=3):<22} "
            f"{fmt_pm(tl_m, tl_s, digits=3):<22} "
            f"{sl_mse:<10.4f} {tl_mse:<10.4f} {winner:<10}"
        )
    lines.append("")

    # Existing-split comparison (if available)
    nat_overall = render_summary_txt.nat_overall if hasattr(render_summary_txt, 'nat_overall') else pd.DataFrame()
    if not nat_overall.empty:
        lines.append("=" * 78)
        lines.append("Native vs optB head-to-head (single-lead models)")
        lines.append("=" * 78)
        lines.append("native = smaller record pool (lead_II: 14 records, lead_V2: 12 records)")
        lines.append("         with the omitted-lead records pinned to train")
        lines.append("optB   = full 15-record pool, only s20121 pinned to train")
        lines.append("")
        header = f"{'sub_design':<10} {'mode':<11} {'R2 native':<22} {'R2 optB':<22} {'winner':<10}"
        lines.append(header)
        for _, n in nat_overall.iterrows():
            o = cv5_overall[(cv5_overall.sub_design == n.sub_design) &
                            (cv5_overall["mode"] == n["mode"])]
            if o.empty:
                continue
            n_r2_m, n_r2_s = n["test_R2_mean"], n["test_R2_std"]
            o_r2_m, o_r2_s = o.iloc[0]["test_R2_mean"], o.iloc[0]["test_R2_std"]
            winner = "native" if n_r2_m > o_r2_m else ("optB" if o_r2_m > n_r2_m else "tie")
            lines.append(
                f"{n['sub_design']:<10} {n['mode']:<11} "
                f"{fmt_pm(n_r2_m, n_r2_s, digits=3):<22} "
                f"{fmt_pm(o_r2_m, o_r2_s, digits=3):<22} "
                f"{winner:<10}"
            )
        lines.append("")

    if not ex_overall.empty:
        lines.append("=" * 78)
        lines.append("Comparison vs existing-split (12/2/2, 5 trials)")
        lines.append("=" * 78)
        header = f"{'sub_design':<10} {'mode':<11} {'R2 CV5':<22} {'R2 existing':<22}"
        lines.append(header)
        for _, c in cv5_overall.iterrows():
            ex_row = ex_overall[(ex_overall.sub_design == c.sub_design) &
                                (ex_overall["mode"] == c["mode"])]
            ex_text = (fmt_pm(ex_row.iloc[0]["test_R2_mean"],
                              ex_row.iloc[0]["test_R2_std"], digits=3)
                       if not ex_row.empty else "N/A")
            lines.append(
                f"{c['sub_design']:<10} {c['mode']:<11} "
                f"{fmt_pm(c['test_R2_mean'], c['test_R2_std'], digits=3):<22} "
                f"{ex_text:<22}"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    cv5_pf_o, cv5_pf_pc, missing = collect_cv5()
    if missing:
        print(f"WARN: {len(missing)} runs missing test_metrics:")
        for n in missing:
            print(f"  - {n}")

    if cv5_pf_o.empty:
        print("ERROR: no CV5 results found; aborting")
        return

    cv5_pf_o.to_csv(OUT_DIR / "cv5_optB_per_fold_overall.csv", index=False)
    cv5_pf_pc.to_csv(OUT_DIR / "cv5_optB_per_fold_per_class.csv", index=False)

    cv5_summary_overall = summarize_overall(cv5_pf_o)
    cv5_summary_per_class = summarize_per_class(cv5_pf_pc)
    cv5_summary_overall.to_csv(OUT_DIR / "cv5_optB_summary_overall.csv", index=False)
    cv5_summary_per_class.to_csv(OUT_DIR / "cv5_optB_summary_per_class.csv", index=False)

    nat_pf_o, nat_pf_pc, nat_missing = collect_native()
    if nat_missing:
        print(f"WARN: {len(nat_missing)} native runs missing test_metrics:")
        for n in nat_missing[:10]:
            print(f"  - {n}")
    if not nat_pf_o.empty:
        nat_pf_o.to_csv(OUT_DIR / "cv5_native_per_fold_overall.csv", index=False)
        nat_pf_pc.to_csv(OUT_DIR / "cv5_native_per_fold_per_class.csv", index=False)
        nat_summary_overall = summarize_overall(nat_pf_o)
        nat_summary_per_class = summarize_per_class(nat_pf_pc)
        nat_summary_overall.to_csv(OUT_DIR / "cv5_native_summary_overall.csv", index=False)
        nat_summary_per_class.to_csv(OUT_DIR / "cv5_native_summary_per_class.csv", index=False)
        # Stash on render fn so the txt summary can reach it without extra plumbing
        render_summary_txt.nat_overall = nat_summary_overall

    ex_pf_o, ex_pf_pc = collect_existing()
    ex_summary_overall = summarize_overall(ex_pf_o, fold_col="trial") if not ex_pf_o.empty else pd.DataFrame()
    ex_summary_per_class = summarize_per_class(ex_pf_pc, fold_col="trial") if not ex_pf_pc.empty else pd.DataFrame()

    txt = render_summary_txt(cv5_summary_overall, cv5_summary_per_class,
                             ex_summary_overall, ex_summary_per_class)
    (OUT_DIR / "cv5_optB_results.txt").write_text(txt)
    print(txt)
    print(f"\nWrote 5 CSVs + cv5_optB_results.txt under {OUT_DIR}")


if __name__ == "__main__":
    main()
