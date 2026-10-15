"""
Aggregate alarm-eval + regression metrics for the CV5 Option B experiment.

Reads:
  - 6 pooled alarm CSVs               (alarm counts per model)
  - 6 eval summary_stats CSVs         (TP/FP/FN/F1 vs GS)
  - cv5_optB_summary_overall.csv      (regression R^2 etc., from
                                       aggregate_cv5_optB_results.py)

Produces:
  - single_lead_union/*               (concat of lead_II + lead_V2 pooled alarms,
                                       re-evaluated against the GS, per mode)
  - cv5_optB_alarm_metrics.csv        (6 + 2 union rows, raw)
  - cv5_optB_alarm_summary.csv        (8 rows, joined with regression metrics)
  - cv5_optB_alarms_results.txt       (human-readable table)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd

NEUROKIT_ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Neurokit2_Pipeline")
PATCHECG_ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
ALARM_ROOT = NEUROKIT_ROOT / "alarm_eval_results_cv5_optB"
REG_SUMMARY = (PATCHECG_ROOT / "results/fine-tune/saved_models/patient-segment-regression"
               / "cv5_optB_summary" / "cv5_optB_summary_overall.csv")
GS_CSV = Path("/scratch/youss99/Gold_Standard_Alarms/gs_per_beat.csv")
OUT_DIR = ALARM_ROOT / "_summary"
UNION_DIR = ALARM_ROOT / "single_lead_union"

# (sub_design, mode, ckpt_tag)
RUNS = [
    ("two_lead", "supervised",  None),
    ("two_lead", "pretrained",  "chapman_ii_v2"),
    ("lead_II",  "supervised",  None),
    ("lead_II",  "pretrained",  "chapman_ii_only"),
    ("lead_V2",  "supervised",  None),
    ("lead_V2",  "pretrained",  "chapman_v2_only"),
]


def mode_name(mode: str, ckpt_tag: str | None) -> str:
    return f"pretrained_{ckpt_tag}" if mode == "pretrained" else "supervised"


def run_dir(sub_design: str, mode: str, ckpt_tag: str | None) -> Path:
    return ALARM_ROOT / sub_design / mode_name(mode, ckpt_tag) / "clean"


def read_summary_stats(d: Path) -> pd.Series | None:
    """Find the single eval summary_stats.csv under d/eval_outputs/."""
    for cand in (d / "eval_outputs").glob("*summary_stats.csv"):
        return pd.read_csv(cand).iloc[0]
    return None


def collect_direct():
    """Read TP/FP/FN/F1 + alarm count for each of the 6 (sub_design, mode) runs."""
    rows = []
    for sub_design, mode, ckpt_tag in RUNS:
        d = run_dir(sub_design, mode, ckpt_tag)
        pooled = d / "ml_alarms_pooled.csv"
        n_alarms = len(pd.read_csv(pooled)) if pooled.exists() else None
        stats = read_summary_stats(d)
        if stats is None:
            print(f"WARN: missing summary_stats for {sub_design}/{mode_name(mode, ckpt_tag)}")
            continue
        rows.append({
            "sub_design": sub_design,
            "mode": mode,
            "ckpt_tag": ckpt_tag or "",
            "n_alarms": n_alarms,
            "n_pred_alarms": int(stats["total_pred_alarms"]),
            "n_gs_alarms": int(stats["total_gs_alarms"]),
            "TP": int(stats["tp"]),
            "FP": int(stats["fp"]),
            "FN": int(stats["fn"]),
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
        })
    return pd.DataFrame(rows)


def compute_single_lead_union(direct: pd.DataFrame) -> pd.DataFrame:
    """For each mode, concat lead_II + lead_V2 pooled alarms, re-eval vs GS."""
    sys.path.insert(0, str(NEUROKIT_ROOT))
    from ltst_pipeline.evaluation import compute_performance_simple, OUT_DIR as EVAL_OUT_REL

    union_rows = []
    for mode in ("supervised", "pretrained"):
        ii_rows = direct[(direct.sub_design == "lead_II") & (direct["mode"] == mode)]
        v2_rows = direct[(direct.sub_design == "lead_V2") & (direct["mode"] == mode)]
        if ii_rows.empty or v2_rows.empty:
            continue
        ii_ck = ii_rows.iloc[0]["ckpt_tag"]
        v2_ck = v2_rows.iloc[0]["ckpt_tag"]
        ii_csv = run_dir("lead_II", mode, ii_ck or None) / "ml_alarms_pooled.csv"
        v2_csv = run_dir("lead_V2", mode, v2_ck or None) / "ml_alarms_pooled.csv"
        ii_df = pd.read_csv(ii_csv)
        v2_df = pd.read_csv(v2_csv)
        union_df = pd.concat([ii_df, v2_df], ignore_index=True)

        out_dir = UNION_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        union_path = out_dir / "ml_alarms_union.csv"
        union_df.to_csv(union_path, index=False)

        # compute_performance_simple writes to <cwd>/<EVAL_OUT_REL>/
        (out_dir / EVAL_OUT_REL).mkdir(parents=True, exist_ok=True)
        saved_cwd = os.getcwd()
        try:
            os.chdir(out_dir)
            compute_performance_simple(
                gs_path=str(GS_CSV),
                pred_paths=[str(union_path)],
                label_prefix=f"union_{mode}_",
            )
        finally:
            os.chdir(saved_cwd)
        cands = list((out_dir / EVAL_OUT_REL).glob("*summary_stats.csv"))
        if not cands:
            print(f"WARN: union eval for {mode} produced no summary_stats")
            continue
        stats = pd.read_csv(cands[0]).iloc[0]
        union_rows.append({
            "sub_design": "single_lead_union",
            "mode": mode,
            "ckpt_tag": "",
            "n_alarms": len(union_df),
            "n_pred_alarms": int(stats["total_pred_alarms"]),
            "n_gs_alarms": int(stats["total_gs_alarms"]),
            "TP": int(stats["tp"]),
            "FP": int(stats["fp"]),
            "FN": int(stats["fn"]),
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
        })
    return pd.DataFrame(union_rows)


def join_regression(metrics: pd.DataFrame) -> pd.DataFrame:
    if not REG_SUMMARY.exists():
        print(f"WARN: regression summary missing at {REG_SUMMARY}; skipping join")
        return metrics
    reg = pd.read_csv(REG_SUMMARY)
    # reg has columns: sub_design, mode, test_R2_mean, test_R2_std, ...
    reg = reg[["sub_design", "mode", "test_R2_mean", "test_R2_std"]]
    return metrics.merge(reg, on=["sub_design", "mode"], how="left")


def render_txt(metrics: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append("CV5 Option B alarm-eval results (clean signal, pooled across 5 folds = 15 records)")
    lines.append("=" * 96)
    lines.append("")
    header = (f"{'sub_design':<22} {'mode':<11} {'n_pred':>7} {'n_gs':>5} "
              f"{'TP':>5} {'FP':>5} {'FN':>5} "
              f"{'F1':>7} {'P':>7} {'R':>7}  {'R2':>14}")
    lines.append(header)
    lines.append("-" * 96)
    for _, r in metrics.iterrows():
        r2_field = (f"{r['test_R2_mean']:.3f} +/- {r['test_R2_std']:.3f}"
                    if pd.notna(r.get("test_R2_mean", float('nan'))) else "n/a")
        lines.append(
            f"{r['sub_design']:<22} {r['mode']:<11} {r['n_pred_alarms']:>7} {r['n_gs_alarms']:>5} "
            f"{r['TP']:>5} {r['FP']:>5} {r['FN']:>5} "
            f"{r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f}  {r2_field:>14}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Pooled CSV concatenates 5 per-fold alarm CSVs (each fold's test patients only).")
    lines.append("  - single_lead_union = concat(lead_II clean alarms, lead_V2 clean alarms),")
    lines.append("    re-evaluated against the GS. Comparable to the two_lead row of the same mode.")
    lines.append("  - R2 column is mean +/- std of overall test R^2 across the 5 CV folds")
    lines.append("    (from aggregate_cv5_optB_results.py).")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    UNION_DIR.mkdir(parents=True, exist_ok=True)

    direct = collect_direct()
    if direct.empty:
        print("ERROR: no direct alarm-eval results found.")
        return

    union = compute_single_lead_union(direct)
    metrics = pd.concat([direct, union], ignore_index=True)
    metrics = join_regression(metrics)

    metrics.to_csv(OUT_DIR / "cv5_optB_alarm_metrics.csv", index=False)
    metrics.to_csv(OUT_DIR / "cv5_optB_alarm_summary.csv", index=False)

    txt = render_txt(metrics)
    (OUT_DIR / "cv5_optB_alarms_results.txt").write_text(txt)
    print(txt)
    print(f"\nWrote outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
