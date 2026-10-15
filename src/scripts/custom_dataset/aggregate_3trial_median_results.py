#!/usr/bin/env python3
"""
Aggregate the 3-trial median LTSTDB regression study.

For each of the 6 variants (two_lead / lead_II / lead_V2  x  supervised / pretrained),
this script:
  1. Locates the three trial dirs (trial01, trial02, trial03).
  2. Reads per-split metrics CSVs (_train_metrics_..., _valid_metrics_..., _test_metrics_...).
  3. Reads training_metrics_gpu0_*.csv to find each trial's best valid_loss epoch.
  4. Picks the "best trial" per variant (lowest min valid_loss).
  5. Emits two summary CSVs and prints a console table.

Outputs (under RESULTS_ROOT/ltstdb_median_3trial_summary/):
  all_trial_metrics.csv           - one row per trial (regression metrics + best-epoch info)
  aggregate_metric_summary.csv    - mean/std per (variant, mode) across 3 trials
  per_lead_metric_summary.csv     - per-lead (II/V2) mean/std (uses _per_class CSVs if present)
  best_trial_per_variant.csv      - the 6 best checkpoints to use for alarm-eval

Designed for downstream `run_alarm_eval_18.sh` to consume best_trial_per_variant.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path("/home/youss99/projects/def-majidk/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
RESULTS_ROOT = REPO_ROOT / "results/fine-tune/saved_models/patient-segment-regression"
SUMMARY_DIR = RESULTS_ROOT / "ltstdb_median_3trial_summary"

VARIANTS = [
    # (lead_tag, mode, ckpt_tag_in_modelname, lead_label)
    ("two_lead", "supervised",  None,                "II,V2"),
    ("two_lead", "pretrained",  "chapman_ii_v2",     "II,V2"),
    ("lead_II",  "supervised",  None,                "II"),
    ("lead_II",  "pretrained",  "chapman_ii_only",   "II"),
    ("lead_V2",  "supervised",  None,                "V2"),
    ("lead_V2",  "pretrained",  "chapman_v2_only",   "V2"),
]
# Two-lead has 5 trials; single-lead has 3. Missing trial dirs are skipped
# with a NOTE, so the union (01..05) is safe to scan.
TRIALS = ["01", "02", "03", "04", "05"]


def model_name(lead_tag: str, mode: str, ckpt_tag: str | None, trial: str) -> str:
    base = f"ltstdb_median_full16_12_2_2_omit_train_{lead_tag}_regression"
    if mode == "supervised":
        return f"{base}_supervised_trial{trial}"
    return f"{base}_pretrained_{ckpt_tag}_trial{trial}"


def read_split_metrics(run_dir: Path, model: str, split: str) -> dict:
    file_split = "valid" if split == "validation" else split
    path = run_dir / f"_{file_split}_metrics_{model}.csv"
    if not path.exists():
        return {}
    row = pd.read_csv(path).iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        # Existing files use prefix like "train_MSE" or just "MSE" - normalize
        metric = k.split("_", 1)[1] if "_" in k and k.split("_", 1)[0] in {"train", "valid", "test"} else k
        out[f"{split}_{metric}"] = v
    return out


def read_per_class_metrics(run_dir: Path, model: str, variant_id: tuple, split: str) -> list[dict]:
    file_split = "valid" if split == "validation" else split
    path = run_dir / f"_{file_split}_metrics_{model}_per_class.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    lead_tag, mode, _ckpt_tag, _lead_label = variant_id
    rows = []
    for r in df.to_dict("records"):
        r.update({
            "lead_tag": lead_tag,
            "mode": mode,
            "split": split,
            "model_name": model,
        })
        rows.append(r)
    return rows


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict] = []
    per_class_rows: list[dict] = []
    missing: list[str] = []

    for variant in VARIANTS:
        lead_tag, mode, ckpt_tag, lead_label = variant
        for trial in TRIALS:
            model = model_name(lead_tag, mode, ckpt_tag, trial)
            run_dir = RESULTS_ROOT / model
            train_log = run_dir / f"_training_metrics_gpu0_{model}.csv"

            if not train_log.exists():
                missing.append(str(train_log))
                continue

            training = pd.read_csv(train_log)
            best_idx = training["valid_loss"].idxmin()
            best = training.loc[best_idx]

            row: dict = {
                "lead_tag": lead_tag,
                "mode": mode,
                "ckpt_tag": ckpt_tag or "",
                "lead_label": lead_label,
                "trial": trial,
                "model_name": model,
                "best_epoch": int(best["epoch"]),
                "best_train_loss": best["train_loss"],
                "best_valid_loss": best["valid_loss"],
                "best_train_MSE": best.get("train_MSE", None),
                "best_valid_MSE": best.get("valid_MSE", None),
                "best_valid_MAE": best.get("valid_MAE", None),
                "best_valid_RMSE": best.get("valid_RMSE", None),
                "best_valid_R2":  best.get("valid_R2",  None),
            }
            for split in ["train", "validation", "test"]:
                row.update(read_split_metrics(run_dir, model, split))
                per_class_rows.extend(read_per_class_metrics(run_dir, model, variant, split))

            aggregate_rows.append(row)

    if not aggregate_rows:
        print("ERROR: no completed trial dirs found. Did training finish?", file=sys.stderr)
        if missing:
            print(f"Missing first few: {missing[:5]}", file=sys.stderr)
        return 1

    aggregate = pd.DataFrame(aggregate_rows).sort_values(["lead_tag", "mode", "trial"])
    aggregate.to_csv(SUMMARY_DIR / "all_trial_metrics.csv", index=False)

    # Mean/std per (lead_tag, mode) across trials
    metric_cols = [c for c in aggregate.columns if c not in {
        "lead_tag", "mode", "ckpt_tag", "lead_label", "trial", "model_name"
    }]
    summary = aggregate.groupby(["lead_tag", "mode"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(SUMMARY_DIR / "aggregate_metric_summary.csv", index=False)

    # Per-lead mean/std (only meaningful when _per_class CSVs exist)
    if per_class_rows:
        per_class = pd.DataFrame(per_class_rows)
        per_class.to_csv(SUMMARY_DIR / "all_trial_per_lead_metrics.csv", index=False)
        per_class_metric_cols = [c for c in ["MSE", "MAE", "RMSE", "R2"] if c in per_class.columns]
        if per_class_metric_cols:
            per_lead_summary = per_class.groupby(["lead_tag", "mode", "split", "label"])[per_class_metric_cols].agg(["mean", "std"])
            per_lead_summary.columns = [f"{m}_{stat}" for m, stat in per_lead_summary.columns]
            per_lead_summary.reset_index().to_csv(SUMMARY_DIR / "per_lead_metric_summary.csv", index=False)

    # Pick best trial per variant (lowest min valid_loss)
    best_rows = []
    for (lead_tag, mode), group in aggregate.groupby(["lead_tag", "mode"]):
        winner = group.loc[group["best_valid_loss"].idxmin()]
        ckpt_path = RESULTS_ROOT / winner["model_name"] / f"{winner['model_name']}_best.pt"
        best_rows.append({
            "lead_tag": lead_tag,
            "mode": mode,
            "trial": winner["trial"],
            "model_name": winner["model_name"],
            "best_valid_loss": winner["best_valid_loss"],
            "best_epoch": winner["best_epoch"],
            "checkpoint_path": str(ckpt_path),
            "checkpoint_exists": ckpt_path.exists(),
        })
    best_df = pd.DataFrame(best_rows).sort_values(["lead_tag", "mode"])
    best_df.to_csv(SUMMARY_DIR / "best_trial_per_variant.csv", index=False)

    # Console summary
    print()
    print("===== Aggregate summary (per variant) =====")
    cols = ["lead_tag", "mode"]
    for col in ["test_MAE_mean", "test_MAE_std", "test_RMSE_mean", "test_R2_mean", "best_valid_loss_mean"]:
        if col in summary.columns:
            cols.append(col)
    print(summary[cols].to_string(index=False))

    print()
    print("===== Best trial per variant =====")
    print(best_df[["lead_tag", "mode", "trial", "best_valid_loss", "best_epoch", "checkpoint_exists"]].to_string(index=False))

    if missing:
        print()
        print(f"WARNING: {len(missing)} expected file(s) missing — trials still running or failed:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    print()
    print(f"Wrote summaries to: {SUMMARY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
