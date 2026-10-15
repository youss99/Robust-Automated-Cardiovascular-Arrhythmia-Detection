#!/usr/bin/env python3
"""
Build 5-fold CV manifests for Option B (lenient: only s20121 pinned to train).

Stratification:
  - Per-lead binary high/low using natural-gap cuts:
       II_high if II_abn_pct >= 25.0
       V2_high if V2_abn_pct >= 50.0
  - 4 strata: IIh_V2h, IIh_V2l, IIl_V2h, IIl_V2l
  - sklearn.model_selection.StratifiedGroupKFold(n_splits=5, random_state=42)

Pinned to train every fold: s20121 (both leads in MANUAL_OMIT).
Rotation pool: the other 15 records.

For each fold:
  - test = 3 records (from the stratified rotation)
  - val  = 2 records (carved from the fold's train pool, stratified)
  - train = remaining 10 records + s20121

Writes:
  src/scripts/custom_dataset/cv5_optB_fold_manifests/fold0[1-5].csv
  src/scripts/custom_dataset/cv5_optB_fold_manifests/_balance_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path("/lustre06/project/6039801/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SUMMARY_DIR = REPO / "results/fine-tune/saved_models/patient-segment-regression/ltstdb_median_3trial_summary"
WIDE_CSV = SUMMARY_DIR / "per_record_lead_summary_wide.csv"
MANIFEST_DIR = REPO / "src/scripts/custom_dataset/cv5_optB_fold_manifests"

PINNED_TO_TRAIN = ["s20121"]   # both leads omitted; can never be evaluated
II_CUT_PCT = 25.0
V2_CUT_PCT = 50.0
N_SPLITS = 5
N_VAL_PER_FOLD = 2
SEED = 42


def stratum_label(row) -> str:
    ii = "h" if row.II_abn_pct >= II_CUT_PCT else "l"
    v2 = "h" if row.V2_abn_pct >= V2_CUT_PCT else "l"
    return f"II{ii}_V2{v2}"


def pick_val_stratified(train_pool_df: pd.DataFrame, n_val: int, rng: np.random.Generator) -> list[str]:
    """Pick n_val records aiming for stratum diversity, then random fill."""
    strata = train_pool_df["stratum"].unique().tolist()
    rng.shuffle(strata)
    picks: list[str] = []
    for s in strata:
        if len(picks) >= n_val:
            break
        cands = train_pool_df[train_pool_df["stratum"] == s].index.tolist()
        if cands:
            rng.shuffle(cands)
            picks.append(cands[0])
    while len(picks) < n_val:
        remaining = [r for r in train_pool_df.index.tolist() if r not in picks]
        if not remaining:
            break
        rng.shuffle(remaining)
        picks.append(remaining[0])
    return picks


def main() -> int:
    if not WIDE_CSV.exists():
        print(f"ERROR: input CSV missing: {WIDE_CSV}", file=sys.stderr)
        return 2

    df = pd.read_csv(WIDE_CSV).set_index("record")
    df["stratum"] = df.apply(stratum_label, axis=1)

    print("=" * 78)
    print("Per-record table (all 16):")
    print("=" * 78)
    print(df.to_string())
    print()
    print("Strata counts over all 16 records:")
    print(df.groupby("stratum").size().to_string())
    print()

    # Rotation pool = all records except pinned
    pool = df.drop(PINNED_TO_TRAIN, errors="ignore").copy()
    print(f"Rotation pool (excluding pinned {PINNED_TO_TRAIN}): {len(pool)} records")
    print("Pool strata counts:")
    print(pool.groupby("stratum").size().to_string())
    print()

    records = pool.index.to_numpy()
    strata  = pool["stratum"].to_numpy()
    groups  = records  # each record is its own group

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_data = {}
    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(records, strata, groups), start=1):
        fold_data[fold_idx] = {
            "test":       records[test_idx].tolist(),
            "train_pool": records[train_idx].tolist(),
        }

    # Carve 2 val records from each fold's train pool (stratified, deterministic)
    for fold_idx, asg in fold_data.items():
        train_pool_df = pool.loc[asg["train_pool"]]
        rng = np.random.default_rng(SEED + fold_idx)
        val = pick_val_stratified(train_pool_df, N_VAL_PER_FOLD, rng)
        train_main = [r for r in asg["train_pool"] if r not in val]
        asg["val"] = val
        asg["train"] = sorted(train_main + PINNED_TO_TRAIN)
        asg["test"]  = sorted(asg["test"])
        asg["val"]   = sorted(asg["val"])

    # Write manifests
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for fold_idx, asg in fold_data.items():
        rows = []
        for r in asg["train"]:
            rows.append({"patient_id": r, "split": "train"})
        for r in asg["val"]:
            rows.append({"patient_id": r, "split": "val"})
        for r in asg["test"]:
            rows.append({"patient_id": r, "split": "test"})
        out = MANIFEST_DIR / f"fold{fold_idx:02d}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Wrote {out}")

    # Balance report
    print()
    print("=" * 78)
    print("Per-fold composition")
    print("=" * 78)
    for fold_idx, asg in fold_data.items():
        print(f"\nFold {fold_idx:02d}:")
        for role in ("train", "val", "test"):
            recs = asg[role]
            strs = df.loc[recs, "stratum"].value_counts().to_dict()
            print(f"  {role:<6}({len(recs):>2}): {recs}")
            print(f"         strata: {strs}")

    # Test-set stratum balance matrix (each record tested exactly once)
    print()
    print("=" * 78)
    print("Test-set stratum balance (each fold's test population)")
    print("=" * 78)
    all_strata = sorted(df["stratum"].unique())
    mat = pd.DataFrame(index=all_strata)
    for fold_idx, asg in fold_data.items():
        s = df.loc[asg["test"], "stratum"].value_counts()
        mat[f"fold{fold_idx:02d}"] = s.reindex(all_strata, fill_value=0)
    mat["total"] = mat.sum(axis=1)
    print(mat.to_string())

    # Coverage check: every non-pinned record appears in exactly one test fold
    test_records_all = sum((asg["test"] for asg in fold_data.values()), [])
    counts = pd.Series(test_records_all).value_counts()
    duplicates = counts[counts > 1].index.tolist()
    missing = [r for r in pool.index if r not in counts.index]
    print()
    print("Coverage check:")
    if duplicates:
        print(f"  WARN: records in test of multiple folds: {duplicates}")
    if missing:
        print(f"  WARN: records never in test: {missing}")
    if not duplicates and not missing:
        print(f"  OK — all {len(pool)} rotation-pool records tested exactly once")

    # Also save the balance report to a file next to the manifests
    report_path = MANIFEST_DIR / "_balance_report.txt"
    with open(report_path, "w") as f:
        f.write("Option B 5-fold CV manifests\n")
        f.write(f"Stratification: II_high if II_abn_pct >= {II_CUT_PCT}, V2_high if V2_abn_pct >= {V2_CUT_PCT}\n")
        f.write(f"Pinned to train: {PINNED_TO_TRAIN}\n")
        f.write(f"Random seed: {SEED}\n\n")
        f.write("Per-record table:\n")
        f.write(df.to_string() + "\n\n")
        f.write("Per-fold composition:\n")
        for fold_idx, asg in fold_data.items():
            f.write(f"\nFold {fold_idx:02d}:\n")
            for role in ("train", "val", "test"):
                recs = asg[role]
                strs = df.loc[recs, "stratum"].value_counts().to_dict()
                f.write(f"  {role:<6}({len(recs):>2}): {recs}\n")
                f.write(f"         strata: {strs}\n")
        f.write("\nTest-set stratum balance:\n")
        f.write(mat.to_string() + "\n")
    print(f"\nWrote balance report: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
