#!/usr/bin/env python3
"""
Build 5-fold patient-grouped CV manifests with count-balanced assignment.

Three modes:
  two_lead  : balance II_abn + V2_abn sums, pin only s20121 (both-leads omitted)
  lead_II   : balance II_abn sum, pin records with II-omit (s20091, s20121)
  lead_V2   : balance V2_abn sum, pin records with V2-omit (s20081, s20121, s20221, s20241)

Algorithm: greedy placement (sorted desc by burden) into the fold that minimizes
the per-objective coefficient of variation, then local pair-swap refinement.

Val carving: 2 records per fold, picked deterministically from the train pool —
one from the low-burden half, one from the high-burden half, so val isn't
dominated by either extreme.

Output: one fold0X.csv per fold + _balance_report.txt next to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path("/lustre06/project/6039801/youss99/Project_1/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
SUMMARY_DIR = REPO / "results/fine-tune/saved_models/patient-segment-regression/ltstdb_median_3trial_summary"
LONG_CSV = SUMMARY_DIR / "median_per_record_lead_summary.csv"
SCRIPT_DIR = REPO / "src/scripts/custom_dataset"

MODE_CONFIG = {
    "two_lead": {
        "pinned":    ["s20121"],
        "objective": "both",
        "out_dir":   "cv5_optB_fold_manifests",
    },
    "lead_II": {
        "pinned":    ["s20091", "s20121"],
        "objective": "II",
        "out_dir":   "cv5_lead_II_fold_manifests",
    },
    "lead_V2": {
        "pinned":    ["s20081", "s20121", "s20221", "s20241"],
        "objective": "V2",
        "out_dir":   "cv5_lead_V2_fold_manifests",
    },
}


def fold_sizes_for(n: int, k: int) -> list[int]:
    sizes = [n // k] * k
    for i in range(n % k):
        sizes[i] += 1
    return sizes


def compute_score(folds: dict, df: pd.DataFrame, objective: str) -> float:
    if objective == "II":
        s = np.array([df.loc[recs, "II_abn"].sum() for recs in folds.values()])
        return s.std() / max(s.mean(), 1)
    if objective == "V2":
        s = np.array([df.loc[recs, "V2_abn"].sum() for recs in folds.values()])
        return s.std() / max(s.mean(), 1)
    # both
    ii = np.array([df.loc[recs, "II_abn"].sum() for recs in folds.values()])
    v2 = np.array([df.loc[recs, "V2_abn"].sum() for recs in folds.values()])
    return ii.std() / max(ii.mean(), 1) + v2.std() / max(v2.mean(), 1)


def sort_records(df: pd.DataFrame, objective: str) -> list[str]:
    if objective == "II":
        return df.sort_values("II_abn", ascending=False).index.tolist()
    if objective == "V2":
        return df.sort_values("V2_abn", ascending=False).index.tolist()
    tmp = df.copy()
    tmp["total"] = tmp["II_abn"] + tmp["V2_abn"]
    return tmp.sort_values("total", ascending=False).index.tolist()


def greedy_assign(df: pd.DataFrame, n_splits: int, objective: str) -> dict:
    sorted_recs = sort_records(df, objective)
    sizes = fold_sizes_for(len(sorted_recs), n_splits)
    folds = {i: [] for i in range(n_splits)}
    for r in sorted_recs:
        best_f, best_s = None, float("inf")
        for f in range(n_splits):
            if len(folds[f]) >= sizes[f]:
                continue
            folds[f].append(r)
            s = compute_score(folds, df, objective)
            folds[f].pop()
            if s < best_s:
                best_s, best_f = s, f
        folds[best_f].append(r)
    return folds


def local_swap(folds: dict, df: pd.DataFrame, objective: str, max_passes: int = 50):
    best = compute_score(folds, df, objective)
    fold_ids = list(folds.keys())
    for _ in range(max_passes):
        improved = False
        for a in fold_ids:
            for b in fold_ids:
                if a >= b:
                    continue
                for i in range(len(folds[a])):
                    for j in range(len(folds[b])):
                        ra = folds[a][i]
                        rb = folds[b][j]
                        folds[a][i], folds[b][j] = rb, ra
                        s = compute_score(folds, df, objective)
                        if s < best - 1e-9:
                            best = s
                            improved = True
                        else:
                            folds[a][i], folds[b][j] = ra, rb
                        if improved:
                            break
                    if improved: break
                if improved: break
            if improved: break
        if not improved:
            break
    return folds, best


def assert_valid_partition(folds: dict, expected: list[str]) -> None:
    flat = [r for recs in folds.values() for r in recs]
    if len(flat) != len(set(flat)):
        dupes = [r for r in flat if flat.count(r) > 1]
        raise AssertionError(f"DUPLICATE records in folds: {set(dupes)}")
    if set(flat) != set(expected):
        raise AssertionError(
            f"Partition mismatch — missing: {set(expected) - set(flat)}, extra: {set(flat) - set(expected)}"
        )


def pick_val_balanced(
    train_pool: list[str], df: pd.DataFrame, objective: str, n_val: int = 2,
    seed: int = 42,
    usage: dict | None = None,
    repeat_penalty: float = 0.10,
) -> list[str]:
    """Pick val records whose burden sums best approximate n_val * mean(train_pool),
    with a soft penalty against records that have already been val in earlier folds.

    score(pair) =
        burden_deviation(pair)                                    # primary
      + repeat_penalty * sum( usage[r] for r in pair )            # diversity term

    `usage` (caller-provided dict, record_id -> int) is incremented externally
    after this function returns, so subsequent folds penalize repeats. The
    penalty 0.10 was chosen so that picking a record for a 2nd time costs about
    as much as a 10 % relative deviation on one lead — meaningful but not
    dominant. Set repeat_penalty=0 to recover the prior pure-optimal behavior.

    With ~10-13 records and n_val=2, exhaustive search is ~45-80 candidates per fold.
    """
    from itertools import combinations
    train_df = df.loc[train_pool]
    usage = usage if usage is not None else {}

    def burden_dev(pair: tuple) -> float:
        recs = list(pair)
        if objective == "II":
            target = n_val * train_df["II_abn"].mean()
            actual = train_df.loc[recs, "II_abn"].sum()
            return abs(actual - target) / max(target, 1)
        if objective == "V2":
            target = n_val * train_df["V2_abn"].mean()
            actual = train_df.loc[recs, "V2_abn"].sum()
            return abs(actual - target) / max(target, 1)
        # both
        t_ii = n_val * train_df["II_abn"].mean()
        t_v2 = n_val * train_df["V2_abn"].mean()
        a_ii = train_df.loc[recs, "II_abn"].sum()
        a_v2 = train_df.loc[recs, "V2_abn"].sum()
        return abs(a_ii - t_ii) / max(t_ii, 1) + abs(a_v2 - t_v2) / max(t_v2, 1)

    def total_score(pair: tuple) -> float:
        return burden_dev(pair) + repeat_penalty * sum(usage.get(r, 0) for r in pair)

    if len(train_pool) < n_val:
        return sorted(train_pool)

    candidates = list(combinations(sorted(train_pool), n_val))
    best_pair = min(candidates, key=lambda p: (total_score(p), p))
    return sorted(list(best_pair))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODE_CONFIG.keys()), default=None,
                   help="Single mode to build. If omitted, builds all 3 modes.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not LONG_CSV.exists():
        print(f"ERROR: missing {LONG_CSV}", file=sys.stderr)
        return 2

    long = pd.read_csv(LONG_CSV)
    per_rec = (long.pivot_table(index="record", columns="lead", values="n_abnormal", aggfunc="sum")
                   .rename(columns={"II": "II_abn", "V2": "V2_abn"}))

    modes = [args.mode] if args.mode else list(MODE_CONFIG.keys())

    for mode in modes:
        cfg = MODE_CONFIG[mode]
        pinned = cfg["pinned"]
        objective = cfg["objective"]
        out_dir = SCRIPT_DIR / cfg["out_dir"]

        pool = per_rec.drop(pinned, errors="ignore").copy()
        print()
        print("=" * 78)
        print(f"MODE: {mode}")
        print(f"  pinned:    {pinned}")
        print(f"  objective: {objective}")
        print(f"  pool:      {len(pool)} records")
        print("=" * 78)

        folds = greedy_assign(pool, n_splits=5, objective=objective)
        assert_valid_partition(folds, pool.index.tolist())
        folds, final_score = local_swap(folds, pool, objective)
        assert_valid_partition(folds, pool.index.tolist())

        out_dir.mkdir(parents=True, exist_ok=True)

        # Write manifests + collect per-fold info
        # Track val-usage across folds so subsequent folds penalize repeats.
        val_usage: dict = {}
        fold_info = []
        for fid in range(5):
            test_recs = sorted(folds[fid])
            train_pool_recs = [r for r in pool.index if r not in test_recs]
            val_recs = pick_val_balanced(train_pool_recs, pool, objective,
                                          n_val=2, seed=args.seed + fid,
                                          usage=val_usage, repeat_penalty=0.10)
            for r in val_recs:
                val_usage[r] = val_usage.get(r, 0) + 1
            train_recs = sorted([r for r in train_pool_recs if r not in val_recs] + pinned)

            fold_csv = out_dir / f"fold{fid+1:02d}.csv"
            rows = []
            for r in train_recs:
                rows.append({"patient_id": r, "split": "train"})
            for r in val_recs:
                rows.append({"patient_id": r, "split": "val"})
            for r in test_recs:
                rows.append({"patient_id": r, "split": "test"})
            pd.DataFrame(rows).to_csv(fold_csv, index=False)

            fold_info.append({
                "fold": fid + 1,
                "test_records":  test_recs,
                "val_records":   val_recs,
                "train_records": train_recs,
                "n_test":  len(test_recs),
                "n_val":   len(val_recs),
                "n_train": len(train_recs),
                "test_II_abn":   int(pool.loc[test_recs, "II_abn"].sum()),
                "test_V2_abn":   int(pool.loc[test_recs, "V2_abn"].sum()),
                "val_II_abn":    int(pool.loc[val_recs, "II_abn"].sum()),
                "val_V2_abn":    int(pool.loc[val_recs, "V2_abn"].sum()),
            })
            print(f"  fold{fid+1:02d}: test={test_recs}  val={val_recs}")

        # Balance report
        report_path = out_dir / "_balance_report.txt"
        with open(report_path, "w") as f:
            f.write(f"CV manifests for mode: {mode}\n")
            f.write(f"Pinned to train every fold: {pinned}\n")
            f.write(f"Balance objective: {objective}\n")
            f.write(f"Rotation pool size: {len(pool)}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Final score (sum CV): {final_score:.4f}\n\n")

            f.write(f"{'fold':<6}{'n_test':>7}{'n_val':>6}{'n_train':>8}"
                    f"{'test_II':>10}{'test_V2':>10}{'val_II':>9}{'val_V2':>9}\n")
            for info in fold_info:
                f.write(f"fold{info['fold']:02}{info['n_test']:>7}{info['n_val']:>6}{info['n_train']:>8}"
                        f"{info['test_II_abn']:>10}{info['test_V2_abn']:>10}"
                        f"{info['val_II_abn']:>9}{info['val_V2_abn']:>9}\n")

            f.write("\nPer-fold details:\n")
            for info in fold_info:
                f.write(f"\nfold{info['fold']:02}:\n")
                f.write(f"  test  ({info['n_test']:>2}): {info['test_records']}\n")
                f.write(f"  val   ({info['n_val']:>2}): {info['val_records']}\n")
                f.write(f"  train ({info['n_train']:>2}): {info['train_records']}\n")

        print(f"\nWrote {len(fold_info)} manifests to: {out_dir}")
        print(f"Wrote balance report:                {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
