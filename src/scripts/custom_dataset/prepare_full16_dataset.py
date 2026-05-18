"""
prepare_full16_dataset.py

Build a "full 16" variant of the LTSTDB regression datasets:
  * Includes ALL 16 LTSTDB records (no MANUAL_OMIT exclusion at dataset level)
  * Stratifies 16 patients into 10 train / 3 val / 3 test via exhaustive
    search, optimizing two objectives:
      1. PRIMARY (segment-level balance): per-split abnormal-segment
         fraction on each lead should match the global abnormal fraction
         as closely as possible. "Abnormal" = |ST| >= 0.1 mV.
      2. SECONDARY (variance preference): train's per-lead target std
         should be >= val and test target std. Strong preference to
         ensure the model trains on at least as wide a distribution as
         it'll be evaluated on.
  * Produces parallel datasets for both `mean` and `median` aggregators.

Notes on the omit-affected records (s20081, s20091, s20121, s20221, s20241):
  The .16a per-segment ST values for these records are valid; only the
  FSM-level alarms are degenerate (they're "full-record alarms" per
  Abdelazez 2017). For regression *training*, these records still
  provide useful supervision. MANUAL_OMIT is applied downstream at the
  FSM-evaluation stage, not at dataset level here.

Output (under <repo>/custom_datasets/):
    ltstdb_regression_mean_100hz_full16_nocompress/
        PatientSegmentRegressionDataset.h5
        modified_dataset/modified_dataset.csv
    ltstdb_regression_median_100hz_full16_nocompress/
        PatientSegmentRegressionDataset.h5
        modified_dataset/modified_dataset.csv

Usage:
    cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
    .venv/bin/python src/scripts/custom_dataset/prepare_full16_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd


REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DATA_ROOT = REPO / "custom_datasets"

# Split enum values must match src.core.datasets.ecg_interface.Split
SPLIT_TRAIN = 1
SPLIT_TEST = 2
SPLIT_VALIDATION = 3

# ST threshold for abnormal classification
ABNORMAL_THR = 0.1

# Default target split sizes for 16 patients (overridable via CLI)
DEFAULT_N_TRAIN = 10
DEFAULT_N_VAL = 3
DEFAULT_N_TEST = 3

# All 16 LTSTDB records
ALL_PATIENTS = [
    "s20011", "s20051", "s20061", "s20071", "s20081", "s20091",
    "s20101", "s20111", "s20121", "s20131", "s20141", "s20201",
    "s20211", "s20221", "s20231", "s20241",
]

VARIANTS = [
    {
        "name": "mean",
        "src_root": DATA_ROOT / "ltstdb_regression_mean_100hz_balanced_min5_nocompress",
        "dst_root": DATA_ROOT / "ltstdb_regression_mean_100hz_full16_nocompress",
    },
    {
        "name": "median",
        "src_root": DATA_ROOT / "ltstdb_regression_median_100hz_balanced_min5_nocompress",
        "dst_root": DATA_ROOT / "ltstdb_regression_median_100hz_full16_nocompress",
    },
]


# ---------------------------------------------------------------------------
# Per-patient feature extraction (directly from source metadata)
# ---------------------------------------------------------------------------
def build_patient_features(src_csv: Path) -> pd.DataFrame:
    """
    For each patient, compute the segment counts and target stats we need:
      n_segments
      II_abnormal_count   (|target_lead0_mV| >= 0.1)
      V2_abnormal_count   (|target_lead1_mV| >= 0.1)
      II_target_std
      V2_target_std
    """
    md = pd.read_csv(src_csv)
    needed = ["patient_id", "target_lead0_mV", "target_lead1_mV"]
    md = md.dropna(subset=["target_lead0_mV", "target_lead1_mV"])[needed]

    rows = []
    for p, sub in md.groupby("patient_id"):
        ii = sub["target_lead0_mV"].to_numpy(dtype=np.float64)
        v2 = sub["target_lead1_mV"].to_numpy(dtype=np.float64)
        rows.append({
            "patient_id": p,
            "n_segments": int(len(sub)),
            "II_abn":     int((np.abs(ii) >= ABNORMAL_THR).sum()),
            "V2_abn":     int((np.abs(v2) >= ABNORMAL_THR).sum()),
            "II_std":     float(ii.std()),
            "V2_std":     float(v2.std()),
        })
    return pd.DataFrame(rows).set_index("patient_id").loc[ALL_PATIENTS]


# ---------------------------------------------------------------------------
# Stratification: exhaustive search over 10/3/3 with weighted objective
# ---------------------------------------------------------------------------
OMIT_PATIENTS = ("s20081", "s20091", "s20121", "s20221", "s20241")


def stratified_full16(
    feats: pd.DataFrame,
    lambda_var: float = 50.0,
    verbose: bool = False,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
    n_test: int = DEFAULT_N_TEST,
    force_omit_to_train: bool = False,
) -> Dict[str, int]:
    """
    Exhaustive search over (16 choose 10) * (6 choose 3) = 160,160 candidate
    assignments. Score each by:

      score = sum over (lead in {II, V2}, split in {TRAIN, VAL, TEST}) of
                | abnormal_frac_in_split[lead] - global_abnormal_frac[lead] |
            + lambda_var * sum over lead of
                ( max(0, val_std[lead] - train_std[lead])
                + max(0, test_std[lead] - train_std[lead]) )

    The first term enforces abnormal/normal balance across splits.
    The second penalizes splits where val or test has wider per-lead std
    than train. Variance-preference penalty kicks in only when train is
    narrower than val/test on a given lead (one-sided constraint).

    Returns: dict patient_id -> SPLIT_* enum value.
    """
    patients = list(feats.index)
    n = len(patients)
    assert n == 16
    if n_train + n_val + n_test != n:
        raise ValueError(f"split sizes must sum to {n}, got {n_train}+{n_val}+{n_test}={n_train+n_val+n_test}")

    n_seg = feats["n_segments"].to_numpy(dtype=np.float64)
    ii_abn = feats["II_abn"].to_numpy(dtype=np.float64)
    v2_abn = feats["V2_abn"].to_numpy(dtype=np.float64)
    ii_std = feats["II_std"].to_numpy(dtype=np.float64)
    v2_std = feats["V2_std"].to_numpy(dtype=np.float64)

    total_seg = n_seg.sum()
    global_ii_abn_frac = ii_abn.sum() / total_seg
    global_v2_abn_frac = v2_abn.sum() / total_seg

    idx_all = set(range(n))
    best_score = float("inf")
    best_assign = None
    best_breakdown = None

    # We approximate per-lead split std as the segment-count-weighted mean of
    # per-patient stds. This is not the exact pooled std, but it preserves
    # the comparison we care about (which split has wider per-lead variance).
    def weighted_std(idxs: list, std_vec: np.ndarray) -> float:
        w = n_seg[idxs]
        return float((std_vec[idxs] * w).sum() / max(w.sum(), 1.0))

    # Build the candidate set of train picks.
    if force_omit_to_train:
        omit_idx = sorted(patients.index(p) for p in OMIT_PATIENTS if p in patients)
        clean_idx = sorted(set(range(n)) - set(omit_idx))
        if len(omit_idx) > n_train:
            raise ValueError(f"--force_omit_to_train requires n_train >= {len(omit_idx)} (number of omit records)")
        remaining_train_slots = n_train - len(omit_idx)
        # Iterate: choose remaining_train_slots from clean records, then split val/test from the leftover
        train_combos = (tuple(sorted(set(omit_idx) | set(clean_combo)))
                        for clean_combo in combinations(clean_idx, remaining_train_slots))
    else:
        train_combos = combinations(range(n), n_train)

    for train_combo in train_combos:
        train_idx = list(train_combo)
        rest = sorted(idx_all - set(train_idx))
        for val_combo in combinations(rest, n_val):
            val_idx = list(val_combo)
            test_idx = [i for i in rest if i not in val_idx]

            # Segment-level abnormal-fraction deviation
            seg_dev = 0.0
            for idxs in (train_idx, val_idx, test_idx):
                total = n_seg[idxs].sum()
                if total == 0:
                    seg_dev += 2.0  # high penalty for empty split
                    continue
                ii_f = ii_abn[idxs].sum() / total
                v2_f = v2_abn[idxs].sum() / total
                seg_dev += abs(ii_f - global_ii_abn_frac) + abs(v2_f - global_v2_abn_frac)

            # Variance-preference penalty
            t_ii = weighted_std(train_idx, ii_std)
            t_v2 = weighted_std(train_idx, v2_std)
            v_ii = weighted_std(val_idx, ii_std)
            v_v2 = weighted_std(val_idx, v2_std)
            x_ii = weighted_std(test_idx, ii_std)
            x_v2 = weighted_std(test_idx, v2_std)
            var_pen = max(0.0, v_ii - t_ii) + max(0.0, x_ii - t_ii) \
                    + max(0.0, v_v2 - t_v2) + max(0.0, x_v2 - t_v2)

            score = seg_dev + lambda_var * var_pen

            if score < best_score:
                best_score = score
                best_assign = (tuple(train_idx), tuple(val_idx), tuple(test_idx))
                best_breakdown = (seg_dev, var_pen,
                                  {"TRAIN": (t_ii, t_v2),
                                   "VAL":   (v_ii, v_v2),
                                   "TEST":  (x_ii, x_v2)})

    if best_assign is None:
        raise RuntimeError("Exhaustive search yielded no candidates")
    train_idx, val_idx, test_idx = best_assign

    out: Dict[str, int] = {}
    for i in train_idx: out[patients[i]] = SPLIT_TRAIN
    for i in val_idx:   out[patients[i]] = SPLIT_VALIDATION
    for i in test_idx:  out[patients[i]] = SPLIT_TEST

    if verbose:
        seg_dev, var_pen, stds = best_breakdown
        print("\n[full16 segment_balanced] best split breakdown")
        print(f"  total score             = {best_score:.4f}")
        print(f"  abnormal-frac deviation = {seg_dev:.4f}")
        print(f"  variance penalty        = {var_pen:.5f} (* lambda {lambda_var} = {lambda_var * var_pen:.4f})")
        print(f"  global %II_abnormal     = {global_ii_abn_frac:.4f}")
        print(f"  global %V2_abnormal     = {global_v2_abn_frac:.4f}")
        # per-split %abnormal and per-lead std
        print(f"  {'split':<6}{'%II_abn':>10}{'%V2_abn':>10}{'II_std':>10}{'V2_std':>10}{'n_segs':>10}")
        for label, idxs in (("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)):
            total = int(n_seg[list(idxs)].sum())
            ii_f = ii_abn[list(idxs)].sum() / max(total, 1)
            v2_f = v2_abn[list(idxs)].sum() / max(total, 1)
            sii, sv2 = stds[label]
            print(f"  {label:<6}{ii_f:>10.4f}{v2_f:>10.4f}{sii:>10.4f}{sv2:>10.4f}{total:>10}")

    return out


# ---------------------------------------------------------------------------
# Dataset build (same pattern as prepare_clean11_dataset.py)
# ---------------------------------------------------------------------------
def build_variant(variant: dict, split_assignments: Dict[str, int]) -> dict:
    src_root = variant["src_root"]
    dst_root = variant["dst_root"]

    src_csv = src_root / "modified_dataset" / "modified_dataset.csv"
    src_h5 = src_root / "PatientSegmentRegressionDataset.h5"

    if not src_csv.exists():
        raise FileNotFoundError(src_csv)
    if not src_h5.exists():
        raise FileNotFoundError(src_h5)

    md = pd.read_csv(src_csv)
    md["_source_row"] = np.arange(len(md), dtype=np.int64)

    # All 16 patients are kept (this is the point of the full16 variant)
    keep_mask = md["patient_id"].isin(split_assignments.keys())
    kept = md[keep_mask].sort_values("_source_row").reset_index(drop=True)
    kept["strat_fold_annotated"] = kept["patient_id"].map(split_assignments).astype(float)

    out_meta_dir = dst_root / "modified_dataset"
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    out_md = kept.drop(columns=["_source_row"])
    out_md.to_csv(out_meta_dir / "modified_dataset.csv", index=False)

    src_indices = kept["_source_row"].to_numpy()
    n_kept = len(src_indices)
    out_h5_path = dst_root / "PatientSegmentRegressionDataset.h5"
    if out_h5_path.exists():
        out_h5_path.unlink()
    with h5py.File(src_h5, "r") as fin, h5py.File(out_h5_path, "w") as fout:
        src_ds = fin["tracings"]
        n_leads, n_samples = src_ds.shape[1], src_ds.shape[2]
        out_ds = fout.create_dataset(
            "tracings",
            shape=(n_kept, n_leads, n_samples),
            dtype=src_ds.dtype,
        )
        CHUNK = 1024
        for s in range(0, n_kept, CHUNK):
            idx = src_indices[s:s + CHUNK]
            out_ds[s:s + len(idx)] = src_ds[list(idx.tolist()), :, :]

    return {
        "n_segments": n_kept,
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "h5": str(out_h5_path),
        "metadata": str(out_meta_dir / "modified_dataset.csv"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambda_var", type=float, default=50.0,
                        help=("Weight on the variance-preference penalty "
                              "(default %(default)s). Higher = stronger "
                              "preference for train-widest distribution."))
    parser.add_argument("--n_train", type=int, default=DEFAULT_N_TRAIN,
                        help=f"Train patient count (default {DEFAULT_N_TRAIN}). Must sum to 16 with val+test.")
    parser.add_argument("--n_val", type=int, default=DEFAULT_N_VAL,
                        help=f"Val patient count (default {DEFAULT_N_VAL}).")
    parser.add_argument("--n_test", type=int, default=DEFAULT_N_TEST,
                        help=f"Test patient count (default {DEFAULT_N_TEST}).")
    parser.add_argument("--force_omit_to_train", action="store_true",
                        help=("Constrain the search so the 5 MANUAL_OMIT-affected records "
                              "(s20081, s20091, s20121, s20221, s20241) MUST land in TRAIN. "
                              "Val/test then come exclusively from the 11 'clean' records."))
    parser.add_argument("--output_suffix", type=str, default="",
                        help=("Optional suffix appended to output dataset directory names. "
                              "Example: --output_suffix=_12_2_2 produces "
                              "ltstdb_regression_<agg>_100hz_full16_12_2_2_nocompress/ "
                              "so different split-ratio builds don't clobber each other."))
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the split assignment then exit without writing datasets.")
    args = parser.parse_args(argv)

    # Apply output_suffix to dataset target paths.
    # Convention: insert the suffix immediately before "_nocompress" so the
    # naming pattern stays  ltstdb_regression_<agg>_100hz_full16<suffix>_nocompress
    if args.output_suffix:
        suf = args.output_suffix if args.output_suffix.startswith("_") else f"_{args.output_suffix}"
        for v in VARIANTS:
            old_name = v["dst_root"].name
            new_name = old_name.replace("_nocompress", f"{suf}_nocompress") if "_nocompress" in old_name else (old_name + suf)
            v["dst_root"] = v["dst_root"].parent / new_name

    # Build features from the source mean dataset metadata (II/V2 target stats
    # are essentially the same between mean and median variants; std differs
    # in the 3rd decimal place but doesn't change the search outcome).
    src_csv = VARIANTS[0]["src_root"] / "modified_dataset" / "modified_dataset.csv"
    if not src_csv.exists():
        raise FileNotFoundError(src_csv)

    feats = build_patient_features(src_csv)
    print(f"\n========== full-16 patient features ==========")
    print(feats.round(4).to_string())

    assignments = stratified_full16(feats, lambda_var=args.lambda_var, verbose=True,
                                     n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
                                     force_omit_to_train=args.force_omit_to_train)

    split_name = {SPLIT_TRAIN: "TRAIN", SPLIT_VALIDATION: "VAL", SPLIT_TEST: "TEST"}
    print("\n========== full-16 patient assignment ==========")
    print(f"Strategy: segment_balanced + train-widest-variance preference (lambda_var={args.lambda_var})")
    table = feats.copy()
    table["split"] = [split_name[assignments[p]] for p in table.index]
    table = table[["split"] + [c for c in table.columns if c != "split"]]
    print(table.to_string())
    counts = pd.Series([split_name[a] for a in assignments.values()]).value_counts()
    print(f"\nsplit counts: {counts.to_dict()}")

    if args.dry_run:
        print("\n--dry_run set: skipping dataset write.")
        return

    print("\n========== building full-16 datasets ==========")
    for v in VARIANTS:
        print(f"\n[{v['name']}] from: {v['src_root']}\n              to: {v['dst_root']}")
        info = build_variant(v, assignments)
        print(f"  -> wrote {info['n_segments']} segments")
        print(f"  -> H5:        {info['h5']}")
        print(f"  -> metadata:  {info['metadata']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
