"""
prepare_clean11_dataset.py

Build a "clean 11" variant of the LTSTDB regression datasets:
  * Drops the 5 records with any lead in MANUAL_OMIT:
        s20081, s20091, s20121, s20221, s20241
  * Restratifies the remaining 11 patients into 7/2/2 train/val/test using
    IterativeStratification on upstream physiological features derived from
    the per-segment .16a-based distribution analysis (NOT from gs_per_beat.csv).
  * Produces parallel datasets for both `mean` and `median` target
    aggregators, each with its own rebuilt H5 file so the patchECG
    PatientSegmentRegressionDataset loader works unchanged.

Output (under <repo>/custom_datasets/):
    ltstdb_regression_mean_100hz_clean11_nocompress/
        PatientSegmentRegressionDataset.h5
        modified_dataset/modified_dataset.csv
    ltstdb_regression_median_100hz_clean11_nocompress/
        PatientSegmentRegressionDataset.h5
        modified_dataset/modified_dataset.csv

Usage:
    cd /home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection
    .venv/bin/python src/scripts/custom_dataset/prepare_clean11_dataset.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from skmultilearn.model_selection import IterativeStratification


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO = Path("/home/student/GIT/Robust-Automated-Cardiovascular-Arrhythmia-Detection")
DATA_ROOT = REPO / "custom_datasets"
DISTRIBUTION_DIR = Path(
    "/home/student/Desktop/Research Data/DATASET NK/Segment Data/regression_mean_median/distribution_analysis"
)

# Drop any record with at least one omitted lead.
OMIT_PATIENTS = {"s20081", "s20091", "s20121", "s20221", "s20241"}

# Split enum values must match src.core.datasets.ecg_interface.Split
SPLIT_TRAIN = 1
SPLIT_TEST = 2
SPLIT_VALIDATION = 3

SEED = 200  # matches the framework default

VARIANTS = [
    {
        "name": "mean",
        "src_root": DATA_ROOT / "ltstdb_regression_mean_100hz_balanced_min5_nocompress",
        "dst_root": DATA_ROOT / "ltstdb_regression_mean_100hz_clean11_nocompress",
        "distribution": DISTRIBUTION_DIR / "summary_mean.csv",
    },
    {
        "name": "median",
        "src_root": DATA_ROOT / "ltstdb_regression_median_100hz_balanced_min5_nocompress",
        "dst_root": DATA_ROOT / "ltstdb_regression_median_100hz_clean11_nocompress",
        "distribution": DISTRIBUTION_DIR / "summary_median.csv",
    },
]


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def build_patient_segment_counts(distribution_csv: Path, patients: List[str]) -> pd.DataFrame:
    """
    Per-patient segment counts by lead × category.

    Returns DataFrame indexed by patient_id with columns:
      II_dep, II_normal, II_elev   (lead 0)
      V2_dep, V2_normal, V2_elev   (lead 1)
      n_segments
    """
    df = pd.read_csv(distribution_csv).set_index("record")
    df = df.loc[patients]
    out = pd.DataFrame({
        "II_dep":     df["lead0_n_dep_le_-0.1"].astype(int),
        "II_normal":  df["lead0_n_normal"].astype(int),
        "II_elev":    df["lead0_n_elev_ge_0.1"].astype(int),
        "V2_dep":     df["lead1_n_dep_le_-0.1"].astype(int),
        "V2_normal":  df["lead1_n_normal"].astype(int),
        "V2_elev":    df["lead1_n_elev_ge_0.1"].astype(int),
        "n_segments": df["n_usable_segments"].astype(int),
    }, index=patients)
    return out


def build_patient_features(distribution_csv: Path, patients: List[str]) -> pd.DataFrame:
    """
    Per-patient binary features built ONLY from upstream .16a-derived stats:
      - dep_II      : >=20% of segments have ST <= -0.1 mV on lead II
      - elev_V2     : >=50% of segments have ST >= +0.1 mV on lead V2
      - hi_II_std   : lead II target std > 0.05 mV
      - hi_V2_std   : lead V2 target std > 0.025 mV
    These are signal/target-distribution features, not gs_per_beat outputs.
    """
    df = pd.read_csv(distribution_csv).set_index("record")
    df = df.loc[patients]  # keep only the clean-11 records (and reorder)

    feats = pd.DataFrame(
        {
            "dep_II":    (df["lead0_frac_dep_le_-0.1"] > 0.2).astype(int),
            "elev_V2":   (df["lead1_frac_elev_ge_0.1"] > 0.5).astype(int),
            "hi_II_std": (df["lead0_std"] > 0.05).astype(int),
            "hi_V2_std": (df["lead1_std"] > 0.025).astype(int),
            # the raw numbers also help the human reader to sanity check
            "_lead0_std":            df["lead0_std"].round(4),
            "_lead1_std":            df["lead1_std"].round(4),
            "_lead0_frac_dep":       df["lead0_frac_dep_le_-0.1"].round(3),
            "_lead1_frac_elev_ge_0.1": df["lead1_frac_elev_ge_0.1"].round(3),
        },
        index=patients,
    )
    return feats


def stratified_7_2_2_segment_balanced(
    features: pd.DataFrame,
    seg_counts: pd.DataFrame,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Exhaustive-search assignment of 11 patients to 7 train / 2 val / 2 test
    that minimizes the deviation of segment-level abnormal-fractions per lead
    from their global values.

    Definitions:
        II abnormal = (II target <= -0.1) or (II target >= +0.1)
        V2 abnormal = (V2 target <= -0.1) or (V2 target >= +0.1)
    For each lead, abnormal + normal = n_segments. Balancing one fraction per
    lead implicitly balances the other.

    Since lead II is essentially depression-or-normal in this cohort and lead
    V2 is essentially elevation-or-normal, the two-bucket-per-lead view is
    equivalent to a three-bucket dep/normal/elev view (verified empirically).

    `features`   : unused except for the patient order; kept for signature
                   compatibility with the patient_binary strategy.
    `seg_counts` : per-patient segment counts (output of
                   build_patient_segment_counts).
    """
    from itertools import combinations

    patients = list(seg_counts.index)
    n = len(patients)

    # Per-patient: [II_abnormal, V2_abnormal, n_segments]
    II_abn = (seg_counts["II_dep"] + seg_counts["II_elev"]).to_numpy(dtype=float)
    V2_abn = (seg_counts["V2_dep"] + seg_counts["V2_elev"]).to_numpy(dtype=float)
    n_seg  = seg_counts["n_segments"].to_numpy(dtype=float)

    global_II_abn_frac = II_abn.sum() / n_seg.sum()
    global_V2_abn_frac = V2_abn.sum() / n_seg.sum()

    idx_all = set(range(n))
    best_score = float("inf")
    best_assignment = None
    best_breakdown = None

    for train_combo in combinations(range(n), 7):
        train_idx = list(train_combo)
        rest_idx_full = sorted(idx_all - set(train_idx))
        for val_combo in combinations(rest_idx_full, 2):
            val_idx = list(val_combo)
            test_idx = [i for i in rest_idx_full if i not in val_idx]

            split_breakdown = {}
            score = 0.0
            for label, idxs in (("TRAIN", train_idx), ("VAL", val_idx), ("TEST", test_idx)):
                total = n_seg[idxs].sum()
                ii_f  = II_abn[idxs].sum() / max(total, 1.0)
                v2_f  = V2_abn[idxs].sum() / max(total, 1.0)
                split_breakdown[label] = (ii_f, v2_f, int(total))
                score += abs(ii_f - global_II_abn_frac) + abs(v2_f - global_V2_abn_frac)

            if score < best_score:
                best_score = score
                best_assignment = (train_idx, val_idx, test_idx)
                best_breakdown = split_breakdown

    if best_assignment is None:
        raise RuntimeError("Exhaustive search produced no candidate")

    train_idx, val_idx, test_idx = best_assignment
    out: Dict[str, int] = {}
    for i in train_idx: out[patients[i]] = SPLIT_TRAIN
    for i in val_idx:   out[patients[i]] = SPLIT_VALIDATION
    for i in test_idx:  out[patients[i]] = SPLIT_TEST

    if verbose:
        print("\n[segment_balanced] best split breakdown")
        print(f"  total abs-deviation     = {best_score:.4f}")
        print(f"  global %II_abnormal     = {global_II_abn_frac:.4f}")
        print(f"  global %V2_abnormal     = {global_V2_abn_frac:.4f}")
        print(f"  {'split':<6}{'%II_abn':>10}{'%V2_abn':>10}{'n_segments':>14}")
        for k, (ii_f, v2_f, total) in best_breakdown.items():
            print(f"  {k:<6}{ii_f:>10.4f}{v2_f:>10.4f}{total:>14}")

    return out


def rest_idx_to_2_2_fallback(features: pd.DataFrame, rest_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic 2/2 split of the 4 'rest' patients if IterativeStratification
    fails to give 2/2. We pick the 2 patients with maximum hamming distance
    to each other for val; the remaining 2 go to test.
    """
    label_cols = [c for c in features.columns if not c.startswith("_")]
    Y = features[label_cols].values
    best = None
    best_pair = None
    rest = rest_idx.tolist()
    for i in range(len(rest)):
        for j in range(i + 1, len(rest)):
            d = int(np.sum(np.abs(Y[rest[i]] - Y[rest[j]])))
            if best is None or d > best:
                best = d
                best_pair = (rest[i], rest[j])
    val_rel = np.array([rest.index(best_pair[0]), rest.index(best_pair[1])])
    test_rel = np.array([k for k in range(len(rest)) if k not in val_rel])
    return val_rel, test_rel


def stratified_7_2_2(features: pd.DataFrame, seed: int = SEED) -> Dict[str, int]:
    """
    Multi-label iterative stratification of 11 patients into 7 train, 2 val,
    2 test. Uses binary feature columns only (drops the `_*` raw-number cols).
    """
    np.random.seed(seed)

    label_cols = [c for c in features.columns if not c.startswith("_")]
    Y = features[label_cols].values
    n = len(features)
    if n != 11:
        raise ValueError(f"Expected exactly 11 patients, got {n}")

    # Step 1: 7 train vs 4 rest. IterativeStratification returns the indices
    # for fold 0 first (whose size matches `sample_distribution_per_fold[0]`).
    strat1 = IterativeStratification(
        n_splits=2, order=1,
        sample_distribution_per_fold=[7 / 11, 4 / 11],
    )
    train_idx, rest_idx = next(strat1.split(np.zeros((n, 1)), Y))
    # Defensive: tolerate the algorithm flipping orders for tight ratios
    if len(train_idx) < len(rest_idx):
        train_idx, rest_idx = rest_idx, train_idx
    assert len(train_idx) == 7 and len(rest_idx) == 4, (
        f"expected 7/4 split, got {len(train_idx)}/{len(rest_idx)}"
    )

    # Step 2: 2 val vs 2 test from the 4-patient `rest`
    Y_rest = Y[rest_idx]
    strat2 = IterativeStratification(
        n_splits=2, order=1,
        sample_distribution_per_fold=[0.5, 0.5],
    )
    val_rel, test_rel = next(strat2.split(np.zeros((len(rest_idx), 1)), Y_rest))
    if len(val_rel) != 2 or len(test_rel) != 2:
        # Reproducible deterministic fallback if iterative-strat gives 1/3
        val_rel, test_rel = rest_idx_to_2_2_fallback(features, rest_idx)
    assert len(val_rel) == 2 and len(test_rel) == 2

    patients = features.index.values
    out: Dict[str, int] = {}
    for i in train_idx:
        out[patients[i]] = SPLIT_TRAIN
    for j in val_rel:
        out[patients[rest_idx[j]]] = SPLIT_VALIDATION
    for j in test_rel:
        out[patients[rest_idx[j]]] = SPLIT_TEST
    return out


# ---------------------------------------------------------------------------
# Dataset build
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

    keep_mask = md["patient_id"].isin(split_assignments.keys())
    kept = md[keep_mask].sort_values("_source_row").reset_index(drop=True)
    kept["strat_fold_annotated"] = kept["patient_id"].map(split_assignments).astype(float)

    # Output directory + metadata CSV
    out_meta_dir = dst_root / "modified_dataset"
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    out_md = kept.drop(columns=["_source_row"])
    out_md.to_csv(out_meta_dir / "modified_dataset.csv", index=False)

    # Output H5 (copy only the kept rows in metadata order so row position == new H5 index)
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
        # Copy in chunks of 1024 to limit peak memory
        CHUNK = 1024
        for s in range(0, n_kept, CHUNK):
            idx = src_indices[s:s + CHUNK]
            # h5py fancy indexing requires sorted unique indices; src order
            # is already sorted because we sorted `kept` by _source_row.
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
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for stratification (default %(default)s).")
    parser.add_argument("--strategy", choices=["segment_balanced", "patient_binary"],
                        default="segment_balanced",
                        help=("Stratification strategy. "
                              "'segment_balanced' (default): exhaustive search over "
                              "1980 7/2/2 assignments, minimize per-split deviation "
                              "of the abnormal-segment fraction on each lead from "
                              "the global value. "
                              "'patient_binary': skmultilearn IterativeStratification "
                              "on 4 binary patient features (original behaviour)."))
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the split assignment then exit without writing datasets.")
    args = parser.parse_args(argv)

    # ----- 1) determine clean-11 patient set from the source metadata -----
    src_md = pd.read_csv(VARIANTS[0]["src_root"] / "modified_dataset" / "modified_dataset.csv")
    all_patients = sorted(src_md["patient_id"].unique().tolist())
    clean_patients = sorted(p for p in all_patients if p not in OMIT_PATIENTS)
    if len(clean_patients) != 11:
        print(f"WARNING: expected 11 clean patients, got {len(clean_patients)}: {clean_patients}",
              file=sys.stderr)

    # ----- 2) build upstream features per patient -----
    # Using the 'mean' distribution CSV (the lead0/lead1 stats are essentially
    # the same in the two CSVs except for the std due to mean-vs-median).
    feats = build_patient_features(VARIANTS[0]["distribution"], clean_patients)
    seg_counts = build_patient_segment_counts(VARIANTS[0]["distribution"], clean_patients)

    # ----- 3) stratify -----
    if args.strategy == "segment_balanced":
        assignments = stratified_7_2_2_segment_balanced(
            features=feats,
            seg_counts=seg_counts,
            verbose=True,
        )
    else:
        assignments = stratified_7_2_2(feats, seed=args.seed)

    split_name = {SPLIT_TRAIN: "TRAIN", SPLIT_VALIDATION: "VAL", SPLIT_TEST: "TEST"}
    print("\n========== clean-11 patient split ==========")
    print(f"Strategy: {args.strategy}")
    if args.strategy == "patient_binary":
        print(f"Stratification seed: {args.seed}")
    print(f"Excluded patients (any-lead omit): {sorted(OMIT_PATIENTS)}\n")
    table = feats.copy()
    table["split"] = [split_name[assignments[p]] for p in table.index]
    table["n_seg"] = seg_counts.loc[table.index, "n_segments"]
    table = table[["split", "n_seg"] + [c for c in table.columns if c not in ("split", "n_seg")]]
    print(table.to_string())
    counts = pd.Series([split_name[a] for a in assignments.values()]).value_counts()
    print("\nsplit counts:", counts.to_dict())

    if args.dry_run:
        print("\n--dry_run set: skipping dataset write.")
        return

    # ----- 4) build the two clean-11 datasets -----
    print("\n========== building clean11 datasets ==========")
    for v in VARIANTS:
        print(f"\n[{v['name']}] from: {v['src_root']}\n             to: {v['dst_root']}")
        info = build_variant(v, assignments)
        print(f"  -> wrote {info['n_segments']} segments")
        print(f"  -> H5:        {info['h5']}")
        print(f"  -> metadata:  {info['metadata']}")
    print("\nDone.")


if __name__ == "__main__":
    main()
