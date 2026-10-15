"""
prepare_fold_dataset.py

Build a "single-fold" LTSTDB regression dataset by consuming a patient-manifest
CSV (no exhaustive split search). Designed for k-fold CV setup.

For each fold we want:
  * Same H5 of segment tracings as the existing full16 dataset
  * A NEW modified_dataset.csv where `strat_fold_annotated` matches the manifest's
    train / val / test assignment for that fold

Implementation:
  * Source: an existing full16 H5 + modified_dataset.csv (already contains all 16
    patients, all segments).
  * Destination: a fold dir that contains
      - PatientSegmentRegressionDataset.h5     (hard-linked to source; no copy)
      - modified_dataset/modified_dataset.csv  (per-fold split labels)

Why hard-link the H5: each fold is just a different label assignment over the
same segment population. The H5 is ~1 GB and identical across folds, so we save
~4 GB and avoid duplicate I/O at staging time.

Usage:
    python prepare_fold_dataset.py \
        --manifest /path/to/fold01_manifest.csv \
        --src_root /scratch/youss99/ltstdb_datasets/ltstdb_regression_median_100hz_full16_12_2_2_omit_train_nocompress \
        --dst_root /scratch/youss99/ltstdb_datasets/cv5_fold01

The manifest CSV must have columns: patient_id, split (values: train|val|test).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Split enum values must match src.core.datasets.ecg_interface.Split
# (same as prepare_full16_dataset.py)
SPLIT_TRAIN = 1
SPLIT_TEST = 2
SPLIT_VALIDATION = 3

SPLIT_LABEL_MAP = {
    "train": SPLIT_TRAIN,
    "val":   SPLIT_VALIDATION,
    "valid": SPLIT_VALIDATION,
    "validation": SPLIT_VALIDATION,
    "test":  SPLIT_TEST,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV with columns patient_id,split (split in {train,val,test}).")
    ap.add_argument("--src_root", type=Path, required=True,
                    help="Existing full16-style dataset root containing modified_dataset/modified_dataset.csv and PatientSegmentRegressionDataset.h5.")
    ap.add_argument("--dst_root", type=Path, required=True,
                    help="Destination dataset root for this fold (will be created).")
    ap.add_argument("--copy_h5", action="store_true",
                    help="Copy the H5 instead of hard-linking. Use if src and dst are on different filesystems.")
    args = ap.parse_args(argv)

    src_csv = args.src_root / "modified_dataset" / "modified_dataset.csv"
    src_h5  = args.src_root / "PatientSegmentRegressionDataset.h5"
    if not src_csv.exists():
        print(f"ERROR: src CSV missing: {src_csv}", file=sys.stderr); return 2
    if not src_h5.exists():
        print(f"ERROR: src H5 missing: {src_h5}", file=sys.stderr); return 2
    if not args.manifest.exists():
        print(f"ERROR: manifest missing: {args.manifest}", file=sys.stderr); return 2

    # Load manifest -> {patient_id: split_enum}
    mani = pd.read_csv(args.manifest)
    required_cols = {"patient_id", "split"}
    if not required_cols.issubset(mani.columns):
        print(f"ERROR: manifest must have columns {required_cols}, got {list(mani.columns)}", file=sys.stderr)
        return 2
    split_assignments = {}
    for _, row in mani.iterrows():
        s = str(row["split"]).strip().lower()
        if s not in SPLIT_LABEL_MAP:
            print(f"ERROR: unrecognized split '{row['split']}' for patient {row['patient_id']}", file=sys.stderr)
            return 2
        split_assignments[str(row["patient_id"])] = SPLIT_LABEL_MAP[s]

    # Read src metadata, rewrite strat_fold_annotated per manifest, validate coverage
    md = pd.read_csv(src_csv)
    src_patients = set(md["patient_id"].unique())
    mani_patients = set(split_assignments.keys())
    missing = src_patients - mani_patients
    extra = mani_patients - src_patients
    if missing:
        print(f"ERROR: manifest does not assign split for source patients: {sorted(missing)}", file=sys.stderr)
        return 2
    if extra:
        print(f"WARN: manifest references patients not in source: {sorted(extra)} (ignored)", file=sys.stderr)

    md["strat_fold_annotated"] = md["patient_id"].map(split_assignments).astype(float)

    # Build dst layout
    dst_meta_dir = args.dst_root / "modified_dataset"
    dst_meta_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_meta_dir / "modified_dataset.csv"
    dst_h5 = args.dst_root / "PatientSegmentRegressionDataset.h5"

    md.to_csv(dst_csv, index=False)

    # Replace any existing dst H5 (link or file) cleanly
    if dst_h5.exists() or dst_h5.is_symlink():
        dst_h5.unlink()
    if args.copy_h5:
        import shutil
        shutil.copy2(src_h5, dst_h5)
    else:
        # Try hard link first; fall back to symlink across filesystems.
        try:
            os.link(src_h5, dst_h5)
        except OSError as exc:
            print(f"NOTE: hard-link failed ({exc}); falling back to symlink.", file=sys.stderr)
            os.symlink(src_h5.resolve(), dst_h5)

    # Summary
    print(f"Wrote: {dst_csv}")
    print(f"H5:    {dst_h5}  ->  {os.readlink(dst_h5) if dst_h5.is_symlink() else 'hardlink/copy'}")
    print()
    print("Per-split breakdown:")
    print(f"  {'split':<14}{'n_patients':>12}{'n_segments':>12}  patients")
    for split_val, split_name in [(SPLIT_TRAIN, "train"), (SPLIT_VALIDATION, "val"), (SPLIT_TEST, "test")]:
        sub = md[md["strat_fold_annotated"] == float(split_val)]
        pats = sorted(sub["patient_id"].unique())
        print(f"  {split_name:<14}{len(pats):>12}{len(sub):>12}  {pats}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
