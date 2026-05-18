import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.datasets.ecg_interface import Split
from src.scripts.custom_dataset.prepare_binary_datasets import (
    SampleRecord,
    build_single_lead_records,
    build_two_lead_records,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert custom binary ECG CSV folders into repo-compatible H5 + metadata datasets with a simple sample-level split."
    )

    parser.add_argument("--single-lead-root", type=str, default=None,
                        help="Root for the single-lead task. Expected layout: root/patient_id/lead0/*.csv and root/patient_id/lead1/*.csv")
    parser.add_argument("--two-lead-root", type=str, default=None,
                        help="Root for the two-lead task. Expected layout: root/patient_id/*.csv where each file contains both leads.")
    parser.add_argument("--single-output-root", type=str, default=None,
                        help="Output root for the single-lead dataset.")
    parser.add_argument("--two-output-root", type=str, default=None,
                        help="Output root for the two-lead dataset.")
    parser.add_argument("--single-lead-separate-by-source-lead", action=argparse.BooleanOptionalAction, default=False,
                        help="If set, write one single-lead dataset per source lead folder (for example lead0 and lead1).")

    parser.add_argument("--source-freq", type=int, default=250,
                        help="Sampling rate of the source CSV signals.")
    parser.add_argument("--target-freq", type=int, default=100,
                        help="Sampling rate of the generated dataset.")
    parser.add_argument("--segment-seconds", type=int, default=10,
                        help="Expected segment length in seconds.")

    parser.add_argument("--signal-start-row", type=int, default=0,
                        help="Index of the first row that contains signal samples.")
    parser.add_argument("--label-source", type=str, choices=["filename", "csv", "auto"], default="filename",
                        help="Where to read labels from.")
    parser.add_argument("--single-label-column", type=int, default=1,
                        help="Label column for single-lead CSVs when --label-source is csv or auto.")
    parser.add_argument("--two-label-column", type=int, default=2,
                        help="Label column for two-lead CSVs when --label-source is csv or auto.")
    parser.add_argument("--single-signal-column", type=int, default=0,
                        help="Signal column index for the single-lead task.")
    parser.add_argument("--two-signal-columns", nargs=2, type=int, default=[0, 1],
                        help="Signal column indices for the two-lead task.")

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--compression", type=str, default="gzip",
                        help="Compression algorithm for the output H5 dataset. Use 'none' to disable compression.")

    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, received {ratio_sum}")

    if not args.single_lead_root and not args.two_lead_root:
        raise ValueError("Provide at least one of --single-lead-root or --two-lead-root.")

    return args


def split_indices(labels: np.ndarray, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int) -> dict[int, int]:
    n_samples = len(labels)
    indices = np.arange(n_samples)
    temp_ratio = valid_ratio + test_ratio

    if temp_ratio == 0:
        return {int(idx): Split.TRAIN.value for idx in indices}

    try:
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices,
            labels,
            test_size=temp_ratio,
            stratify=labels,
            random_state=seed,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=temp_ratio,
            random_state=seed,
        )
        temp_labels = None

    assignments = {int(idx): Split.TRAIN.value for idx in train_idx}

    if valid_ratio == 0:
        for idx in temp_idx:
            assignments[int(idx)] = Split.TEST.value
        return assignments

    if test_ratio == 0:
        for idx in temp_idx:
            assignments[int(idx)] = Split.VALIDATION.value
        return assignments

    test_fraction_within_temp = test_ratio / temp_ratio

    try:
        if temp_labels is not None:
            valid_idx, test_idx = train_test_split(
                temp_idx,
                test_size=test_fraction_within_temp,
                stratify=temp_labels,
                random_state=seed,
            )
        else:
            valid_idx, test_idx = train_test_split(
                temp_idx,
                test_size=test_fraction_within_temp,
                random_state=seed,
            )
    except ValueError:
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_fraction_within_temp,
            random_state=seed,
        )

    for idx in valid_idx:
        assignments[int(idx)] = Split.VALIDATION.value
    for idx in test_idx:
        assignments[int(idx)] = Split.TEST.value

    return assignments


def write_dataset(records: list[SampleRecord], output_root: Path, h5_name: str, args):
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "modified_dataset"
    metadata_root.mkdir(parents=True, exist_ok=True)

    records = sorted(records, key=lambda record: (record.patient_id, record.filename, record.source_path))
    labels = np.asarray([record.label for record in records])
    assignments = split_indices(
        labels=labels,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    metadata_rows = []
    for index, record in enumerate(records):
        metadata_rows.append(
            {
                "label": str([record.label]),
                "filename": record.filename,
                "patient_id": record.patient_id,
                "segment_id": record.segment_id,
                "source_path": record.source_path,
                "source_lead": record.source_lead,
                "strat_fold_annotated": assignments[index],
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(metadata_root / "modified_dataset.csv", index=False)

    compression = None if args.compression.lower() == "none" else args.compression
    tracing_shape = (len(records),) + records[0].tracing.shape
    with h5py.File(output_root / h5_name, "w") as h5_file:
        dataset = h5_file.create_dataset(
            "tracings",
            shape=tracing_shape,
            dtype=np.float32,
            compression=compression,
        )
        for index, record in enumerate(records):
            dataset[index, :, :] = record.tracing

    split_counts = metadata.groupby("strat_fold_annotated").size().to_dict()
    print(f"\nWrote {len(records)} samples to {output_root}")
    print(f"H5 file: {output_root / h5_name}")
    print(f"Metadata: {metadata_root / 'modified_dataset.csv'}")
    print(f"Split counts by sample: {split_counts}")
    print("Sample-level split mapping:")
    print(f"  train={Split.TRAIN.value}, validation={Split.VALIDATION.value}, test={Split.TEST.value}")
    print("This simple script does not prevent patient leakage across train/validation/test.")


def write_single_lead_outputs(records: list[SampleRecord], output_root: Path, args):
    if not args.single_lead_separate_by_source_lead:
        write_dataset(
            records=records,
            output_root=output_root,
            h5_name="SingleLeadBinaryDataset.h5",
            args=args,
        )
        return

    grouped_records: dict[str, list[SampleRecord]] = {}
    for record in records:
        lead_name = record.source_lead or "unknown_lead"
        grouped_records.setdefault(lead_name, []).append(record)

    print("\nWriting separate single-lead datasets by source lead:")
    for lead_name, lead_records in sorted(grouped_records.items()):
        print(f"  {lead_name}: {len(lead_records)} samples")
        write_dataset(
            records=lead_records,
            output_root=output_root / lead_name,
            h5_name="SingleLeadBinaryDataset.h5",
            args=args,
        )


def main():
    args = parse_args()

    if args.single_lead_root:
        if not args.single_output_root:
            raise ValueError("--single-output-root is required when --single-lead-root is provided.")
        single_records = build_single_lead_records(root=Path(args.single_lead_root), args=args)
        if not single_records:
            raise ValueError(f"No CSV files were found under {args.single_lead_root}")
        write_single_lead_outputs(
            records=single_records,
            output_root=Path(args.single_output_root),
            args=args,
        )

    if args.two_lead_root:
        if not args.two_output_root:
            raise ValueError("--two-output-root is required when --two-lead-root is provided.")
        two_lead_records = build_two_lead_records(root=Path(args.two_lead_root), args=args)
        if not two_lead_records:
            raise ValueError(f"No CSV files were found under {args.two_lead_root}")
        write_dataset(
            records=two_lead_records,
            output_root=Path(args.two_output_root),
            h5_name="TwoLeadBinaryDataset.h5",
            args=args,
        )


if __name__ == "__main__":
    main()
